import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase

# --- LangChain Imports ---
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.runnables import RunnableLambda
# CHANGED: We now import JsonOutputParser to handle messy LLM outputs automatically
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# ==========================================
# 0. CONFIGURATION: CHOOSE YOUR LLM
# ==========================================
# Options: "openai" or "local"
LLM_PROVIDER = "local"

print(f"--- SELECTED LLM PROVIDER: {LLM_PROVIDER.upper()} ---")

# --- 1. Initialize Models ---

llm = None

if LLM_PROVIDER == "openai":
    print("Initializing OpenAI GPT-5-mini...")
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is missing in .env file")
        exit()

    llm = ChatOpenAI(
        model="gpt-5-mini",
        model_kwargs={"temperature": 1}
    )
else:
    # Initialize Local Llama
    print("Initializing local LLM (Llama 3.1)...")
    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
        # Keep raw=True or format="json" can sometimes help, but the Parser below is best
    )

print("Initializing Embedding Model...")
# !!! CRITICAL !!!
# You MUST use the same embedding model used to create the database.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cuda'}  # Change to 'cpu' if you don't have CUDA
)

# --- 2. Connect to Neo4j ---
print("Connecting to Neo4j Database...")

# Driver for writing history and analysis
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    database=os.getenv("NEO4J_DATABASE")
)

# Vector Store for RAG retrieval
try:
    vector_store = Neo4jVector.from_existing_index(
        embedding=embedding_model,
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
        index_name="Math",
        node_label="MathChunks",
        embedding_node_property="embedding",
        text_node_property="text",
    )
    vector_retriever = vector_store.as_retriever()
    print("Neo4j Vector Index connected.")
except Exception as e:
    print(f"Failed to connect to Vector Index: {e}")
    exit()


# --- 3. HELPER: Get Last Context ---
def get_last_bot_message(user_id):
    """
    Retrieves the last answer the bot gave.
    """
    query = """
    MATCH (u:User {userId: $user_id})-[:ASKED]->(:Question)-[:RECEIVED_ANSWER]->(a:Answer)
    RETURN a.text AS last_answer
    ORDER BY a.timestamp DESC LIMIT 1
    """
    with neo4j_driver.session() as session:
        result = session.run(query, user_id=user_id).single()
        return result["last_answer"] if result else "No previous context."


# --- 4. THE ANALYZER CHAIN (Metadata Extractor) ---
# This runs in parallel to the chat to categorize what is happening.

analyzer_template = """
You are a Metadata Extractor for a Math Learning App.
Analyze the User Input based on the Previous Context.

User Input: {question}
Previous Bot Message: {last_context}

Tasks:
1. **Topic**: Identify the specific math topic ('Addition', 'Multiplication', 'Division', or 'Subtraction'). If the input does not fit any of these topics, return 'None'.
2. **Intent**: Is the user ASKING a question or ANSWERING a question? (Return 'question', 'answer', or 'Not applicable').
3. **Correctness**: If Intent is 'answer', is it mathematically correct based on the Previous Bot Message? (Return true, false, or null).

Return ONLY a JSON object in this format:
{{
  "topic": "TopicName",
  "intent": "question",
  "is_correct": null
}}
"""

analyzer_prompt = ChatPromptTemplate.from_template(analyzer_template)


# JsonOutputParser will automatically strip markdown and find the JSON object
analyzer_chain = analyzer_prompt | llm | JsonOutputParser()


def analyze_interaction(user_id, user_input):
    last_msg = get_last_bot_message(user_id)
    try:
        # Invoke LLM
        # Result is now ALREADY a Python dictionary, no need to json.loads()
        result = analyzer_chain.invoke({
            "question": user_input,
            "last_context": last_msg
        })

        return result
    except Exception as e:
        print(f"Analysis Error: {e}")
        # Debugging: Print what might have happened if possible
        return {"topic": "General", "intent": "question", "is_correct": None}


# --- 5. THE LOGGER (Using InteractionTopic) ---

def log_interaction(driver, user_id, user_text, bot_response, metadata):
    """
    Logs the interaction using the new separate schema:
    (:User)-[:ASKED]->(:Question)-[:CONCERNS_TOPIC]->(:InteractionTopic)
    """
    topic_name = metadata.get("topic", "General")
    intent = metadata.get("intent", "question")
    is_correct = metadata.get("is_correct")

    query = """
    MERGE (u:User {userId: $user_id})

    // 1. Handle the Topic - Using NEW Label 'InteractionTopic'
    MERGE (t:InteractionTopic {name: $topic_name})

    // 2. Create the Question Node with Analysis Properties
    CREATE (q:Question {
        text: $user_text, 
        timestamp: datetime(),
        intent: $intent,
        is_correct: $is_correct
    })

    // 3. Create the Answer Node
    CREATE (a:Answer {
        text: $bot_response, 
        timestamp: datetime()
    })

    // 4. Connect the User flow
    MERGE (u)-[:ASKED {timestamp: datetime()}]->(q)
    MERGE (q)-[:RECEIVED_ANSWER]->(a)

    // 5. Connect to the Interaction Topic
    MERGE (q)-[:CONCERNS_TOPIC]->(t)
    """

    try:
        with driver.session() as session:
            session.run(query,
                        user_id=user_id,
                        topic_name=topic_name,
                        user_text=user_text,
                        intent=intent,
                        is_correct=is_correct,
                        bot_response=bot_response)
        print(f" -> Logged to Neo4j. Topic: '{topic_name}' | Intent: '{intent}' | Correct: {is_correct}")
    except Exception as e:
        print(f"Logging Error: {e}")


# --- 6. THE RAG CHAIN (The Chatbot) ---

def get_personalized_context(input_dict):
    question = input_dict["question"]
    user_id = input_dict["user_id"]

    # 1. Vector Retrieval (General Knowledge)
    # (Behold din eksisterende kode her...)
    vector_docs = vector_retriever.invoke(question)
    vector_context = "\n".join([doc.page_content for doc in vector_docs])

    # 2. History Retrieval
    history_context = "INGEN TIDLIGERE SAMTALE."
    history_query = """
    MATCH (u:User {userId: $user_id})-[:ASKED]->(q:Question)-[:RECEIVED_ANSWER]->(a:Answer)
    RETURN q.text AS question, a.text AS answer
    ORDER BY q.timestamp DESC LIMIT 5
    """
    with neo4j_driver.session() as session:
        results = session.run(history_query, user_id=user_id)
        # Vi formaterer det som et script, så LLM forstår rækkefølgen
        past_interactions = [f"AI: {r['answer']}\nElev: {r['question']}" for r in results]
        if past_interactions:
            # Reversed, så det ældste kommer først (kronologisk rækkefølge er vigtig for LLM)
            history_context = "\n".join(reversed(past_interactions))

    return f"{history_context}"
    # BEMÆRK: Jeg har fjernet 'vector_context' fra return her midlertidigt,
    # hvis du kun vil fokusere på samtalen. Hvis du vil bruge bogen, så sæt den ind igen:
    # return f"FRA LÆREBOGEN:\n{vector_context}\n\nCHAT HISTORIK:\n{history_context}"


rag_template = """
### ROLLE
Du er en matematik-tutor for eleven {user_id}. Du taler dansk.

### NUVÆRENDE SITUATION
Her er de seneste beskeder i samtalen (Chat Historik):
{context}

### DATA OM FORRIGE KONTEKST
(Brug dette hvis historikken er tom eller uklar):
Du spurgte sidst om: "{last_context}"

### ELEVENS NYE BESKED:
"{question}"

"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)

rag_chain = (
        {
            "context": RunnableLambda(get_personalized_context),
            "question": lambda x: x["question"],
            "user_id": lambda x: x["user_id"],
            "last_context": lambda x: get_last_bot_message(x["user_id"])
        }
        | rag_prompt
        | llm
        | StrOutputParser()
)


# --- 7. CHAT CYCLE FUNCTION ---
def run_chat_cycle(user_id, user_input):
    """
    Processes a single chat interaction: analyze, generate response, and log.

    Args:
        user_id: The student identifier
        user_input: The text input from the user
    """
    try:
        # 1. Analyze the interaction
        metadata = analyze_interaction(user_id, user_input)

        # 2. Generate bot response using RAG
        bot_response = rag_chain.invoke({
            "question": user_input,
            "user_id": user_id
        })

        # 3. Log everything to Neo4j
        log_interaction(neo4j_driver, user_id, user_input, bot_response, metadata)

        print(f"Bot: {bot_response}")

    except Exception as e:
        print(f"Error in chat cycle: {e}")


# --- 8. BATCH TEST FUNCTION ---
def run_batch_test(student_name, question_groups, answers, operations):
    """
    Runs a batch of questions and answers for testing purposes.

    Args:
        student_name: The user_id to use
        question_groups: Nested list of questions by operation (4 groups of 5 questions)
        answers: Flat list of all 20 answers
        operations: List of 4 operation names
    """
    print(f"\n{'=' * 60}")
    print(f"BATCH TEST MODE - Student: {student_name}")
    print(f"{'=' * 60}\n")

    answer_counter = 0

    for op_index, operation in enumerate(operations):
        print(f"\n--- OPERATION: {operation} ---")

        for q_index, question in enumerate(question_groups[op_index]):
            student_answer = answers[answer_counter]

            print(f"\nQ{answer_counter + 1}: {question}")

            # First, bot asks the question
            run_chat_cycle(student_name, question)

            print(f"Student Answer: {student_answer}")

            # Then, student answers
            run_chat_cycle(student_name, student_answer)

            answer_counter += 1
            print(f"Progress: {answer_counter}/{len(answers)}")

    print(f"\n{'=' * 60}")
    print(f"BATCH TEST COMPLETED - {answer_counter} questions processed")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # --- CONFIGURATION: Choose mode ---
    RUN_MODE = "batch"  # Options: "interactive" or "batch"

    if RUN_MODE == "batch":
        # Test data from Test_User_Model_Simpel.py
        question_groups = [
            ["Hvad er 3+5?", "Hvad er 7+6?", "Hvad er 12+4?", "Hvad er 9+8?", "Hvad er 15+3?"],
            ["Hvad er 10 - 4?", "Hvad er 18 - 7?", "Hvad er 9 - 3?", "Hvad er 14 - 5?", "Hvad er 20 - 9?"],
            ["Hvad er 3*4?", "Hvad er 5*6?", "Hvad er 2*7?", "Hvad er 8*3?", "Hvad er 4*9?"],
            ["Hvad er 12/3?", "Hvad er 20/4?", "Hvad er 18/2?", "Hvad er 15/5?", "Hvad er 24/6?"]
        ]

        smartElev = [
            "8", "13", "16", "17", "18",
            "6", "12", "6", "9", "11",
            "12", "30", "14", "24", "36",
            "4", "5", "9", "4", "4"
        ]

        notSmartElev = [
            "6", "10", "13", "20", "11",
            "3", "10", "12", "10", "8",
            "9", "25", "14", "12", "24",
            "3", "6", "8", "3", "8"
        ]

        operations = ["Addition", "Subtraktion", "Multiplikation", "Division"]

        # Choose which student to test
        TEST_STUDENT = "SmartElev"  # Change to "NotSmartElev" for the other student

        answers = smartElev if TEST_STUDENT == "SmartElev" else notSmartElev

        run_batch_test(TEST_STUDENT, question_groups, answers, operations)

    else:  # interactive mode
        user_id = "Jakob"
        print(f"--- Chat Session Started for {user_id} ---")
        print("\nType 'exit', 'quit', or 'q' to end the session.\n")

        while True:
            try:
                user_input = input("You: ")

                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Ending session...")
                    break

                if not user_input.strip():
                    continue

                run_chat_cycle(user_id, user_input)

            except KeyboardInterrupt:
                print("\nInterrupted by user. Exiting...")
                break

    neo4j_driver.close()
    print("\n--- Script Finished ---")