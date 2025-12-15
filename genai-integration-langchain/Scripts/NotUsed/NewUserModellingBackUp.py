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
    # Initialize GPT-4o
    print("Initializing OpenAI GPT-4o...")
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is missing in .env file")
        exit()

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0
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
1. **Topic**: Identify the specific math topic (e.g., Addition, Algebra, Geometry).
2. **Intent**: Is the user ASKING a question or ANSWERING a question? (Return 'question' or 'answer').
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
    vector_docs = vector_retriever.invoke(question)
    vector_context = "\n".join([doc.page_content for doc in vector_docs])

    # 2. History Retrieval (Short Term Memory)
    history_context = ""
    history_query = """
    MATCH (u:User {userId: $user_id})-[:ASKED]->(q:Question)-[:RECEIVED_ANSWER]->(a:Answer)
    RETURN q.text AS question, a.text AS answer
    ORDER BY q.timestamp DESC LIMIT 3
    """
    with neo4j_driver.session() as session:
        results = session.run(history_query, user_id=user_id)
        past_interactions = [f"User: {r['question']}\nAI: {r['answer']}" for r in results]
        if past_interactions:
            history_context = "--- Recent Chat History ---\n" + "\n\n".join(reversed(past_interactions))

    return f"{history_context}\n\n--- Book Context ---\n{vector_context}"


rag_template = """
You are an expert math tutor, who wants to help elementary school pupils learn mathematics.
You want to give students math problems to solve, and help them when they get stuck. 
These math problems should be within the topics addition, subtraction, multiplication, division.

You want to help the students improve by providing scaffolding, hints, and explanations. 
As such, you should look at the context and see what the pupil has done so far (e.g. how many addition problems they have solved and how many they got right), and adjust the difficulty of the next problems accordingly.
Additionally, if the student has answered one type of 

If the pupil says a brief answer such as "Det er 5!", it is likely they are trying to answer a question you asked just before so look at your previous question.


Answer the question based ONLY on the following context.
Use the user's history to understand what they are referring to.

If the context does not contain the answer, continue answering the question but start the answer with: (The context did not provide the information needed.)

Context:
{context}

Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)

rag_chain = (
        {
            "context": RunnableLambda(get_personalized_context),
            "question": lambda x: x["question"],
            "user_id": lambda x: x["user_id"]
        }
        | rag_prompt
        | llm
        | StrOutputParser()
)


# --- 7. RUN FUNCTION ---

def run_chat_cycle(user_id, user_input):
    print(f"\n\n--- User '{user_id}': {user_input} ---")

    # Step 1: Analyze (What topic is this? Is it an answer?)
    print(" >> Analyzing input...")
    metadata = analyze_interaction(user_id, user_input)

    # Step 2: Generate Answer (RAG)
    print(" >> Generating response...")
    response = rag_chain.invoke({"question": user_input, "user_id": user_id})
    print(f"AI Response: {response}")

    # Step 3: Log everything to Neo4j
    print(" >> Logging to Graph...")
    log_interaction(neo4j_driver, user_id, user_input, response, metadata)


# --- 8. EXECUTION SCENARIOS ---

user_id = "Jakob"

# Scenario 1: Asking a general question
run_chat_cycle(user_id, "Svaret er 7!")

"""
run_chat_cycle(user_id, "Hvad er 10 plus 10?")

# Scenario 2: The user tries to answer a math problem
run_chat_cycle(user_id, "Det er 10.")

# Scenario 3: Asking about a specific strategy
run_chat_cycle(user_id, "Hvordan bruger man split metoden?")
"""

# Close connections
neo4j_driver.close()
print("\n--- Script Finished ---")