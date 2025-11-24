import os
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase

# --- LangChain Imports ---
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# Load environment variables (.env)
load_dotenv()

# --- CONFIGURATION ---
USER_ID = "Jakob_Student"
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")
NEO4J_DB = os.getenv("NEO4J_DATABASE")
LLM_MODEL = "llama3.1"

# *** STRICT FILTER LIST ***
# The system will ONLY track proficiency stats for these 4 topics.
# Anything else (like "strategies", "geometry", "chat") becomes "General".
VALID_MATH_TOPICS = ["Addition", "Subtraction", "Multiplication", "Division"]

# --- 1. CONNECT TO NEO4J ---
print(f"--- CONNECTING TO NEO4J: {NEO4J_DB} ---")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS), database=NEO4J_DB)

# --- 2. INITIALIZE AI MODELS ---
print("Initializing LLM and Embeddings...")
llm = ChatOllama(model=LLM_MODEL, temperature=0)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cuda'}  # Change to 'cpu' if no GPU available
)

# --- 3. CONNECT TO VECTOR STORE (TEXTBOOK KNOWLEDGE) ---
try:
    vector_store = Neo4jVector.from_existing_index(
        embedding=embedding_model,
        url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS, database=NEO4J_DB,
        index_name="Math",
        node_label="MathChunks",
        text_node_property="text",
    )
    vector_retriever = vector_store.as_retriever()
    print("Vector Store connected.")
except Exception as e:
    print(f"Vector Store Connection Failed: {e}")
    print("Ensure you have run the Ingestion Script first to create the 'Math' index.")
    exit()

# --- 4. DEFINE HELPER CHAINS ---

# A. TOPIC CLASSIFIER (Strict Mapping)
# We force the LLM to map "strategies" to the core operation if possible.
classifier_template = """
You are a classification tool. 
Categorize the interaction into ONE of these EXACT topics:
[Addition, Subtraction, Multiplication, Division, General]

RULES:
1. If the text mentions "strategies" (regnestrategier), map it to the underlying operation. 
   (e.g., "Addition strategies" -> "Addition").
2. If the user says "Hello", "Help", or non-math text -> "General".
3. If the text is ambiguous, choose the closest math topic or "General".

AI Question: {last_response}
User Input: {user_input}

Return ONLY the topic name (one word).
"""
classifier_chain = (
        ChatPromptTemplate.from_template(classifier_template)
        | llm
        | StrOutputParser()
)

# B. GRADER (Evalutes Correctness)
grader_template = """
You are a math teacher grading a student.
Context: The teacher asked "{history}".
The student answered "{input}".

Is the student correct?
Return ONLY the word "CORRECT" or "WRONG".
"""
grader_chain = (
        ChatPromptTemplate.from_template(grader_template)
        | llm
        | StrOutputParser()
)


# --- 5. DATABASE FUNCTIONS (The "Brain") ---

def ensure_core_topics(driver):
    """Ensures ONLY the 4 main topics exist as Subject nodes."""
    query = """
    UNWIND $topics AS topic
    MERGE (s:Subject {id: topic})
    """
    with driver.session() as session:
        # We create nodes for the Valid topics + General
        session.run(query, topics=VALID_MATH_TOPICS + ["General"])


def update_student_model(driver, user_id, subject_name, is_correct):
    """
    Updates the User's Proficiency Stats.
    **GATEKEEPER**: This function does NOTHING if the topic is not in VALID_MATH_TOPICS.
    """
    if subject_name not in VALID_MATH_TOPICS:
        # We do not track proficiency for "General" or "Regnestrategier"
        return

    query = """
    MATCH (u:User {userId: $user_id})
    MERGE (s:Subject {id: $subject_name}) 
    MERGE (u)-[r:LEARNS]->(s)
    ON CREATE SET r.correct_count=0, r.incorrect_count=0, r.total_attempts=0
    ON MATCH SET
        r.total_attempts = r.total_attempts + 1,
        r.last_interacted_at = datetime(),
        r.correct_count = r.correct_count + (CASE WHEN $is_correct THEN 1 ELSE 0 END),
        r.incorrect_count = r.incorrect_count + (CASE WHEN $is_correct THEN 0 ELSE 1 END),
        r.success_rate = toFloat(r.correct_count + (CASE WHEN $is_correct THEN 1 ELSE 0 END)) / (r.total_attempts + 1)
    """
    try:
        with driver.session() as session:
            session.run(query, user_id=user_id, subject_name=subject_name, is_correct=is_correct)
            status = "CORRECT" if is_correct else "WRONG"
            print(f"   [DB Update] User model updated for '{subject_name}': {status}")
    except Exception as e:
        print(f"Error updating model: {e}")


def log_interaction(user_id, question_text, answer_text, topic):
    """
    Saves chat history.
    If the classifier gave us a weird topic, we force it to 'General' here so the graph stays clean.
    """
    clean_topic = topic if topic in (VALID_MATH_TOPICS + ["General"]) else "General"

    query = """
    MERGE (u:User {userId: $uid})
    MERGE (s:Subject {id: $topic})
    CREATE (q:Question {text: $q_text, timestamp: datetime()})
    CREATE (a:Answer {text: $a_text, timestamp: datetime()})
    MERGE (u)-[:ASKED]->(q)
    MERGE (q)-[:RECEIVED_ANSWER]->(a)
    MERGE (q)-[:RELATES_TO]->(s)
    """
    with driver.session() as session:
        session.run(query, uid=user_id, q_text=question_text,
                    a_text=answer_text, topic=clean_topic)


def get_student_context(input_dict):
    user_id = input_dict["user_id"]
    question = input_dict["question"]

    # 1. Stats Query - ONLY fetches the Valid Topics
    stats_query = """
    MATCH (u:User {userId: $user_id})-[r:LEARNS]->(s:Subject)
    WHERE s.id IN $valid_topics
    RETURN s.id AS topic, coalesce(r.success_rate, 0.0) AS rate, r.correct_count AS c, r.incorrect_count AS w
    ORDER BY r.last_interacted_at DESC LIMIT 5
    """

    # 2. History Query - Fetches the very last question asked
    history_query = """
    MATCH (u:User {userId: $user_id})-[:ASKED]->(q:Question)
    RETURN q.text AS last_q ORDER BY q.timestamp DESC LIMIT 1
    """

    context_str = ""
    with driver.session() as session:
        stats = session.run(stats_query, user_id=user_id, valid_topics=VALID_MATH_TOPICS).data()
        if stats:
            context_str += "--- STUDENT HISTORY (PROFICIENCY) ---\n"
            for r in stats:
                context_str += f"Topic: {r['topic']} | Success Rate: {round(r['rate'] * 100)}% ({r['c']} Correct, {r['w']} Wrong)\n"

        last_q = session.run(history_query, user_id=user_id).data()
        if last_q:
            context_str += f"\n--- PREVIOUS QUESTION ASKED BY TUTOR ---\n{last_q[0]['last_q']}\n"

    # 3. Textbook Context (Vector Search)
    docs = vector_retriever.invoke(question)
    book_content = "\n".join([d.page_content for d in docs])

    return f"{context_str}\n\n--- TEXTBOOK KNOWLEDGE ---\n{book_content}"


# --- 6. MAIN RAG CHAIN ---

template = """
You are an expert math tutor for elementary students.
Topics: Addition, Subtraction, Multiplication, Division.

1. Analyze the Student History to adjust difficulty.
   - Low success rate? Be encouraging, give hints.
   - High success rate? Challenge them.
2. If the student answers briefly (e.g., "10"), look at the 'PREVIOUS QUESTION' to see if they are correct.

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
        {
            "context": RunnableLambda(get_student_context),
            "question": lambda x: x["question"],
            "user_id": lambda x: x["user_id"]
        }
        | prompt
        | llm
        | StrOutputParser()
)


# --- 7. EXECUTION LOOP ---

def main():
    # A. Setup Data
    with driver.session() as session:
        session.run("MERGE (u:User {userId: $uid})", uid=USER_ID)
    ensure_core_topics(driver)

    # B. Simulation (Seed Data for demo purposes)
    # We force feed some data so the user isn't starting from zero.
    print(f"\n--- SEEDING SIMULATION DATA FOR {USER_ID} ---")
    update_student_model(driver, USER_ID, "Multiplication", False)
    update_student_model(driver, USER_ID, "Multiplication", False)
    update_student_model(driver, USER_ID, "Addition", True)

    # C. Interactive Chat Loop
    print(f"\n\n--- MATH TUTOR SESSION STARTED ---")
    print(f"Tracking ONLY: {VALID_MATH_TOPICS}")
    print("Type 'q' to quit.")

    # Memory variables for the loop
    last_ai_response = None
    last_topic = "General"

    while True:
        try:
            user_input = input(f"\n{USER_ID}: ")
        except EOFError:
            break

        if user_input.lower() in ["q", "quit"]:
            break
        if not user_input.strip():
            continue

        # --- STEP 1: GRADING (Update Graph) ---
        # We only grade if the PREVIOUS topic was one of the 4 Math Topics.
        if last_ai_response and last_topic in VALID_MATH_TOPICS:
            print("   [System] Grading previous answer...")
            grade = grader_chain.invoke({"history": last_ai_response, "input": user_input})
            is_correct = ("CORRECT" in grade.strip().upper())

            # Update DB (Gatekeeper inside function will double check topic)
            update_student_model(driver, USER_ID, last_topic, is_correct)

        # --- STEP 2: GENERATE RESPONSE ---
        print("AI is thinking...")
        response = rag_chain.invoke({"question": user_input, "user_id": USER_ID})
        print(f"\nTutor: {response}")

        # --- STEP 3: CLASSIFY & LOG ---
        # Classify the NEW interaction so we know what to grade next time
        raw_topic = classifier_chain.invoke({"last_response": response, "user_input": user_input})

        # Clean topic string
        clean_topic = raw_topic.strip().replace(".", "").capitalize()

        # Final fallback: if LLM hallucinates a topic not in our list, call it General
        if clean_topic not in VALID_MATH_TOPICS:
            clean_topic = "General"

        log_interaction(USER_ID, user_input, response, clean_topic)

        # Update memory
        last_ai_response = response
        last_topic = clean_topic


if __name__ == "__main__":
    main()
    driver.close()
    print("Session Closed.")