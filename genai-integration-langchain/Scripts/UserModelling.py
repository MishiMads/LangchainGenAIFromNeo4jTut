import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# --- LangChain Imports ---
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Neo4jVector
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# --- Helper Function to Log Interactions ---

def log_interaction(driver, user_id, question, answer):
    """
    Logs a user's question and the RAG chain's answer to Neo4j.
    """
    query = """
    // Find or create the user
    MERGE (u:User {userId: $user_id})

    // Create the new question and answer nodes
    CREATE (q:Question {text: $question, timestamp: datetime()})
    CREATE (a:Answer {text: $answer, timestamp: datetime()})

    // Connect the user to their question, and the question to its answer
    MERGE (u)-[:ASKED {timestamp: datetime()}]->(q)
    MERGE (q)-[:RECEIVED_ANSWER]->(a)
    """
    try:
        with driver.session() as session:
            session.run(query, user_id=user_id, question=question, answer=answer)
    except Exception as e:
        print(f"Error logging interaction: {e}")

# --- 1. Initialize the LLM for Generation ---
print("Initializing local LLM via Ollama...")
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

# --- 2. Initialize the Embedding Model for Retrieval ---
print("Initializing Hugging Face embedding model...")
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_kwargs = {'device': 'cuda'}  # Use GPU (change to 'cpu' if needed)
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs
)

# --- 3. Connect to Neo4j (Both Driver and Vector Store) ---

# !!! --- ADD THIS DEBUG LINE --- !!!
print(f"--- DEBUG: Attempting to connect to database: '{os.getenv('NEO4J_DATABASE')}' ---")
# !!! --- END DEBUG LINE --- !!!

# A. Initialize the Neo4j Driver (for writing/reading history)
print("Connecting to Neo4j for writes/history...")

try:
    neo4j_driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        database = os.getenv("NEO4J_DATABASE")
    )
    neo4j_driver.verify_connectivity()
    print("Neo4j driver connected.")
except Exception as e:
    print(f"Failed to connect to Neo4j driver: {e}")
    exit()


# B. Connect to the Existing Neo4j Vector Index (for RAG)
print("Connecting to existing Neo4j vector index...")
try:
    vector_store = Neo4jVector.from_existing_index(
        embedding=embedding_model,
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
        index_name="Math",  # <-- MUST match Script 1

        # --- ADD THESE MISSING LINES ---
        node_label="MathChunks",  # <-- This was the main problem
        embedding_node_property="embedding",  # <-- Add for completeness
        # ---

        text_node_property="text",  # <-- MUST match Script 1
    )

    # Create the base retriever (the part that gets relevant chunks, the default is 4 so it gets the 4 most relevant)
    vector_retriever = vector_store.as_retriever()
    print("Neo4j vector index connected successfully.")  # Changed message
except Exception as e:
    print(f"Failed to connect to Neo4j vector index: {e}")
    neo4j_driver.close()
    exit()

# --- 4. Create the PERSONALIZED RAG Chain ---
print("Creating the PERSONALIZED RAG chain...")

def get_personalized_context(input_dict):
    """
    Retrieves both general knowledge from vectors and specific user history.
    """
    question = input_dict["question"]
    user_id = input_dict["user_id"]

    # 1. Get vector context (general knowledge)
    vector_docs = vector_retriever.invoke(question)
    vector_context = "\n".join([doc.page_content for doc in vector_docs])

    # 2. Get user history context (from graph)
    history_context = ""
    history_query = """
    // Find the user and their last 3 interactions
    MATCH (u:User {userId: $user_id})-[:ASKED]->(q:Question)-[:RECEIVED_ANSWER]->(a:Answer)
    RETURN q.text AS question, a.text AS answer
    ORDER BY q.timestamp DESC
    LIMIT 3
    """
    try:
        with neo4j_driver.session() as session:
            results = session.run(history_query, user_id=user_id)
            past_interactions = [
                f"User previously asked: {r['question']}\nYou answered: {r['answer']}"
                for r in results
            ]
            if past_interactions:
                # Reverse to show oldest first in the prompt
                past_interactions.reverse()
                history_context = "--- User's Recent History (for context) ---\n" + "\n\n".join(past_interactions)
    except Exception as e:
        print(f"Warning: Could not fetch user history: {e}")
        # Continue without history if it fails

    # 3. Combine contexts
    final_context = f"{history_context}\n\n--- General Knowledge Context ---\n{vector_context}"
    # print(f"--- DEBUG: CONTEXT ---\n{final_context}\n--- END DEBUG ---") # Uncomment for debugging
    return final_context

# Create the new prompt template
template = """Answer the question based ONLY on the following context.
The context may contain 'General Knowledge' and 'User's Recent History'.
Use the user's history to understand what they are talking about.

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Build the new chain
rag_chain = (
    {
        "context": RunnableLambda(get_personalized_context),
        "question": lambda x: x["question"],
        "user_id": lambda x: x["user_id"] # Pass user_id through
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. Invoke the chain and Log Interactions ---

# Define our user
user_id = "Jakob"

# --- First Question ---
question_1 = "Hej jeg hedder Jakob, hvad for noget matematik skal jeg lære?"
print(f"\nInvoking chain for user '{user_id}' with question: {question_1}")

response_1 = rag_chain.invoke({"question": question_1, "user_id": user_id})

print("\n--- Final Answer 1 ---")
print(response_1)

# Log this interaction
print("Logging interaction 1 to Neo4j...")
log_interaction(neo4j_driver, user_id, question_1, response_1)

# --- Second (Follow-up) Question ---
question_2 = "Hvad var det nu, jeg spurgte om lige før?"
print(f"\nInvoking chain for user '{user_id}' with follow-up: {question_2}")

response_2 = rag_chain.invoke({"question": question_2, "user_id": user_id})

print("\n--- Final Answer 2 (Follow-up) ---")
print(response_2)
print("(This answer should use the history of the first question)")

# Log this interaction
print("Logging interaction 2 to Neo4j...")
log_interaction(neo4j_driver, user_id, question_2, response_2)

question_7 = "Hvad er mit navn?"
print(f"\nInvoking chain for user '{user_id}' with question: {question_7}")

response_7 = rag_chain.invoke({"question": question_7, "user_id": user_id})
print("\n--- Final Answer 7 (Follow-up) ---")
print(response_7)


# --- 6. Clean up ---
neo4j_driver.close()
print("\nNeo4j driver closed. Script finished.")