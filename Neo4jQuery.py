import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jVector, Neo4jGraph
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# --- Initialize the embedding model (needed to embed the user's query) ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cuda'}
)

# --- Connect to the EXISTING Neo4j Vector Index ---
# This is the key difference: we instantiate Neo4jVector directly
# instead of using 'from_documents'.
neo4j_vector = Neo4jVector(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    embedding=embedding_model,
    index_name="algebra", # The name of your existing index
    node_label="Chunk", # The label of the nodes in the index
    text_node_property="text", # The property containing the text
    embedding_node_property="embedding", # The property containing the embedding
)

# --- Ask a question ---
query = "Hvad er tidlig algebra?"
print(f"Searching for: '{query}'")


query = "Hvad skal jeg lære?"
print(f"Searching for: '{query}'")

result_with_scores = neo4j_vector.similarity_search_with_score(query, k=3)

print("\n--- Similarity Search Results ---")
for doc, score in result_with_scores:
    print(f"Similarity Score: {score:.4f}") # Formatted for readability
    print(f"Source Page: {doc.metadata.get('page', 'N/A')}")
    print(f"Text: {doc.page_content}\n")