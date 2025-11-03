import os
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jVector, Neo4jGraph

# Load environment variables from .env file
load_dotenv()

# --- 1. Connect to Neo4j Database ---
print("Connecting to Neo4j...")
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

# For a fresh start, you can uncomment the next line to delete all existing data
# print("Clearing the database...")
# graph.query("MATCH (n) DETACH DELETE n")

# --- 2. Load and Process the PDF Document ---

# --- 2. Load and Process Multiple PDF Documents ---
pdf_paths = [
    R"C:\Users\mnj-7\Medialogi\LangchainGenAIFromNeo4jTut\PDF_Files\Fokus_På_Regnestrategier.pdf",
    R"C:\Users\mnj-7\Medialogi\LangchainGenAIFromNeo4jTut\PDF_Files\Talforståelse.pdf", # Add paths to your other PDFs
    R"C:\Users\mnj-7\Medialogi\LangchainGenAIFromNeo4jTut\PDF_Files\Tidlig_Algebra.pdf"
]

all_documents = [] # This will store chunks from all PDFs

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len
)

for pdf_path in pdf_paths:
    print(f"Loading content from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    documents = text_splitter.split_documents(pages)
    all_documents.extend(documents) # Add the chunks to our main list
    print(f" -> Split into {len(documents)} chunks.")

print(f"\nTotal chunks from all documents: {len(all_documents)}")

# --- 3. Extract a Knowledge Graph using a Local LLM ---
print("Extracting entities and relationships using local LLM...")

# Instantiate the local LLM you want to use through Ollama
# IMPORTANT: Replace "llama3.1" with the model you pulled (e.g., "mistral")
llm = ChatOllama(model="llama3.1", temperature=0)

# The transformer that will convert text chunks into graph structures
llm_transformer = LLMGraphTransformer(llm=llm)

# Process the documents and convert them to graph structures
# This step can be slow, especially without a GPU
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Add the extracted graph data to Neo4j
print("Adding extracted graph to Neo4j...")
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)
print("Successfully stored knowledge graph in Neo4j.")


# --- 4. Create a Vector Index for Semantic Search ---
# This part is similar to your original script. It creates separate 'Chunk' nodes
# and a vector index on them, allowing for similarity searches.

print("\nCreating embedding model for vector index...")
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#model_kwargs = {'device': 'cpu'}
model_kwargs = {'device': 'cuda'}  # Use GPU
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs
)

print("Embedding documents and creating the 'algebra' vector index...")
neo4j_vector = Neo4jVector.from_documents(
    documents,
    embedding=embedding_model,
    graph=graph,
    #index_name="algebra",
    index_name="Math",
    node_label="MathChunks",
    embedding_node_property="embedding",
    text_node_property="text",
)
print("Successfully created and populated the 'algebra' vector index.")


# --- 5. Verify the Results ---
print("\n--- Verification ---")

# A. Verify the knowledge graph by counting nodes and relationships
try:
    node_count = graph.query("MATCH (n) RETURN count(n) AS count")[0]['count']
    rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) AS count")[0]['count']
    print(f"Knowledge Graph Stats:")
    print(f" -> Number of nodes: {node_count}")
    print(f" -> Number of relationships: {rel_count}")
except Exception as e:
    print(f"Could not query graph stats: {e}")

# B. Verify the vector index with a similarity search
print("\nPerforming a similarity search...")
query = "Hvad er tidlig algebra?"
result_with_scores = neo4j_vector.similarity_search_with_score(query, k=3)

print("\n--- Similarity Search Results ---")
for doc, score in result_with_scores:
    print(f"Similarity Score: {score:.4f}")
    print(f"Source Page: {doc.metadata.get('page', 'N/A')}")
    print(f"Text: {doc.page_content[:250]}...\n")

print("--- Script Finished ---")