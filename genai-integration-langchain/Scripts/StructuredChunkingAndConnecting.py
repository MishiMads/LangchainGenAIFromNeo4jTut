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


# --- 2. Load and Process Multiple PDF Documents ---
pdf_paths = [
    R"C:\Users\mnj-7\Medialogi\LangchainGenAIFromNeo4jTut\PDF_Files\Fokus_På_Regnestrategier.pdf",
    R"C:\Users\mnj-7\Medialogi\LangchainGenAIFromNeo4jTut\PDF_Files\Talforståelse.pdf",
    R"C:\Users\mnj-7\Medialogi\LangchainGenAIFromNeo4jTut\PDF_Files\Tidlig_Algebra.pdf"
]

all_documents = []  # This will store chunks from all PDFs

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
    all_documents.extend(documents)  # Add the chunks to our main list
    print(f" -> Split into {len(documents)} chunks.")

print(f"\nTotal chunks from all documents: {len(all_documents)}")

# --- 3. Extract a Knowledge Graph using a Local LLM (Guided by Your Schema) ---
print("Extracting entities and relationships using local LLM...")

# Instantiate the local LLM through Ollama
llm = ChatOllama(model="llama3.1", temperature=0)

# *** MODIFICATION HERE ***
# We instruct the transformer to ONLY use your specified schema
llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Topic", "Subject"],  # Your KC (Knowledge Component)
    allowed_relationships=["HAS_SUBJECT"]  # Your connection from KC to item
)

# Process ALL documents and convert them to graph structures
# *** BUG FIX: Using all_documents, not just documents ***
print("Converting all documents to graph format...")
graph_documents = llm_transformer.convert_to_graph_documents(all_documents)

# Add the extracted graph data to Neo4j
print(f"Adding {len(graph_documents)} graph documents to Neo4j...")
graph.add_graph_documents(
    graph_documents,
    include_source=True
    # We remove baseEntityLabel=True to keep the schema clean
)
print("Successfully stored knowledge graph in Neo4j.")

# --- 4. Create a Vector Index for Semantic Search ---
# This part is separate from the knowledge graph. It creates 'MathChunks' nodes
# for similarity search on the raw text chunks.

print("\nCreating embedding model for vector index...")
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_kwargs = {'device': 'cuda'}  # Use GPU
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs
)

print("Embedding documents and creating the 'Math' vector index...")
neo4j_vector = Neo4jVector.from_documents(
    all_documents,  # Use all_documents here as well for the vector index
    embedding=embedding_model,
    graph=graph,
    index_name="Math",
    node_label="MathChunks",  # This is a separate label from "Topic" or "Subject"
    embedding_node_property="embedding",
    text_node_property="text",
)
print("Successfully created and populated the 'Math' vector index.")

# --- 5. Verify the Results ---
print("\n--- Verification ---")

# A. Verify the knowledge graph by counting nodes and relationships
try:
    # Check for your specific labels
    topic_count = graph.query("MATCH (t:Topic) RETURN count(t) AS count")[0]['count']
    subject_count = graph.query("MATCH (s:Subject) RETURN count(s) AS count")[0]['count']
    rel_count = graph.query("MATCH (:Topic)-[r:HAS_SUBJECT]->(:Subject) RETURN count(r) AS count")[0]['count']

    print(f"Knowledge Graph Stats (Your Schema):")
    print(f" -> Number of Topics: {topic_count}")
    print(f" -> Number of Subjects: {subject_count}")
    print(f" -> Number of HAS_SUBJECT relationships: {rel_count}")

    # You can also check for vector nodes
    chunk_count = graph.query("MATCH (c:MathChunks) RETURN count(c) AS count")[0]['count']
    print(f"\nVector Index Stats:")
    print(f" -> Number of MathChunks: {chunk_count}")

except Exception as e:
    print(f"Could not query graph stats: {e}")

# B. Verify the vector index with a similarity search
print("\nPerforming a similarity search...")
query = "Hvad er tidlig algebra?"
result_with_scores = neo4j_vector.similarity_search_with_score(query, k=3)

print("\n--- Similarity Search Results (from Vector Index) ---")
for doc, score in result_with_scores:
    print(f"Similarity Score: {score:.4f}")
    print(f"Source Page: {doc.metadata.get('page', 'N/A')}")
    print(f"Text: {doc.page_content[:250]}...\n")

print("--- Script Finished ---")