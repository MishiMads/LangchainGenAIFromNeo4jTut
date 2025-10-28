import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jVector, Neo4jGraph
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# --- 1. Connect to Neo4j ---
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

# --- 2. Create the embedding model ---
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_kwargs = {'device': 'cpu'}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs
)

# --- 3. Load and Process the Local PDF Document ---
pdf_path = R"C:\Users\mnj-7\Medialogi\genai-integration-langchain\tidlig-algebra.pdf"
print(f"Loading content from {pdf_path}...")

loader = PyPDFLoader(pdf_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
documents = text_splitter.split_documents(pages)
print(f"Loaded and split the PDF into {len(documents)} chunks.")

# --- 4. Embed and Store the Document in Neo4j ---
print("Embedding documents and storing them in Neo4j...")
neo4j_vector = Neo4jVector.from_documents(
    documents,
    embedding=embedding_model,
    graph=graph,
    index_name="algebra",
    node_label="Chunk",
    embedding_node_property="embedding",
    text_node_property="text",
)
print("Successfully created and populated the 'algebra' vector index.")

# --- 5. Verify by Searching for Similar Content ---
query = "Hvad er tidlig algebra?"
# Use similarity_search_with_score to get documents AND scores
result_with_scores = neo4j_vector.similarity_search_with_score(query, k=3)

print("\n--- Similarity Search Results ---")
# The loop now unpacks each tuple into a 'doc' and a 'score'
for doc, score in result_with_scores:
    print(f"Similarity Score: {score}") # Use the 'score' variable
    print(f"Source Page: {doc.metadata.get('page', 'N/A')}")
    print(f"Text: {doc.page_content}\n")