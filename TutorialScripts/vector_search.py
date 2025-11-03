import os
from dotenv import load_dotenv
#from openai.types import embedding_model
from langchain_neo4j import Neo4jVector

load_dotenv()

from langchain_neo4j import Neo4jGraph
from langchain_community.embeddings import HuggingFaceEmbeddings


# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("neo4j://127.0.0.1:7687"),
    username=os.getenv("neo4j"),
    password=os.getenv("CuteAndFunny"),
)

# Create the embedding model

# 1. Define the model name
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 2. Instantiate the embedding model
# By default, it runs on your CPU. If you have a GPU, see the note below.

# To run on GPU
model_kwargs = {'device': 'cuda'}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs
)

# Code below makes it run on CPU:
#embedding_model = HuggingFaceEmbeddings(model_name=model_name)



# --- You can now use 'embedding_model' anywhere you need embeddings ---

# 3. Example Usage
# Embed a single piece of text (e.g., a user's query)
query = "Hvordan er vejret i Aalborg i dag?"
query_embedding = embedding_model.embed_query(query)

print(f"Embedding for the query (first 5 dimensions): {query_embedding[:5]}")
print(f"Vector dimension: {len(query_embedding)}") # This model outputs 384-dimensional vectors

# Embed a list of documents
documents = [
    "I dag er det mandag, og solen skinner over Nordjylland.",
    "Jens er taget på arbejde på kontoret.",
    "Vejrudsigten lover mildt vejr med let vind."
]
document_embeddings = embedding_model.embed_documents(documents)

print(f"\nNumber of document embeddings: {len(document_embeddings)}")
print(f"Embedding for the first document (first 5 dimensions): {document_embeddings[0][:5]}")


# Create Vector

plot_vector = Neo4jVector.from_existing_index(
    embedding_model,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
)

# Search for similar movie plots

plot = "Toys come alive"
result = plot_vector.similarity_search(plot, k=3)
print(result)

# Parse the documents
for doc in result:
    print(f"Title: {doc.metadata['title']}")
    print(f"Plot: {doc.page_content}\n")
