import os
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQA

# (Optional but recommended) For a more modern LCEL chain approach
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# --- 1. Initialize the LLM for Generation ---

print("Initializing local LLM via Ollama...")
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

# --- Comment above out and use this instead to use Mistral -> it can only answer in English though ---
#llm = Ollama(model="mistral")

# --- 2. Initialize the Embedding Model for Retrieval ---
# IMPORTANT: Use the exact same model you used for embedding the document
print("Initializing Hugging Face embedding model...")
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#model_kwargs = {'device': 'cpu'}
model_kwargs = {'device': 'cuda'}  # Use GPU
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs
)

# --- 3. Connect to the Existing Neo4j Vector Index ---
# This assumes you have already run your other script to embed and store the data
print("Connecting to existing Neo4j vector index...")
vector_store = Neo4jVector.from_existing_index(
    embedding=embedding_model,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="Math",  # The name of the index you created
    text_node_property="text", # The property containing the text
)

# --- 4. Create the RAG Chain ---
print("Creating the RAG chain...")

# Create a retriever from the vector store
retriever = vector_store.as_retriever()

# Create a prompt template
template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Build the chain using LCEL (LangChain Expression Language)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. Invoke the chain with your question ---
question = "Kan du fortælle mig hvilke opgaver jeg skal lave? Jeg er en skolelev"
print(f"\nInvoking chain with question: {question}")

response = rag_chain.invoke(question)

print("\n--- Final Answer ---")
print(response)