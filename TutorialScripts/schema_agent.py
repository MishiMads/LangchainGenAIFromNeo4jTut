import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.llms import Ollama  # Import Ollama
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict

# Connect to Neo4j
from langchain_neo4j import Neo4jGraph

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("neo4j://127.0.0.1:7687"),
    username=os.getenv("neo4j"),
    password=os.getenv("CuteAndFunny"),
)


# Initialize the local Mistral LLM using Ollama
model = Ollama(model="mistral")

# Create a prompt
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)

# Define state for application
class State(TypedDict):
    question: str
    context: List[dict]
    answer: str

# Define functions for each step in the application

# Retrieve context
def retrieve(state: State):
    """
    Retrieves the Neo4j graph schema.
    """
    context = graph.query("CALL db.schema.visualization()")
    return {"context": context}

# Generate the answer based on the question and context
def generate(state: State):
    """
    Generates an answer using the local LLM.
    """
    # Format the prompt with the retrieved context and question
    formatted_prompt = prompt.format(
        context=state["context"], question=state["question"]
    )
    # Invoke the local model
    response = model.invoke(formatted_prompt)
    return {"answer": response}

# Define application steps
workflow = StateGraph(State)

# Add the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Set the entrypoint
workflow.set_entry_point("retrieve")

# Add the edges
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# Run the application
question = "How is the graph structured?"
response = app.invoke({"question": question})
print("Answer:", response["answer"])