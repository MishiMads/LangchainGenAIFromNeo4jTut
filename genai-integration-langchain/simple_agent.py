from dotenv import load_dotenv
load_dotenv()

from langchain_community.llms import Ollama # Import Ollama
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict

# Initialize the local Mistral LLM using Ollama
# By default, Ollama runs on http://localhost:11434
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
    context = [
        {"location": "London", "weather": "Cloudy, sunny skies later"},
        {"location": "San Francisco", "weather": "Sunny skies, raining overnight."},
    ]
    return {"context": context}

# Generate the answer based on the question and context
def generate(state: State):
    # Note: The structure of invoking the prompt and model might differ slightly
    # depending on the LangChain version and the specific model wrapper.
    # The following is a common pattern.
    formatted_prompt = prompt.format(question=state["question"], context=state["context"])
    response = model.invoke(formatted_prompt)
    return {"answer": response}

# Define application steps
workflow = StateGraph(State)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Set the entrypoint
workflow.set_entry_point("retrieve")

# Add edges
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "__end__")


app = workflow.compile()

# Run the application
question = "What is the weather in San Francisco?"
response = app.invoke({"question": question})
print("Answer:", response["answer"])