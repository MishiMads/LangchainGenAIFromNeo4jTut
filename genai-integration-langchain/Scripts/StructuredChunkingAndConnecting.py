import os
import pytest
import pandas as pd
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- DeepEval Imports ---
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric
)

# ==========================================
# 0. SETUP & ENVIRONMENT
# ==========================================
load_dotenv(override=True)

print("\n--- DEBUG: ENVIRONMENT CHECK ---")
uri = os.getenv("NEO4J_URI")
print(f"Current Working Directory: {os.getcwd()}")
print(f"NEO4J_URI Found: {uri if uri else 'NO - CHECK .env FILE'}")

# Options: "openai" or "local"
LLM_PROVIDER = "local"
print(f"--- SELECTED LLM PROVIDER: {LLM_PROVIDER.upper()} ---")

# ==========================================
# 1. DATA LOADING (Excel Support)
# ==========================================
filename = R"C:\Users\mnj-7\Medialogi\LangchainGenAIFromNeo4jTut\genai-integration-langchain\Scripts\Golden Dataset.xlsx"
test_data = []

print(f"\n--- Loading Test Data from {filename} ---")

try:
    df = pd.DataFrame()
    _, ext = os.path.splitext(filename)

    if ext.lower() == '.xlsx':
        print("Detected Excel file. Using read_excel...")
        df = pd.read_excel(filename)
    else:
        print("Detected CSV. Using read_csv...")
        try:
            df = pd.read_csv(filename, encoding='cp1252', sep=None, engine='python')
        except:
            df = pd.read_csv(filename, encoding='utf-8', sep=None, engine='python')

    if not df.empty:
        df.columns = [c.strip() for c in df.columns]
        required_cols = ["Elevens Spørgsmål", "Ideelt Robotsvar (Guidende)"]

        if all(col in df.columns for col in required_cols):
            df = df.dropna(subset=required_cols)
            for index, row in df.iterrows():
                test_data.append({
                    "input": row["Elevens Spørgsmål"],
                    "expected_output": row["Ideelt Robotsvar (Guidende)"],
                    "category": row.get("Kategori", "General")
                })
            print(f"Successfully loaded {len(test_data)} test cases.")
        else:
            print(f"ERROR: Missing columns. Found: {df.columns}")
    else:
        print("WARNING: Dataframe is empty.")

except Exception as e:
    print(f"CRITICAL ERROR loading file: {e}")

# ==========================================
# 2. MODEL & DATABASE SETUP
# ==========================================

# --- A. Initialize LLM ---
llm = None

if LLM_PROVIDER == "openai":
    print("Initializing OpenAI GPT-5-mini...")
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is missing in .env file")
        exit()
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
else:
    print("Initializing local LLM (Llama 3.1)...")
    # Added stop tokens here to prevent looping
    llm = ChatOllama(
        model="llama3.1",
        temperature=0.1,
        stop=["<|eot_id|>", "SPØRGSMÅL:", "KONTEKST:", "SVAR (til eleven):"]
    )

# --- B. Initialize Embeddings ---
print("Initializing Embedding Model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'}  # Changed to CPU to prevent OOM errors
)

# --- C. Connect to Neo4j ---
retriever = None
if os.getenv("NEO4J_URI"):
    try:
        vector_store = Neo4jVector.from_existing_index(
            embedding=embedding_model,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE"),
            index_name="Math",
            node_label="MathChunks",
            embedding_node_property="embedding",
            text_node_property="text",
        )
        # Using k=1 to keep context focused
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        print("Neo4j Retriever connected (k=1).")
    except Exception as e:
        print(f"Skipping Neo4j connection: {e}")

# --- D. The "Idiot-Proof" Prompt ---
prompt_template = """
Du er en hjælpsom matematik-lærer.

Opgave:
Forklar eleven, hvordan man løser regnestykket ved at bruge strategien fra KONTEKSTEN nedenfor.

REGLER:
1. Hold svaret KORT (max 3 sætninger).
2. Brug KUN strategien fra konteksten.
3. Hvis konteksten ikke passer til spørgsmålet, så sig bare: "Jeg kender ikke en god huskeregel for dette."
4. Opfind IKKE dine egne tal eller metoder.

KONTEKST (Strategi):
{context}

SPØRGSMÅL:
{question}

SVAR (til eleven):
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
chain = prompt | llm | StrOutputParser()


# ==========================================
# 3. TEST LOOP
# ==========================================

@pytest.mark.parametrize("data", test_data)
def test_math_rag(data):
    if not retriever:
        pytest.skip("Neo4j Retriever not connected")

    query = data["input"]
    expected = data["expected_output"]

    print(f"\n---------------------------------------------------")
    print(f"Testing Query: {query}")

    # 1. Retrieve
    docs = retriever.invoke(query)
    retrieval_context = [doc.page_content for doc in docs]
    context_str = "\n".join(retrieval_context)

    # 2. Generate
    actual_output = chain.invoke({"question": query, "context": context_str})

    print(f"Expected: {expected}")
    print(f"Actual:   {actual_output}")

    # 3. Metrics
    # NOTE: DeepEval uses OpenAI for grading by default.
    context_metric = ContextualRelevancyMetric(threshold=0.5)
    faithfulness_metric = FaithfulnessMetric(threshold=0.7)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)

    # 4. Define Test Case
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected
    )

    # 5. Run Assert
    assert_test(
        test_case,
        [context_metric, faithfulness_metric, answer_relevancy_metric]
    )