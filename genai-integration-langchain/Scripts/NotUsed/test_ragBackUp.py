import os
import json
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
from deepeval.models.base_model import DeepEvalBaseLLM
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
# Check for Confident AI credentials (either in env or local login)
if not os.getenv("CONFIDENT_AI_API_KEY"):
    print("NOTE: No CONFIDENT_AI_API_KEY found in .env. Assuming you have run 'deepeval login' locally.")

uri = os.getenv("NEO4J_URI")
print(f"Current Working Directory: {os.getcwd()}")
print(f"NEO4J_URI Found: {uri if uri else 'NO - CHECK .env FILE'}")

# Options: "openai" or "local"
LLM_PROVIDER = "local"  # Set to OpenAI to use your requested model
print(f"--- SELECTED LLM PROVIDER: {LLM_PROVIDER.upper()} ---")


# ==========================================
# 1. ROBUST LOCAL JUDGE (EVALUATOR)
# ==========================================
# We use a local model for the 'Judge' to save costs on evaluation metrics
class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model="llama3.1"):
        self.model_name = model
        self.chat = ChatOllama(model=model, temperature=0, format="json")

    def load_model(self):
        return self.chat

    def _parse_to_pydantic(self, json_str: str, schema):
        try:
            clean_str = json_str.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]

            data = json.loads(clean_str.strip())
            return schema(**data)
        except Exception as e:
            print(f"JSON Parsing Error: {e}")
            print(f"Raw Output: {json_str}")
            return None

    def generate(self, prompt: str, schema=None) -> str:
        res = self.chat.invoke(prompt).content
        if schema:
            return self._parse_to_pydantic(res, schema)
        return res

    async def a_generate(self, prompt: str, schema=None):
        res = await self.chat.ainvoke(prompt)
        content = res.content
        if schema:
            return self._parse_to_pydantic(content, schema)
        return content

    def get_model_name(self):
        return self.model_name


# Initialize the Judge
local_judge = OllamaJudge("llama3.1")

# ==========================================
# 2. DATA LOADING
# ==========================================
# Using a relative path is safer than an absolute C:\ path
filename = R"/genai-integration-langchain/Scripts/Golden Dataset.xlsx"
test_data = []

print(f"\n--- Loading Test Data from {filename} ---")

try:
    df = pd.DataFrame()

    if not os.path.exists(filename):
        print(f"ERROR: File '{filename}' not found in current directory.")
        # Fallback to absolute path if necessary (Uncomment if needed)
        # filename = R"C:\Users\mnj-7\Medialogi\LangchainGenAIFromNeo4jTut\genai-integration-langchain\Scripts\Golden Dataset.xlsx"

    if os.path.exists(filename):
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
        # Normalize column names
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
        print("WARNING: Dataframe is empty or file not found.")

except Exception as e:
    print(f"CRITICAL ERROR loading file: {e}")

# ==========================================
# 3. GENERATION MODEL SETUP (THE TUTOR)
# ==========================================

chat_llm = None

if LLM_PROVIDER == "openai":
    print("Initializing OpenAI for Chat...")
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is missing in .env file")
        exit()

    # Using the requested model
    chat_llm = ChatOpenAI(model="gpt-5-mini", temperature=0.1)
else:
    print("Initializing local LLM (Llama 3.1) for Chat...")
    chat_llm = ChatOllama(
        model="llama3.1",
        temperature=0.1,
        stop=["<|eot_id|>", "SPØRGSMÅL:", "KONTEKST:", "SVAR (til eleven):"]
    )

print("Initializing Embedding Model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'}
)

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
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        print("Neo4j Retriever connected (k=1).")
    except Exception as e:
        print(f"Skipping Neo4j connection: {e}")

# --- The Prompt ---
prompt_template = """
Du er en hjælpsom matematik-lærer for indskolingen (1.-3. klasse).

DIN OPGAVE:
Forklar eleven, hvordan man løser regnestykket.

REGLER FOR DIT SVAR:
1. Tjek først KONTEKSTEN. Hvis den indeholder en smart strategi (fx "Tier-venner", "Dobbelt-op"), så BRUG DEN.
2. Hvis konteksten IKKE passer (eller er tom), så brug din egen viden, men vær PÆDAGOGISK.
   - Forklar tankegangen i stedet for bare at give resultatet.
   - Eksempel: Sig "Tænk på 2+2+2" i stedet for bare "6".
3. Hold svaret kort og venligt (max 3-4 linjer).

KONTEKST (Strategi fra bogen):
{context}

SPØRGSMÅL:
{question}

SVAR (til eleven):
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
chain = prompt | chat_llm | StrOutputParser()


# ==========================================
# 4. TEST LOOP
# ==========================================

@pytest.mark.parametrize("data", test_data)
def test_math_rag(data):
    # Skip test if Neo4j isn't running
    if not retriever:
        pytest.skip("Neo4j Retriever not connected")

    query = data["input"]
    expected = data["expected_output"]

    print(f"\n---------------------------------------------------")
    print(f"Testing Query: {query}")

    # 1. Retrieve
    # Using invoke() directly on retriever for simplicity
    docs = retriever.invoke(query)
    retrieval_context = [doc.page_content for doc in docs]
    context_str = "\n".join(retrieval_context)

    # 2. Generate
    actual_output = chain.invoke({"question": query, "context": context_str})

    print(f"Expected: {expected}")
    print(f"Actual:   {actual_output}")

    # 3. Metrics (Using local_judge for evaluation)
    # Adjust thresholds as needed
    context_metric = ContextualRelevancyMetric(
        threshold=0.4,
        model=local_judge,
        include_reason=True
    )
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.4,
        model=local_judge,
        include_reason=True
    )
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.4,
        model=local_judge,
        include_reason=True
    )

    # 4. Define Test Case
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected
    )

    # 5. Run Assert
    # This automatically pushes results to Confident AI if logged in
    assert_test(
        test_case,
        [context_metric, faithfulness_metric, answer_relevancy_metric]
    )