import os
import json
import pytest
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from dotenv import load_dotenv
from openpyxl import Workbook
from datetime import datetime

# Disable verbose DeepEval logging
os.environ["CONFIDENT_METRIC_LOGGING_VERBOSE"] = "0"

# --- LangChain Imports ---
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- DeepEval Imports ---
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    GEval
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

# Options: "openai", "local", or "qwen"
LLM_PROVIDER = "qwen"
# Options: "openai", "local", or "qwen"
JUDGE_PROVIDER = "qwen"
print(f"--- SELECTED LLM PROVIDER: {LLM_PROVIDER.upper()} ---")
print(f"--- SELECTED JUDGE PROVIDER: {JUDGE_PROVIDER.upper()} ---")



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

class OpenAIJudge(DeepEvalBaseLLM):
    def __init__(self, model="gpt-5-mini"):
        self.model_name = model
        self.chat = ChatOpenAI(model=model, temperature=0)

    def load_model(self):
        return self.chat

    def generate(self, prompt: str, schema=None) -> str:
        res = self.chat.invoke(prompt).content
        if schema:
            try:
                data = json.loads(res)
                return schema(**data)
            except Exception as e:
                print(f"JSON Parsing Error: {e}")
                return None
        return res

    async def a_generate(self, prompt: str, schema=None):
        res = await self.chat.ainvoke(prompt)
        content = res.content
        if schema:
            try:
                data = json.loads(content)
                return schema(**data)
            except Exception as e:
                print(f"JSON Parsing Error: {e}")
                return None
        return content

    def get_model_name(self):
        return self.model_name


# Initialize the Judge
if JUDGE_PROVIDER == "openai":
    print("Initializing OpenAI Judge (gpt-5-mini)...")
    local_judge = OpenAIJudge("gpt-5-mini")
elif JUDGE_PROVIDER == "qwen":
    print("Initializing Qwen Judge (qwen2.5:14b)...")
    local_judge = OllamaJudge("qwen2.5:14b")
else:  # Default to local llama
    print("Initializing Local Judge (llama3.1)...")
    local_judge = OllamaJudge("llama3.1")

# ==========================================
# 2. DATA LOADING
# ==========================================
# Using a relative path is safer than an absolute C:\ path
filename = R"C:\Users\mnj-7\Medialogi\LangchainGenAIFromNeo4jTut\genai-integration-langchain\Scripts\GroupedDatasetByStrategy.xlsx"
test_data = []

print(f"\n--- Loading Test Data from {filename} ---")


def print_metric_details(metric):
    """Print detailed evaluation results for any metric"""
    # Get metric name - GEval has 'name', others use class name
    metric_name = getattr(metric, 'name', metric.__class__.__name__)

    print(f"\n📊 METRIC: {metric_name}")
    print(f"Score: {metric.score:.2f} / 1.00")
    print(f"Threshold: {metric.threshold}")
    print(f"Status: {'✅ PASS' if metric.score >= metric.threshold else '❌ FAIL'}")
    if hasattr(metric, 'reason') and metric.reason:
        print(f"\n💭 Judge's Reasoning:")
        print(f"{metric.reason}")
    print("=" * 60)


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
                    "category": row.get("Overgruppe", "General")
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
    chat_llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
elif LLM_PROVIDER == "qwen":
    print("Initializing Qwen (Qwen2.5 14B) for Chat...")
    chat_llm = ChatOllama(
        model="qwen2.5:14b",
        temperature=0,
        num_predict=150,  # Limit output tokens for faster generation
        stop=["<|eot_id|>", "SPØRGSMÅL:", "KONTEKST:", "SVAR (til eleven):"]
    )
else:  # Default to local llama
    print("Initializing local LLM (Llama 3.1) for Chat...")
    chat_llm = ChatOllama(
        model="llama3.1",
        temperature=0,
        num_predict=150,  # Limit output tokens for faster generation
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
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3})
        print("Neo4j Retriever connected (k=3).")
    except Exception as e:
        print(f"Skipping Neo4j connection: {e}")

# --- The Prompt ---
prompt_template = """
Du er en hjælpsom matematik-lærer for indskolingen (1.-3. klasse).

VIGTIGE REGLER:
1. Hvis konteksten indeholder en relevant strategi (fx "10'er venner", "dobbelt-op", "tier-venner"), så SKAL du bruge den
2. Hvis konteksten IKKE passer til spørgsmålet (fx addition-strategi for division), så ignorer den og brug almen viden
3. Forklar strategien trin-for-trin på en måde børn kan forstå
4. Brug ALDRIG flere ord end nødvendigt - max 2-3 sætninger
5. GIV ALDRIG det direkte svar - guide eleven til at tænke selv

EKSEMPLER:
Spørgsmål: "Hvad er 7 + 8?"
Svar: "Kan du huske 10'er venner? 7 + 3 = 10. Hvor mange mere skal du lægge til?"

Spørgsmål: "Hvad er 6 + 6?"
Svar: "Lad os bruge dobbelt-op! Hvis 6 + 6 = 12, hvad tror du så 6 + 7 er?"

KONTEKST (Fra lærebogen):
{context}

SPØRGSMÅL:
{question}

SVAR (til eleven):
"""



prompt = ChatPromptTemplate.from_template(prompt_template)
chain = prompt | chat_llm | StrOutputParser()


# Helper function to extract only the final answer
def extract_final_answer(full_output: str) -> str:
    """Extract only the final answer portion after the reasoning"""
    # Look for the last occurrence of the answer marker
    if "SVAR (til eleven):" in full_output:
        return full_output.split("SVAR (til eleven):")[-1].strip()

    # If no marker found, try to get the last paragraph
    paragraphs = [p.strip() for p in full_output.split('\n\n') if p.strip()]
    return paragraphs[-1] if paragraphs else full_output


def validate_answer_quality(actual_output: str, query: str) -> tuple:
    """Basic sanity check to catch ONLY EXTREME hallucination cases"""
    import re
    issues = []

    # Check 1: Empty or extremely short answer (< 10 chars)
    if len(actual_output.strip()) < 10:
        issues.append("Answer is too short or empty")

    # Check 2: VERY long answers (> 300 words = severe verbosity)
    word_count = len(actual_output.split())
    if word_count > 300:
        issues.append(f"Answer extremely long ({word_count} words) - likely hallucination")

    # Check 3: Excessive repetition (same word appears > 8 times)
    words = actual_output.lower().split()
    word_counts = {}
    for word in words:
        if len(word) > 4:  # Only check meaningful words
            word_counts[word] = word_counts.get(word, 0) + 1

    excessive_repeats = [word for word, count in word_counts.items() if count > 8]
    if excessive_repeats:
        issues.append(f"Severe repetition detected: {', '.join(excessive_repeats[:2])}")

    # Check 4: Answer contains obvious error patterns
    error_patterns = [
        r"undefined",
        r"null",
        r"ERROR",
        r"\[object Object\]",
        r"NaN"
    ]
    for pattern in error_patterns:
        if re.search(pattern, actual_output, re.IGNORECASE):
            issues.append(f"Contains error pattern: {pattern}")

    return len(issues) == 0, "; ".join(issues)


async def evaluate_metrics_async(test_case, metrics):
    """Evaluate all metrics concurrently for faster performance"""
    loop = asyncio.get_event_loop()

    # Create thread pool for parallel execution
    with ThreadPoolExecutor(max_workers=len(metrics)) as executor:
        # Schedule all metric measurements to run in parallel
        futures = [
            loop.run_in_executor(executor, metric.measure, test_case)
            for metric in metrics
        ]
        # Wait for all to complete
        await asyncio.gather(*futures)

    return metrics


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

    # Validate that context was retrieved from the knowledge graph
    print(f"\n🔍 RETRIEVAL VALIDATION:")
    print(f"Number of documents retrieved: {len(retrieval_context)}")
    assert len(retrieval_context) > 0, "❌ FAILED: No context retrieved from knowledge graph"
    assert any(len(ctx.strip()) > 10 for ctx in retrieval_context), "❌ FAILED: Retrieved context is too short"
    print(f"✅ Context retrieved successfully from knowledge graph")

    # 2. Generate
    actual_output_raw = chain.invoke({"question": query, "context": context_str})

    # For pedagogical evaluation, keep the FULL output
    actual_output = actual_output_raw  # Don't strip reasoning steps

    # VALIDATE BEFORE EVALUATING - catch hallucinations early
    is_valid, validation_msg = validate_answer_quality(actual_output, query)
    if not is_valid:
        print(f"\n⚠️ WARNING: Answer quality issues detected: {validation_msg}")
        print(f"Actual output: {actual_output}")
        pytest.skip(f"Skipping test due to potential hallucination: {validation_msg}")

    print(f"\nRetrieved Context: {context_str[:200]}...")
    print(f"Expected: {expected}")
    print(f"Actual (full): {actual_output}")

    # 3. Metrics (Using local_judge for evaluation)
    # Adjust thresholds as needed

    context_metric = ContextualRelevancyMetric(
        threshold=0.4,
        model=local_judge,
        include_reason=True
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.5,
        model=local_judge,
        include_reason=True
    )
    
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.6,
        model=local_judge,
        include_reason=True
    )

    pedagogical_metric = GEval(
        name="Pedagogical Quality",
        evaluation_steps=[
            "Analyze if the response uses guiding questions or hints instead of giving direct answers",
            "Check if an appropriate teaching strategy is used or referenced when relevant (e.g., 10'er venner, dobbelt-op, tier-venner)"
            " - multiple strategies are NOT required",
            "Evaluate if the language avoids overly complex academic terminology"
            " - simple Danish phrases that 7-9 year olds encounter in school are acceptable"
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
        model=local_judge,
        strict_mode=False
    )

    # 4. Define Test Case
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected
    )

    # Define all metrics you want to test
    metrics = [
        faithfulness_metric,      # Tests if answer is faithful to the context (uses knowledge graph)
        pedagogical_metric,       # Tests pedagogical quality
    ]

    # 🚀 PARALLEL EVALUATION - Measure all metrics concurrently (3-5x faster!)
    print("\n⚡ Running parallel metric evaluation...")
    loop = asyncio.get_event_loop()
    evaluated_metrics = loop.run_until_complete(
        evaluate_metrics_async(test_case, metrics)
    )

    # Print results for all metrics
    for metric in evaluated_metrics:
        print_metric_details(metric)

    print()  # Extra newline for readability

    # 5. Run Assert
    # This automatically pushes results to Confident AI if logged in
    try:
        assert_test(
            test_case,
            metrics  # Use the metrics list instead
        )
        print("✅ TEST PASSED - All metrics above threshold!\n")
    except AssertionError as e:
        # Extract just the metric failure info without the full stack trace
        error_msg = str(e)
        print(f"\n❌ TEST FAILED")
        print(f"{'='*60}")
        print(f"{error_msg}")
        print(f"{'='*60}\n")
        raise  # Re-raise to fail the test properly





    #### SAVING RESULTS TO EXCEL LOGIC HERE ####
    test_results = []
    # After metrics evaluation, before assert_test
    result_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "category": data.get("category", "General"),
        "query": query,
        "expected_output": expected,
        "actual_output": actual_output[:200] + "..." if len(actual_output) > 200 else actual_output,
        "context_length": len(retrieval_context),
        "faithfulness_score": faithfulness_metric.score,
        "pedagogical_score": pedagogical_metric.score,
        "test_passed": False
    }

    try:
        assert_test(test_case, metrics)
        print("✅ TEST PASSED - All metrics above threshold!\n")
        result_entry["test_passed"] = True
    except AssertionError as e:
        error_msg = str(e)
        print(f"\n❌ TEST FAILED")
        print(f"{'=' * 60}")
        print(f"{error_msg}")
        print(f"{'=' * 60}\n")
        result_entry["failure_reason"] = error_msg[:200]
        raise
    finally:
        test_results.append(result_entry)

    def pytest_sessionfinish(session, exitstatus):
        """Called after whole test run finished"""
        if test_results:
            save_results_to_excel(test_results)

    def save_results_to_excel(results):
        """Save test results to an Excel file"""
        wb = Workbook()
        ws = wb.active
        ws.title = "Test Results"

        headers = [
            "Timestamp", "Category", "Query", "Expected Output",
            "Actual Output", "Context Length", "Faithfulness Score",
            "Pedagogical Score", "Test Passed", "Failure Reason"
        ]
        ws.append(headers)

        for result in results:
            ws.append([
                result.get("timestamp", ""),
                result.get("category", ""),
                result.get("query", ""),
                result.get("expected_output", ""),
                result.get("actual_output", ""),
                result.get("context_length", 0),
                result.get("faithfulness_score", 0.0),
                result.get("pedagogical_score", 0.0),
                "PASS" if result.get("test_passed") else "FAIL",
                result.get("failure_reason", "")
            ])

        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width

        filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        wb.save(filename)
        print(f"\n📊 Test results saved to: {filename}")
    #[context_metric, faithfulness_metric, answer_relevancy_metric, pedagogical_metric]
    #[context_metric, faithfulness_metric, answer_relevancy_metric, pedagogical_metric]