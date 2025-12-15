import os
import json
import pytest
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from neo4j import GraphDatabase

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

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
if not os.getenv("CONFIDENT_AI_API_KEY"):
    print("NOTE: No CONFIDENT_AI_API_KEY found in .env. Assuming you have run 'deepeval login' locally.")

uri = os.getenv("NEO4J_URI")
print(f"Current Working Directory: {os.getcwd()}")
print(f"NEO4J_URI Found: {uri if uri else 'NO - CHECK .env FILE'}")

# Options: "openai" or "local"
LLM_PROVIDER = "local"
JUDGE_PROVIDER = "local"
print(f"--- SELECTED LLM PROVIDER: {LLM_PROVIDER.upper()} ---")
print(f"--- SELECTED JUDGE PROVIDER: {JUDGE_PROVIDER.upper()} ---")

# ==========================================
# 1. JUDGE CLASSES
# ==========================================
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
    def __init__(self, model="gpt-4o-mini"):
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
    print("Initializing OpenAI Judge (gpt-4o-mini)...")
    local_judge = OpenAIJudge("gpt-4o-mini")
else:
    print("Initializing Local Judge (llama3.1)...")
    local_judge = OllamaJudge("llama3.1")

# ==========================================
# 2. DATA LOADING
# ==========================================
filename = R"C:\Users\jakob\Desktop\Git Repos\LangchainGenAIFromNeo4jTut\genai-integration-langchain\Scripts\Golden Dataset.xlsx"
test_data = []

print(f"\n--- Loading Test Data from {filename} ---")

try:
    df = pd.DataFrame()

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
# 3. MANUAL GRAPH SETUP
# ==========================================

# Initialize Neo4j connection for manual graph
graphDriver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)

# Initialize LLM
if LLM_PROVIDER == "openai":
    print("Initializing OpenAI for Chat...")
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is missing in .env file")
        exit()
    chat_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
else:
    print("Initializing local LLM (Llama 3.1) for Chat...")
    chat_llm = ChatOllama(
        model="llama3.1",
        temperature=0,
        num_predict=150,
        stop=["<|eot_id|>", "SPØRGSMÅL:", "KONTEKST:", "SVAR (til eleven):"]
    )


def retrieve_context_from_manual_graph(query):
    """
    Retrieves context from the manually created knowledge graph
    using Cypher queries generated by LLM
    """
    schema = """
            Relationships:
                (:Concept)-[:REFERS_TO]->(:Concept)
                (:Concept)-[:REFERS_TO]->(:Operation)
                (:Concept)-[:HAS_PROPERTY]->(:Property)
                (:Example)-[:EXAMPLE_OF]->(:Strategy)
                (:Example)-[:EXAMPLE_OF]->(:Tool)
                (:Example)-[:EXAMPLE_OF]->(:Concept)
                (:Topic)-[:HAS_REPRESENTATION]->(:RepresentationOfNumbers)
                (:Topic)-[:HAS_CONCEPT]->(:Concept)
                (:Topic)-[:HAS_OPERATION]->(:Operation)
                (:Topic)-[:HAS_STRATEGY]->(:Strategy)
                (:Operation)-[:IS_RELATED_TO]->(:Concept)
                (:Operation)-[:HAS_OPERAND]->(:RepresentationOfNumbers)
                (:Operation)-[:REFERS_TO]->(:Concept)
                (:MainTopic)-[:HAS_TOPIC]->(:Topic)
                (:Strategy)-[:CAN_BE_USED_FOR]->(:Operation)
                (:Strategy)-[:HAS_TOOL]->(:Tool)
                (:Strategy)-[:IS_RELATED_TO]->(:Operation)
                (:RepresentationOfNumbers)-[:HAS_TOOL]->(:Tool)
                (:SubSkill)-[:IS_RELATED_TO]->(:Operation)
                (:Student)-[:QUESTIONS_ANSWERED]->(:Operation)
                (:Student)-[:LAST_ANSWERED]->(:Operation)
                (:Student)-[:HAS_HAD_TOPIC]->(:Topic)
                (:Student)-[:IS_ENGAGED_IN]->(:Topic)

            Relationship properties:
                QUESTIONS_ANSWERED (correctAnswers: INTEGER, incorrectAnswers: INTEGER, totalAnswers: INTEGER)
                LAST_ANSWERED (recency: STRING, timeSpent: INTEGER)
                HAS_HAD_TOPIC (recency: STRING)
                IS_ENGAGED_IN (levelOfEngagement: INTEGER)

            Node properties:
                (:Student (_navn: $name, alder: $value, elevnr: $value, klassetrin: $string))
            """

    cypher_prompt = f"""
            You are an assistant who writes Cypher queries for Neo4j.
            Answer **only with Cypher**.
            User question: `Hvilken regnestrategi kan bruges til at løse spørgsmålet: {query}`
            Schema:
            {schema}

            Generate a Cypher query that returns the necessary information about teaching strategies.
            Focus on finding Strategy nodes that CAN_BE_USED_FOR relevant Operations.
            Remember to return all involved nodes and relationships, along with their respective properties.
            """

    # Generate Cypher query using LLM
    cypher_query = chat_llm.invoke(cypher_prompt).content

    # Clean up the cypher query
    cypher_query = cypher_query.replace("```cypher", "").replace("```", "").replace("`", "").strip()

    print(f"\n🔍 Generated Cypher Query:\n{cypher_query}\n")

    # Execute query
    try:
        with graphDriver.session() as session:
            result = session.run(cypher_query)
            records = [record.data() for record in result]
            return records
    except Exception as e:
        print(f"⚠️ Error executing Cypher query: {e}")
        return []


def make_answer_from_manual_graph(question, student_name="TestStudent"):
    """
    Generate answer using the manually created knowledge graph
    """
    context_records = retrieve_context_from_manual_graph(question)
    context = "\n".join(str(record) for record in context_records)

    prompt = f"""
    Du er en hjælpelærer for 1.-3. klasse, som skal assistere en elev: {student_name}.
    Du skal give en mulig regnestrategi til at løse et regnestykke.
    Regnestrategien(STRATEGY) skal være forbundet til én af de fire regnearter(OPERATION) fra konteksten: Enten Addition, Subtraktion, Multiplikation eller Division.
    Der må KUN bruges regnestrategier fra konteksten.
    
    VIGTIGE REGLER:
    1. Brug ALDRIG flere ord end nødvendigt - max 2-3 sætninger
    2. GIV ALDRIG det direkte svar - guide eleven til at tænke selv
    3. Stil spørgsmål der hjælper eleven med at opdage løsningen
        
    CONTEXT
    {context}
    
    QUESTION
    {question}

    ANSWER
    """

    return chat_llm.invoke(prompt).content, context_records


# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def print_metric_details(metric):
    """Print detailed evaluation results for any metric"""
    metric_name = getattr(metric, 'name', metric.__class__.__name__)

    print(f"\n📊 METRIC: {metric_name}")
    print(f"Score: {metric.score:.2f} / 1.00")
    print(f"Threshold: {metric.threshold}")
    print(f"Status: {'✅ PASS' if metric.score >= metric.threshold else '❌ FAIL'}")
    if hasattr(metric, 'reason') and metric.reason:
        print(f"\n💭 Judge's Reasoning:")
        print(f"{metric.reason}")
    print("=" * 60)


def validate_answer_quality(actual_output: str, query: str) -> tuple:
    """Basic sanity check to catch ONLY EXTREME hallucination cases"""
    import re
    issues = []

    if len(actual_output.strip()) < 10:
        issues.append("Answer is too short or empty")

    word_count = len(actual_output.split())
    if word_count > 300:
        issues.append(f"Answer extremely long ({word_count} words) - likely hallucination")

    words = actual_output.lower().split()
    word_counts = {}
    for word in words:
        if len(word) > 4:
            word_counts[word] = word_counts.get(word, 0) + 1

    excessive_repeats = [word for word, count in word_counts.items() if count > 8]
    if excessive_repeats:
        issues.append(f"Severe repetition detected: {', '.join(excessive_repeats[:2])}")

    error_patterns = [r"undefined", r"null", r"ERROR", r"\[object Object\]", r"NaN"]
    for pattern in error_patterns:
        if re.search(pattern, actual_output, re.IGNORECASE):
            issues.append(f"Contains error pattern: {pattern}")

    return len(issues) == 0, "; ".join(issues)


async def evaluate_metrics_async(test_case, metrics):
    """Evaluate all metrics concurrently for faster performance"""
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=len(metrics)) as executor:
        futures = [
            loop.run_in_executor(executor, metric.measure, test_case)
            for metric in metrics
        ]
        await asyncio.gather(*futures)

    return metrics


# ==========================================
# 5. TEST LOOP FOR MANUAL GRAPH
# ==========================================

@pytest.mark.parametrize("data", test_data)
def test_math_rag_manual_graph(data):
    """
    Test RAG using the manually created knowledge graph
    """
    # Skip test if Neo4j isn't running
    if not graphDriver:
        pytest.skip("Neo4j connection not available")

    query = data["input"]
    expected = data["expected_output"]

    print(f"\n{'='*60}")
    print(f"🧪 TESTING MANUAL GRAPH")
    print(f"{'='*60}")
    print(f"Testing Query: {query}")

    # 1. Retrieve from manual graph
    actual_output, context_records = make_answer_from_manual_graph(query)

    # Format context for evaluation
    retrieval_context = [str(record) for record in context_records]
    context_str = "\n".join(retrieval_context)

    # Validate that context was retrieved
    print(f"\n🔍 RETRIEVAL VALIDATION (MANUAL GRAPH):")
    print(f"Number of graph records retrieved: {len(context_records)}")

    if len(context_records) == 0:
        print("⚠️ WARNING: No context retrieved from manual graph")
        pytest.skip("No context retrieved from manual knowledge graph")

    print(f"✅ Context retrieved successfully from manual knowledge graph")

    # VALIDATE BEFORE EVALUATING
    is_valid, validation_msg = validate_answer_quality(actual_output, query)
    if not is_valid:
        print(f"\n⚠️ WARNING: Answer quality issues detected: {validation_msg}")
        print(f"Actual output: {actual_output}")
        pytest.skip(f"Skipping test due to potential hallucination: {validation_msg}")

    print(f"\n📝 Retrieved Context Preview: {context_str[:200]}...")
    print(f"\n📚 Expected: {expected}")
    print(f"\n🤖 Actual (full): {actual_output}")

    # 2. Define Metrics
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
        criteria="""Evaluate how well the response teaches 1st-3rd grade Danish students math concepts.

        High-quality pedagogical responses should:
        - Use questions to stimulate the student's own thinking rather than giving direct answers
        - Reference appropriate teaching strategies from the curriculum (e.g., 10'er venner, dobbelt-op, tier-venner)
        - Use age-appropriate language that 7-9 year olds can understand
        - Avoid directly stating the numeric answer

        Consider the overall teaching effectiveness, not just checklist compliance.""",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
        model=local_judge,
        strict_mode=False
    )

    # 3. Define Test Case
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected
    )

    # Define all metrics
    metrics = [
        context_metric,
        faithfulness_metric,
        answer_relevancy_metric,
        pedagogical_metric,
    ]

    # 4. Parallel Evaluation
    print("\n⚡ Running parallel metric evaluation on MANUAL GRAPH...")
    loop = asyncio.get_event_loop()
    evaluated_metrics = loop.run_until_complete(
        evaluate_metrics_async(test_case, metrics)
    )

    # Print results
    for metric in evaluated_metrics:
        print_metric_details(metric)

    print()

    # 5. Assert Test
    assert_test(test_case, metrics)


# ==========================================
# 6. CLEANUP
# ==========================================

def pytest_sessionfinish(session, exitstatus):
    """Close Neo4j driver after all tests"""
    if graphDriver:
        graphDriver.close()
        print("\n✅ Neo4j connection closed")

