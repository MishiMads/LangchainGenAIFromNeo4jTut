import os
import sys
import json
import ollama
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from neo4j import Query, GraphDatabase

#DeepEval Imports
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    GEval
)

import pytest
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor

import unicodedata

from dotenv import load_dotenv
load_dotenv()

# llm = ChatOpenAI(
#     openai_api_key=os.getenv('OPENAI_API_KEY'),
#     model = "gpt-5-mini",
#     model_kwargs={"temperature": 1}
# )
# # llm = Ollama(model="llama3.1")


#Choose LLM's and Judges. "openai", "local", or "qwen"
LLM_PROVIDER = "openai"
JUDGE_PROVIDER = "openai"


golden_dataset = []

graphDriver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)

class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model="llama3.1"):
        self.model_name = model
        self.chat = Ollama(model=model, temperature=0, format="json")

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

local_judge = OllamaJudge("llama3.1")

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

#Choose provicer
if JUDGE_PROVIDER == "openai":
    print("Initializing OpenAI Judge (gpt-4o-mini)...")
    local_judge = OpenAIJudge("gpt-4o-mini")
elif JUDGE_PROVIDER == "qwen":
    print("Initializing Qwen Judge (qwen2.5:14b)...")
    local_judge = OllamaJudge("qwen2.5:14b")
else:
    print("Initializing Local Judge (llama3.1)...")
    local_judge = OllamaJudge("llama3.1")

#initialize llm
if LLM_PROVIDER == "openai":
    print("Initializing OpenAI for Chat...")
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is missing in .env file")
        exit()
    chat_llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
elif LLM_PROVIDER == "qwen":
    print("Initializing Qwen (Qwen2.5 14B) for Chat...")
    chat_llm = Ollama(
        model="qwen2.5:14b",
        temperature=0,
        num_predict=150,  # Limit output tokens for faster generation
        stop=["<|eot_id|>", "SPØRGSMÅL:", "KONTEKST:", "SVAR (til eleven):"]
    )
else:
    print("Initializing local LLM (Llama 3.1) for Chat...")
    chat_llm = Ollama(
        model="llama3.1",
        temperature=0,
        num_predict=150,
        stop=["<|eot_id|>", "SPØRGSMÅL:", "KONTEKST:", "SVAR (til eleven):"]
    )

#Load DATA ! --------------------------------------------
filename = r"C:\Users\jakob\Desktop\Git Repos\LangchainGenAIFromNeo4jTut\Jakob's_Stuff\Golden Dataset IGEN.xlsx"
test_data = []

print(f"\n--- Loading Test Data from {filename} ---")

try:
    df = pd.DataFrame()

    if os.path.exists(filename):
        _, ext = os.path.splitext(filename)

        df = pd.read_excel(filename)
        print("Loaded excel file!!!!!!!!!")


    if not df.empty:
        df.columns = [c.strip() for c in df.columns]
        required_cols = ["Elevens Spørgsmål", "Ideelt Robotsvar (Guidende)"]

        print([repr(c) for c in df.columns]) #Print excel

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


#Ting og sager
def retrieve_context(query):
    context = f"""
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
                Student (alder: STRING, elevnr: STRING, klassetrin: STRING, _navn: STRING)
            """

    cypher_prompt = f"""
            You are an assistant who writes Cypher queries for Neo4j.
            Answer with **one single valid Cypher query**.
            Do NOT include multiple RETURN statements.
            Do NOT include multiple query blocks.
            Always produce one MATCH/OPTIONAL MATCH sequence followed by ONE RETURN.
            User question: `Hvilken regnestrategi kan bruges til at løse spørgsmålet: {query}`
            Schema:
            {context}

            Generate a Cypher query that returns the necessary information about teaching strategies.
            Focus on finding Strategy nodes that CAN_BE_USED_FOR relevant Operations.
            Remember to return all involved nodes and relationships, along with their respective properties.
            """

    if LLM_PROVIDER == "openai":
        cypher_query = chat_llm.invoke(cypher_prompt).content
    elif LLM_PROVIDER == "qwen":
        result = ollama.chat(
            model="qwen2.5:14b",
            messages=[{"role": "user", "content": cypher_prompt}],
        )

        cypher_query = result["message"]["content"] \
            .replace("```cypher", "").replace("```", "").replace("`", "").strip()
    else:
        result = ollama.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": cypher_prompt}],
        )

        cypher_query = result["message"]["content"] \
            .replace("```cypher", "").replace("```", "").replace("`", "").strip()


    print("-----------Cypher Query:-----------\n\n" + cypher_query)

    with graphDriver.session() as session:
        result = session.run(cypher_query)
        return [record.data() for record in result]

#Modsvar til make_answer_from_manual_graph
def answer_query(question, student_name="Test_Student"):
    # context = retrieve_context(query)
    context_records = retrieve_context(question)

    context = "\n".join(str(record) for record in context_records)

    #print("-------------------- CONTEXT: --------------------\n\n" + context)

    prompt = f"""
    Du er en hjælpelærer for 1.-3. klasse, som skal assistere en elev: {student_name}.
    Du skal give en mulig regnestrategi til at løse et regnestykke.
    Regnestrategien(STRATEGY) skal være forbundet til én af de fire regnearter(OPERATION) fra konteksten: Enten Addition, Subtraktion, Multiplikation eller Division.
    Der må KUN bruges regnestrategier fra konteksten.
    
    VIGTIGE REGLER:
    1. Brug ALDRIG flere ord end nødvendigt - max 2-3 sætninger
    2. GIV ALDRIG det direkte svar - guide eleven til at tænke selv
    3. Stil spørgsmål der hjælper eleven med at opdage løsningen

    ### CONTEXT
    {context}

    ### QUESTION
    {question}

    ### ANSWER
    """

    #Den her er sat op anderledes med Ollama :^)
    if LLM_PROVIDER == "openai":
        response = chat_llm.invoke(prompt).content
        return response, context_records
    elif LLM_PROVIDER == "qwen":
        result = ollama.chat(
            model = "qwen2.5:14b",
            messages = [{"role": "user", "content": prompt}]
        )

        return result["message"]["content"], context_records
    else:
        result = ollama.chat(
            model = "llama3.1",
            messages = [{"role": "user", "content": prompt}]
        )

        return result["message"]["content"], context_records




#Try to save the result of the student's answer in the Knowledge Graph
def save_result_to_graph(student_name, operation_name, is_correct):
    with graphDriver.session() as session:
        if is_correct:
            query = f"""
            MATCH (s:Student {{_navn: '{student_name}'}}),
                  (o:Operation {{name: '{operation_name}'}})
            MERGE (s)-[rel:QUESTIONS_ANSWERED]->(o)
            ON MATCH SET rel.correctAnswers = rel.correctAnswers + 1,
                          rel.totalAnswers = rel.totalAnswers + 1
            ON CREATE SET rel.correctAnswers = 1,
                          rel.incorrectAnswers = 0,
                          rel.totalAnswers = 1;
            """
        else:
            query = f"""
            MATCH (s:Student {{_navn: '{student_name}'}}),
                  (o:Operation {{name: '{operation_name}'}})
            MERGE (s)-[rel:QUESTIONS_ANSWERED]->(o)
            ON MATCH SET rel.incorrectAnswers = rel.incorrectAnswers + 1,
                          rel.totalAnswers = rel.totalAnswers + 1
            ON CREATE SET rel.correctAnswers = 0,
                          rel.incorrectAnswers = 1,
                          rel.totalAnswers = 1;
            """

        session.run(query)
        print()

#Evaluate whether the answer given by the student is correct or not - NOT USING ?!
def evaluate_answer(user_answer, question):
    prompt = f"""
    Brug KUN informationen herunder.
    Du er en læringsassistent til børn i 1-3. klasse. Du har stillet et spørgsmål: {question}
    Bestem om elevens svar er korrekt.

    ### STUDENT ANSWER
    {user_answer}

    Svar som JSON:
    {{
        "is_correct": true/false
    }}
    """

    #Den her er sat op anderledes med Ollama :^)
    if LLM_PROVIDER == "openai":
        response = chat_llm.invoke(prompt).content
        return response
    elif LLM_PROVIDER == "qwen":
        result = ollama.chat(
            model="qwen2.5:14b",
            messages=[{"role": "user", "content": prompt}]
        )

        return result["message"]["content"]
    else:
        result = ollama.chat(
            model = "llama3.1",
            messages = [{"role": "user", "content": prompt}]
        )

        return result["message"]["content"]


#Helper functions --------------------------------
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


#Test
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
    actual_output, context_records = answer_query(query)

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

    #Metrics
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.5,
        model=local_judge,
        include_reason=True
    )

    pedagogical_metric = GEval(
        name="Pedagogical Quality",
        criteria="""Evaluate how well the response teaches 1st-3rd grade Danish students (ages 7-9) math concepts""",
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
        faithfulness_metric,
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

#Close session after finishing
def pytest_sessionfinish(session, exitstatus):
    if graphDriver:
        graphDriver.close()
        print("Connection closed!")
