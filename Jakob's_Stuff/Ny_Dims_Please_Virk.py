import os
import json
import pandas as pd

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GPTModel

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from neo4j import GraphDatabase

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature = 1,
)

# llm = ChatOllama(
#     model = "llama3.1",
#     temperature = 1,
# )

evaluation_llm = GPTModel(
    model = "gpt-4o-mini",
    temperature = 1,
)

graphDriver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

#Trunctation for keeping down the use of tokens
MAX_LEN = 1500  #Truncation limit
def truncate(text, max_len=MAX_LEN):
    text = str(text)
    return text if len(text) <= max_len else text[:max_len] + "...[TRUNCATED]"

def clean_cypher(query: str) -> str:
    query = query.replace("```cypher", "")
    query = query.replace("```", "")

    return query.strip()

def retrieve_context(query):
    schema = f"""
        Relationships:
                (:Concept)-[:REFERS_TO]->(:Concept)
                (:Concept)-[:HAS_PROPERTY]->(:Property)
                (:Example)-[:EXAMPLE_OF]->(:Strategy)
                (:Example)-[:EXAMPLE_OF]->(:Tool)
                (:Example)-[:EXAMPLE_OF]->(:Concept)
                (:Topic)-[:HAS_REPRESENTATION]->(:RepresentationOfNumbers)
                (:Topic)-[:HAS_STRATEGY]->(:Strategy)
                (:Topic)-[:HAS_CONCEPT]->(:Concept)
                (:Topic)-[:HAS_OPERATION]->(:Operation)
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

        User query: `{query}`

        Schema:
        {schema}

        Generate a cypher query returning all relevant info.
    """



    cypher_raw = llm.invoke(cypher_prompt).content.strip()
    cypher = clean_cypher(cypher_raw)

    with graphDriver.session() as session:
        result = session.run(cypher)
        return [record.data() for record in result]


def generate_rag_answer(question):
    context_records = retrieve_context(question)
    context = "\n".join([str(r) for r in context_records])

    prompt = f"""
    Brug KUN følgende graf-viden til at svare:

    ### GRAPH KNOWLEDGE
    {context}

    ### ELEV SPØRGSMÅL
    {question}

    Svar tydeligt og pædagogisk.
    """

    answer = llm.invoke(prompt).content
    return answer

geval = GEval(
    name="Semantic Alignment",
    criteria="""Evaluate how well the response teaches 1st-3rd grade Danish students math concepts.

        High-quality pedagogical responses should:
        - Use questions to stimulate the student's own thinking rather than giving direct answers
        - Reference appropriate teaching strategies from the curriculum (e.g., 10'er venner, dobbelt-op, tier-venner)
        - Use age-appropriate language that 7-9 year olds can understand
        - Avoid directly stating the numeric answer

        Consider the overall teaching effectiveness, not just checklist compliance.""",
    model = evaluation_llm,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
)

def evaluate_with_geval(question, model_answer, ideal_answer, category):
    #Truncation here
    question_t = truncate(question)
    ideal_answer_t = truncate(ideal_answer)
    model_answer_t = truncate(model_answer)

    test_case = LLMTestCase(
        input=question_t,
        actual_output=model_answer_t,
        expected_output=ideal_answer_t,
        category=category,
    )

    result = geval.measure(test_case)
    if isinstance(result, float):
        return {"score": result, "passed": None, "reason": None}
    else:
        return {"score": result.score, "passed": result.success, "reason": result.reason}

df = pd.read_excel(r"C:\Users\jakob\Desktop\Git Repos\LangchainGenAIFromNeo4jTut\Jakob's_Stuff\Golden Dataset IGEN.xlsx")

def run_pipeline(row):
    question = row["Elevens_Spørgsmål"]
    ideal = row["Ideelt_Robotsvar_(Guidende)"]
    category = row["Kategori"]

    print(f"\n==============================")
    print(f"📘 Question: {question}")
    print(f"📚 Category: {category}")

    # Step 1: Generate answer using RAG + Neo4j
    model_answer = generate_rag_answer(question)
    print("\n🤖 Model Answer:")
    print(model_answer)

    # Step 2: Evaluate using GEval
    result = evaluate_with_geval(
        question,
        model_answer,
        ideal,
        category
    )

    print("\n🏁 Evaluation:")
    print(json.dumps(result, indent=2))
    return result

results = []
for i, row in df.iterrows():
    print(f"\n=== Running Test {i+1}/{len(df)} ===")
    results.append(run_pipeline(row))

# Save results
pd.DataFrame(results).to_excel("/geval_results.xlsx", index=False)
print("\n📄 Results saved to /geval_results.xlsx")
