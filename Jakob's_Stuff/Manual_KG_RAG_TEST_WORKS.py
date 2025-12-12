import os
import json
import pandas as pd
from deepeval.evaluate import evaluate

from deepeval.metrics import GEval, faithfulness, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GPTModel

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError, Neo4jError

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
        You are to search in a Neo4j graph based on the following User query: `{query}`
        Find relevant strategies that can be used to answer the question from query.

        *** Examples ***
        
        Finding strategies:
        MATCH (n:Topic)-[r:HAS_STRATEGY]->(m) RETURN n, r, m
        MATCH (strategy:Strategy)-[r:CAN_BE_USED_FOR]->(m) RETURN strategy, r, m
        
        Important Nodes:
        (:Topic (topicName:"Regnestrategier"))
        (:Operation (name:"Addition"))
        (:Operation (name:"Subtraktion"))
        (:Operation (name:"Multiplikation"))
        (:Operation (name:"Division"))

        Answer **only with Cypher**.
        
        ONLY use **ONE** MATCH per node relationship
        ONLY ever use RETURN **ONCE**
        
        Generate a cypher query returning all relevant info.
    """

    cypher_raw = llm.invoke(cypher_prompt).content.strip()
    cypher = clean_cypher(cypher_raw)

    #Test print Cypher query
    #print(cypher)

    try:
        with graphDriver.session() as session:
            result = session.run(cypher)
            return [record.data() for record in result]
    except (CypherSyntaxError, Neo4jError) as e:
        print("Error in retrieving context")
        return [] #Return an empty array if there is an error

#Test RAG
# context_records = retrieve_context("Jeg kan ikke huske hvordan jeg skal udregne 5+4?")
# context = "\n".join([str(r) for r in context_records])
# print("\nContext: " + context)
# exit()

def generate_rag_answer(question, input_context):
    prompt = f"""
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
        {input_context}
        
        SPØRGSMÅL:
        {question}
        
        SVAR (til eleven):
        """

    answer = llm.invoke(prompt).content
    return answer

geval = GEval(
    name="Semantic Alignment",
    evaluation_steps=[
            "Analyze if the response uses guiding questions or hints instead of giving direct answers",
            "Check if an appropriate teaching strategy is used or referenced when relevant (e.g., 10'er venner, dobbelt-op, tier-venner)"
            " - multiple strategies are NOT required",
            "Evaluate if the language avoids overly complex academic terminology"
            " - simple Danish phrases that 7-9 year olds encounter in school are acceptable"
    ],
    model = evaluation_llm,
    #evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT], #We only use this because we only use this
)

faithful = FaithfulnessMetric(
    model = evaluation_llm,
    #threshold = 0.5,
    include_reason = True
)

#Evaluate using both metrics
def evaluate_with_metrics(question, model_answer, ideal_answer, category, retrieval_context):
    #Truncation here
    question_t = truncate(question)
    ideal_answer_t = truncate(ideal_answer)
    model_answer_t = truncate(model_answer)

    test_case = LLMTestCase(
        input=question_t,
        actual_output=model_answer_t,
        expected_output=ideal_answer_t,
        category=category,
        retrieval_context=[retrieval_context]
    )

    result_geval = geval.measure(test_case)
    result_faithful = faithful.measure(test_case)
    #print(faithful.reason)

    def unpack(result):
        if hasattr(result, "reason"):
            return {
                "score": result.score,
                "passed": result.success,
                "reason": result.reason
            }
        else:
            return {"score": float(result), "passed": None, "reason": None}

    return{
        "semantic_alignment": unpack(geval),
        "faithfulness": unpack(faithful),
    }

df = pd.read_excel(r"C:\Users\jakob\Desktop\Git Repos\LangchainGenAIFromNeo4jTut\Jakob's_Stuff\GroupedDatasetByStrategy.xlsx")

def run_pipeline(row):
    question = row["Elevens_Spørgsmål"]
    ideal = row["Ideelt_Robotsvar_(Guidende)"]
    category = row["Overgrupper"]

    print(f"\n==============================")
    print(f"📘 Question: {question}")
    print(f"📚 Category: {category}")

    #Get context and do RAG
    context_records = retrieve_context(question)
    context = "\n".join([str(r) for r in context_records])
    print("\n Context retrieved: ")
    print(context)
    model_answer = generate_rag_answer(question, context)
    print("\n🤖 Model Answer:")
    print(model_answer)

    #Metric evaluation
    result = evaluate_with_metrics(
        question,
        model_answer,
        ideal,
        category,
        context
    )

    print("\n🏁 Evaluation:")
    print(json.dumps(result, indent=2))
    return result

results = []

#Proper loop and save
for i, row in df.iterrows():
    print(f"\n=== Running Test {i+1}/{len(df)} ===")
    results.append(run_pipeline(row))

# Save results
pd.DataFrame(results).to_excel("/geval_results.xlsx", index=False)
print("\n📄 Results saved to /geval_results.xlsx")
