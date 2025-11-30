import os
import sys
import json

import ollama
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
#import langchain_ollama
from neo4j import Query, GraphDatabase
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model = "gpt-5-mini",
    model_kwargs={"temperature": 1}
)
# llm = Ollama(model="llama3.1")

user_id = "Mads"

graphDriver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)


def retrieve_context(query):
    context = f"""
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
            User question: `{query}`
            Schema:
            {context}

            Generate a Cypher query that returns the necessary information.
            Remember to return all involved nodes and relationships, along with their respective properties.
            """

    # result = ollama.chat(
    #     model="llama3.1",
    #     messages=[{"role": "user", "content": cypher_prompt}],
    # )
    #
    # cypher_query = result["message"]["content"] \
    #     .replace("```cypher", "").replace("```", "").replace("`", "").strip()

    cypher_query = llm.invoke(cypher_prompt).content

    #print("-----------Cypher Query:-----------\n\n" + cypher_query)

    with graphDriver.session() as session:
        result = session.run(cypher_query)
        return [record.data() for record in result]


def generate_exercise(topic_query):
    context_records = retrieve_context(topic_query)
    context = "\n".join(str(record) for record in context_records)

    prompt = f"""
    Du er en hjælpelærer for 1.-3. klasse.
    Brug KUN den givne viden fra grafen herunder til at lave en matematikopgave.

    Opgaven skal indeholde:
    - En tekstbaseret opgave
    - Ét korrekt svar (SKJUL svaret for eleven)
    - Relateret til: {topic_query}

    Formatér som JSON:
    {{
        "exercise": "Her skriver du selve opgaven",
        "correct_answer": <det korrekte tal>,
        "operation": "<operation name>"
    }}

    ### GRAPH CONTEXT
    {context}
    """

    raw = llm.invoke(prompt).content \
        .replace("```json", "") \
        .replace("```", "") \
        .strip()

    exercise = json.loads(raw)
    return exercise


def evaluate_answer(user_answer, exercise_data):
    prompt = f"""
    Brug KUN informationen herunder.
    Bestem om elevens svar er korrekt.

    ### TASK
    {exercise_data}

    ### STUDENT ANSWER
    {user_answer}

    Svar som JSON:
    {{
        "is_correct": true/false
    }}
    """
    return llm.invoke(prompt).content

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




exercise = generate_exercise("multiplikation")
print("--------------Opgave:--------------\n\n", exercise["exercise"])

user_answer = input("Svar: ")
result_raw = evaluate_answer(user_answer, exercise)
result = json.loads(result_raw)

is_correct = result["is_correct"]
save_result_to_graph("Elev1", exercise["operation"], is_correct)

print("Rigtigt! 😄" if is_correct else "Det er ikke korrekt, prøv igen! ❌")