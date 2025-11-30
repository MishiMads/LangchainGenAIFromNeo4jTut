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

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model = "gpt-5-mini",
    model_kwargs={"temperature": 1}
)
# llm = Ollama(model="llama3.1")


graphDriver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)


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

    print("-----------Cypher Query:-----------\n\n" + cypher_query)

    with graphDriver.session() as session:
        result = session.run(cypher_query)
        return [record.data() for record in result]

def make_answer(question, student_name):
    context_records = retrieve_context("Hvilken regnestrategi kan bruges til at løse spørgsmålet: " + question)
    context = "\n".join(str(record) for record in context_records)
    print("-------Context:-------\n\n" + context)

    prompt = f"""
    Du er en hjælpelærer for 1.-3. klasse, som skal assistere en elev: {student_name}.
    Du skal give en mulig regnestrategi til at løse et regnestykke.
    Regnestrategien(STRATEGY) skal være forbundet til én af de fire regnearter(OPERATION) fra konteksten: Enten Addition, Subtraktion, Multiplikation eller Division.
    Der må KUN bruges regnestrategier fra konteksten.
        
    CONTEXT
    {context}
    
    QUESTION
    {question}

    ANSWER
    """

    #Use llama3.1
    # result = ollama.chat(
    #     model="llama3.1",
    #     messages=[{"role": "user", "content": prompt}],
    # )
    #
    # return result["message"]["content"] \
    #     .replace("```", "").replace("`", "").strip()

    #Use openai
    return llm.invoke(prompt).content

dataset_question = input("-------Spørgsmål-------\n\n  Question: ")

student_name = "Elev1"

result_raw = make_answer(dataset_question, student_name)


print("-------Svar-------\n\n" + result_raw)

