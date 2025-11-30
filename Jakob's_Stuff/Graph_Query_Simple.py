import os
import sys

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
                Student (alder: STRING, elevnr: STRING, klassetrin: STRING, _navn: STRING)
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



def answer_query(query):
    #context = retrieve_context(query)
    context_records = retrieve_context(query)

    context = "\n".join(str(record) for record in context_records)

    print("-------------------- CONTEXT: --------------------\n\n" + context)

    #You are a teaching assistant, that has extensive knowledge in math for students in 1-3. grade.
    #Use ONLY the context below to answer the user's query. If the context does not contain the answer, say you don't know.

    #konteksten ikke hjalp og derefter byde ind med anden viden.

    prompt = f"""
    Du er en hjælpelærer, som skal assistere læring af matematik til elever i 1-3. klasse.
    Du må KUN bruge kontekten hernede under, og hvis konteksten ikke har svaret, skal du sige, at du ikke ved svaret.
    
    ### CONTEXT
    {context}
    
    ### QUESTION
    {query}
    
    ### ANSWER
    """

    # result = ollama.chat(
    #     model = "llama3.1",
    #     messages = [{"role": "user", "content": prompt}]
    # )
    #
    # return result["message"]["content"]

    response = llm.invoke(prompt).content
    return response

#print("-------------------- HERUNDER ER SVARET FRA QUERY: --------------------\n\n" + answer_query("Baseret på Elev1's tidligere besvarede spørgsmål, og hvor mange de fik rigtige og forkerte, hvilket emne burde de så få stillet et spørgsmål til nu?"))
print("-------------------- HERUNDER ER SVARET FRA QUERY: --------------------\n\n" + answer_query("Hvilken regnestrategier ville være god at bruge til regnestykket: 6+7?"))
