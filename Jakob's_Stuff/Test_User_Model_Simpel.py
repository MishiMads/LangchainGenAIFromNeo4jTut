import os
import sys
import json
import datetime #For getting the time

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
    model = "gpt-4o",
    model_kwargs={"temperature": 1},
    response_format={"type": "json_object"}
)
# llm = Ollama(model="llama3.1")

graphDriver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)


def evaluate_answer(user_answer, topic, question):
    prompt = f"""
    Du er en hjælpelærer for 1.-3. klasse.
    Du har lige stillet en elev spørgsmålet: {question}
    Dette spørgsmål er relateret til: {topic}
    Bestem om elevens svar er korret. Brug matematik som kontekst.

    ### STUDENT ANSWER
    {user_answer}

    Svar som JSON:
    {{
        "is_correct": true/false
    }}
    """
    return llm.invoke(prompt).content


def save_result_to_graph(student_name, operation_name, is_correct):
    now_time = str(datetime.datetime.now())

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

        #Save when the last question was answered
        query2 = f"""
            MATCH (s:Student {{_navn: '{student_name}'}}),
                  (o:Operation {{name: '{operation_name}'}})
            MERGE (s)-[rel2:LAST_ANSWERED]->(o)
            SET rel2.recency = "{now_time}"
            """
        session.run(query2)

        #print("Saved to graph!")



question_groups = [
    ["Hvad er 2+3?", "Hvad er 5+5?", "Hvad er 12 + 10?", "Hvad er 17+69?", "Hvad er 96+28?"], #Addition
    ["Hvad er 2-1?", "Hvad er 8-5?", "Hvad er 10-5?", "Hvad er 72-24?", "Hvad er 114-27?"], #Subtraktion
    ["Hvad er 2*2?", "Hvad er 4*3?", "Hvad er 10*5?", "Hvad er 12*11?", "Hvad er 9*17?"], #Multiplikation
    ["Hvad er 4/2?", "Hvad er 9/3?", "Hvad er 12/4?", "Hvad er 60/6?", "Hvad er 52/13?"] #Division
    ]

#Elever's svar:
Elev1 = [5, 10, 22, 84, 124, 1, 3, 5, 48, 87, 4, 12, 50, 131, 121, 2, 3, 3, 10, 4] #19/20

Elev2 = [5, 10, 22, 80, 104, 1, 3, 5, 56, 98, 4, 12, 50, 54, 67, 2, 3, 4, 5, 17] #11/20

Elev3 = [5, 8, 13, 70, 101, 1, 5, 2, 60, 100, 2, 10, 15, 20, 25, 2, 6, 8, 12, 40] #3/20

Elev4 = [5, 10, 22, 86, 112, 1, 3, 5, 50, 90, 2, 7, 15, 23, 25, 6, 12, 16, 50, 40] #7/20

Elev5 = [5, 10, 22, 86, 124, 1, 3, 5, 48, 87, 4, 12, 20, 100, 97, 2, 3, 3, 20, 13] #15/20

#Egentligt svar:
Real_Answer = [5, 10, 22, 84, 124, 1, 3, 5, 48, 87, 4, 12, 50, 153, 121, 2, 3, 3, 10, 4]

# #Test if works
# result_raw = evaluate_answer(smartElev[0], "Addition", question_groups[0][0])
# result = json.loads(result_raw)
# is_correct = result["is_correct"]
# save_result_to_graph("SmartElev", "Addition", is_correct)
# exit()

answerCounter = 0
student_name = "Elev5"
operationCounter = 0
operations = ["Addition", "Subtraktion", "Multiplikation", "Division"]
print("We start with: " + operations[operationCounter])

FIVE_COUNTER = [0, 1, 2, 3, 4] #5

#Loop for smartElev
for operation in operations:
    current_operation = operation
    for n in FIVE_COUNTER:
        result_raw = evaluate_answer(Elev5[answerCounter], current_operation, question_groups[operationCounter][n])

        result = json.loads(result_raw) #Take raw json from evaluate_answer()
        is_correct = result["is_correct"] #Get info about correctness
        save_result_to_graph(student_name, current_operation, is_correct) #Save result to graph
        print("Question number " + str(answerCounter+1) + ": " + ("Correct" if is_correct else "Wrong"))

        answerCounter += 1 #Advance to next question
        #print("Answer nr: " + str(answerCounter))

    operationCounter += 1 #Advance to next operation
    print("Next is: " + operations[operationCounter])
