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
    model = "gpt-5-mini",
    model_kwargs={"temperature": 1}
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
            MERGE (s)-[rel:QUESTIONS_ANSWERED]->(o:Operation {{name: '{operation_name}'}})
            ON MATCH SET rel.correctAnswers = rel.correctAnswers + 1,
                          rel.totalAnswers = rel.totalAnswers + 1
            ON CREATE SET rel.correctAnswers = 1,
                          rel.incorrectAnswers = 0,
                          rel.totalAnswers = 1;
            """
        else:
            query = f"""
            MATCH (s:Student {{_navn: '{student_name}'}}),
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

        print("Saved to graph!")



question_groups = [
    ["Hvad er 3+5?", "Hvad er 7+6?", "Hvad er 12+4?", "Hvad er 9+8?", "Hvad er 15+3?"], #Addition
    ["Hvad er 10 - 4?", "Hvad er 18 - 7?", "Hvad er 9 - 3?", "Hvad er 14 - 5?", "Hvad er 20 - 9?"], #Subtraktion
    ["Hvad er 3*4?", "Hvad er 5*6?", "Hvad er 2*7?", "Hvad er 8*3?", "Hvad er 4*9?"], #Multiplikation
    ["Hvad er 12/3?", "Hvad er 20/4?", "Hvad er 18/2?", "Hvad er 15/5?", "Hvad er 24/6?"] #Division
    ]

smartElev = [
    #Addition
    "8",
    "13",
    "16",
    "17",
    "18",
    #Subtraktion
    "6",
    "12", #Forkert med vilje, bare fordi
    "6",
    "9",
    "11",
    #Multiplikation
    "12",
    "30",
    "14",
    "24",
    "36",
    #Division
    "4",
    "5",
    "9",
    "4", #Forkert med vilje
    "4",
]

notSmartElev = [
    #Addition
    "6",
    "10",
    "13",
    "20",
    "11", #Rigtigt med vilje
    #Subtraktion
    "3",
    "10",
    "12",
    "10",
    "8",
    #Multiplikation
    "9",
    "25",
    "14", #Rigtigt med vilje
    "12",
    "24",
    #Division
    "3",
    "6",
    "8",
    "3", #Rigtigt med vilje
    "8",
]

# #Test if works
# result_raw = evaluate_answer(smartElev[0], "Addition", question_groups[0][0])
# result = json.loads(result_raw)
# is_correct = result["is_correct"]
# save_result_to_graph("SmartElev", "Addition", is_correct)
# exit()

answerCounter = 0
student_name = "SmartElev"
operationCounter = 0
operations = ["Addition", "Subtraktion", "Multiplikation", "Division"]
print("We start with: " + operations[operationCounter])

FIVE_COUNTER = [0, 1, 2, 3, 4] #5

#Loop for smartElev
for operation in operations:
    current_operation = operation
    for n in FIVE_COUNTER:
        result_raw = evaluate_answer(smartElev[answerCounter], current_operation, question_groups[operationCounter][n])

        result = json.loads(result_raw) #Take raw json from evaluate_answer()
        is_correct = result["is_correct"] #Get info about correctness
        save_result_to_graph(student_name, current_operation, is_correct) #Save result to graph
        print("Question number " + str(answerCounter+1) + ": " + ("Correct" if is_correct else "Wrong"))

        answerCounter += 1 #Advance to next question
        print("Answer nr: " + str(answerCounter))

    operationCounter += 1 #Advance to next operation
    print("Next is: " + operations[operationCounter])
