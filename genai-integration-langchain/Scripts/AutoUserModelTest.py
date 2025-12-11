import os
import sys
import json
import re
import datetime
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM for evaluation
llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="gpt-4o-mini",
    temperature=1
)

# Initialize Neo4j driver
graphDriver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
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
    response = llm.invoke(prompt)
    content = response.content

    # Remove markdown code blocks if present
    content = re.sub(r'```json\s*|\s*```', '', content).strip()

    return content


def save_result_to_graph(student_name, operation_name, is_correct):
    now_time = str(datetime.datetime.now())

    with graphDriver.session() as session:
        # Ensure both Student and Operation nodes exist
        session.run("""
            MERGE (s:Student {_navn: $student_name})
            ON CREATE SET s.created = datetime()
            MERGE (o:Operation {name: $operation_name})
        """, student_name=student_name, operation_name=operation_name)

        # Update QUESTIONS_ANSWERED relationship
        if is_correct:
            query = """
            MATCH (s:Student {_navn: $student_name}),
                  (o:Operation {name: $operation_name})
            MERGE (s)-[rel:QUESTIONS_ANSWERED]->(o)
            ON MATCH SET rel.correctAnswers = coalesce(rel.correctAnswers, 0) + 1,
                          rel.totalAnswers = coalesce(rel.totalAnswers, 0) + 1
            ON CREATE SET rel.correctAnswers = 1,
                          rel.incorrectAnswers = 0,
                          rel.totalAnswers = 1
            """
        else:
            query = """
            MATCH (s:Student {_navn: $student_name}),
                  (o:Operation {name: $operation_name})
            MERGE (s)-[rel:QUESTIONS_ANSWERED]->(o)
            ON MATCH SET rel.incorrectAnswers = coalesce(rel.incorrectAnswers, 0) + 1,
                          rel.totalAnswers = coalesce(rel.totalAnswers, 0) + 1
            ON CREATE SET rel.correctAnswers = 0,
                          rel.incorrectAnswers = 1,
                          rel.totalAnswers = 1
            """

        session.run(query, student_name=student_name, operation_name=operation_name)

        # Update LAST_ANSWERED relationship
        query2 = """
            MATCH (s:Student {_navn: $student_name}),
                  (o:Operation {name: $operation_name})
            MERGE (s)-[rel2:LAST_ANSWERED]->(o)
            SET rel2.recency = $now_time
        """
        session.run(query2, student_name=student_name, operation_name=operation_name, now_time=now_time)

        print("Saved to graph!")


# Test data
question_groups = [
    ["Hvad er 3+5?", "Hvad er 7+6?", "Hvad er 12+4?", "Hvad er 9+8?", "Hvad er 15+3?"],
    ["Hvad er 10 - 4?", "Hvad er 18 - 7?", "Hvad er 9 - 3?", "Hvad er 14 - 5?", "Hvad er 20 - 9?"],
    ["Hvad er 3*4?", "Hvad er 5*6?", "Hvad er 2*7?", "Hvad er 8*3?", "Hvad er 4*9?"],
    ["Hvad er 12/3?", "Hvad er 20/4?", "Hvad er 18/2?", "Hvad er 15/5?", "Hvad er 24/6?"]
]

smartElev = [
    "8", "13", "16", "17", "18",
    "6", "12", "6", "9", "11",
    "12", "30", "14", "24", "36",
    "4", "5", "9", "4", "4"
]

notSmartElev = [
    "6", "10", "13", "20", "11",
    "3", "10", "12", "10", "8",
    "9", "25", "14", "12", "24",
    "3", "6", "8", "3", "8"
]

operations = ["Addition", "Subtraktion", "Multiplikation", "Division"]
FIVE_COUNTER = [0, 1, 2, 3, 4]


def run_simulation(student_name, student_answers):
    answerCounter = 0
    operationCounter = 0

    print(f"\n=== Starting simulation for {student_name} ===")
    print("We start with: " + operations[operationCounter])

    for operation in operations:
        current_operation = operation
        for n in FIVE_COUNTER:
            result_raw = evaluate_answer(
                student_answers[answerCounter],
                current_operation,
                question_groups[operationCounter][n]
            )

            try:
                result = json.loads(result_raw)
                is_correct = result["is_correct"]
            except json.JSONDecodeError as e:
                print(f"JSON Parse Error: {e}")
                print(f"Raw response: {result_raw}")
                is_correct = False

            save_result_to_graph(student_name, current_operation, is_correct)
            print(f"Question number {answerCounter + 1}: {'Correct' if is_correct else 'Wrong'}")

            answerCounter += 1
            print(f"Answer nr: {answerCounter}")

        operationCounter += 1
        if operationCounter < len(operations):
            print("Next is: " + operations[operationCounter])


if __name__ == "__main__":
    print("Starting simulations...\n")
    run_simulation("SmartElev", smartElev)
    run_simulation("NotSmartElev", notSmartElev)
    print("\nSimulations complete!")
    graphDriver.close()