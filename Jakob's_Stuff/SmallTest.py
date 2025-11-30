import os
from langchain_community.llms import Ollama
from langchain.graphs import Neo4jGraph
from neo4j import Query, GraphDatabase
from langchain.chains import GraphCypherQAChain

from dotenv import load_dotenv
load_dotenv()

# 1. Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

print(graph.schema)

graphDriver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)

schema_query = """
CALL apoc.schema.assert(
  {
    Concept: ["name"],
    Example: ["exampleName"],
    Topic: ["topicName"],
    Operation: ["name"],
    MainTopic: ["id"],
    Strategy: ["name"],
    RepresentationOfNumbers: ["name"],
    Property: ["name"],
    SubSkill: ["name"],
    Student: ["elevnr"]
  },
  {
    QUESTIONS_ANSWERED: ["totalAnswers"],
    LAST_ANSWERED: ["timeSpent"],
    HAS_HAD_TOPIC: ["recency"],
    IS_ENGAGED_IN: ["levelOfEngagement"]
  }
)
"""


with graphDriver.session() as session:
    result = session.run(schema_query)
    print(result)
    for record in result:
        print(record.data())