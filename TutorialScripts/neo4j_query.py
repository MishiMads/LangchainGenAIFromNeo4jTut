import os
from dotenv import load_dotenv
load_dotenv()
from langchain_neo4j import Neo4jGraph

# Create Neo4jGraph instance
graph = Neo4jGraph(
    url=os.getenv("neo4j://127.0.0.1:7687"),
    username=os.getenv("neo4j"),
    password=os.getenv("CuteAndFunny"),
)



# Run a query and print the result
result = graph.query("""
MATCH (m:Movie {title: "Mission: Impossible"})<-[a:ACTED_IN]-(p:Person)
RETURN p.name AS actor, a.role AS role
""")

print(result)