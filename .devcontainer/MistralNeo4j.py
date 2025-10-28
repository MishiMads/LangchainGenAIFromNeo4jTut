import json
from neo4j import GraphDatabase
from mistralai import Mistral
from getpass import getpass


api_key= getpass("Type your API Key")
neo4j_password =  getpass("CuteAndFunny")
neo4j_user =  getpass("neo4j")
neo4j_uri =  getpass("neo4j://127.0.0.1:7687")
client = Mistral(api_key=api_key)

