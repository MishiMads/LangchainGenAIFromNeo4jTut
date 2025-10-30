import os

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_neo4j import Neo4jGraph
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI

from dotenv import load_dotenv
load_dotenv()

#DOCS_PATH = r"C:\Users\jakob\PycharmProjects\P7_Main_Project\P7_Environment\PDF_Files"
DOCS_PATH = r"C:\Users\mnj-7\Medialogi\LangchainGenAIFromNeo4jTut\PDF_Files"


llm = Ollama(model="llama3.1")
#llm = Ollama(model="mistral")


graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

#embedding_provider = OllamaEmbeddings(model="nomic-embed-text")
embedding_provider = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cuda'}
)




# Define a custom prompt to guide the LLM's output
CYPHER_GENERATION_TEMPLATE = """
Task:Generate a knowledge graph from the text.
1. Extract entities and relationships from the provided text.
2. The output must be a JSON object with two keys: "nodes" and "relationships".
3. For nodes, each element should have an "id" (the node name) and a "type" (the node's category).
4. For relationships, each element should have a "source" (source node id), a "target" (target node id), and a "type" (the relationship name).
5. IMPORTANT: The "id" for nodes and the "source" and "target" for relationships must be strings, not lists or other data types.

Example:
Text: "Marie Curie, a Polish and naturalized-French physicist, conducted pioneering research on radioactivity. She married Pierre Curie, a French physicist."
JSON Output:
{{
  "nodes": [
    {{"id": "Marie Curie", "type": "Person"}},
    {{"id": "Pierre Curie", "type": "Person"}},
    {{"id": "Physics", "type": "Field"}},
    {{"id": "Radioactivity", "type": "Concept"}}
  ],
  "relationships": [
    {{"source": "Marie Curie", "target": "Pierre Curie", "type": "SPOUSE_OF"}},
    {{"source": "Marie Curie", "target": "Physics", "type": "FIELD_OF_STUDY"}},
    {{"source": "Marie Curie", "target": "Radioactivity", "type": "RESEARCHED"}}
  ]
}}

Text:
{input}
"""

# Create the prompt from the template
custom_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

# Instantiate the transformer with the custom prompt
doc_transformer = LLMGraphTransformer(
    llm=llm,
    prompt=custom_prompt
    )

loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=500,
    chunk_overlap=20,
)

docs = loader.load()
chunks = text_splitter.split_documents(docs)

for chunk in chunks:

    filename = os.path.basename(chunk.metadata['source'])
    chunk_id = f"{filename}.{chunk.metadata['page']}"
    print("Processing -", chunk_id)

    # Embed the chunk
    chunk_embedding = embedding_provider.embed_query(chunk.page_content)

    # Add the Document and Chunk nodes to the graph
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding
    }

    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
        """,
                properties
                )

    # Generate the entities and relationships from the chunk
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    # Map the entities in the graph documents to the chunk node
    for graph_doc in graph_docs:
        chunk_node = Node(
            id=chunk_id,
            type="Chunk"
        )

        for node in graph_doc.nodes:
            graph_doc.relationships.append(
                Relationship(
                    source=chunk_node,
                    target=node,
                    type="HAS_ENTITY"
                )
            )

    # add the graph documents to the graph
    graph.add_graph_documents(graph_docs)