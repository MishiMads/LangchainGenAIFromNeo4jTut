import os
import json
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase

# --- LangChain Imports ---
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# ==========================================
# 0. CONFIGURATION
# ==========================================

# Vælg LLM: "openai" eller "local"
LLM_PROVIDER = "openai"

# A/B TESTING SWITCH
# True  = Adaptiv Tutor (Bruger historik, statistik og tilpasser sværhedsgrad)
# False = Generisk Tutor (Glemmer historik, kører 'blindt' som kontrol)
USE_USER_MODELING = True

print(f"--- SELECTED LLM: {LLM_PROVIDER.upper()} ---")
print(f"--- ADAPTIVE USER MODELING: {'AKTIVERET' if USE_USER_MODELING else 'DEAKTIVERET (Kontrol)'} ---")

# --- 1. Initialize Models ---

llm = None

if LLM_PROVIDER == "openai":
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY mangler i .env filen")
        exit()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
else:
    print("Initializing local LLM (Llama 3.1)...")
    llm = ChatOllama(model="llama3.1", temperature=0)

print("Initializing Embedding Model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'}  # Ret til 'cuda' hvis du har GPU
)

# --- 2. Connect to Neo4j ---
print("Connecting to Neo4j Database...")

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    database=os.getenv("NEO4J_DATABASE")
)

# Vector Store (Lærebogen / Curriculum)
try:
    vector_store = Neo4jVector.from_existing_index(
        embedding=embedding_model,
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
        index_name="Math",
        node_label="MathChunks",
        embedding_node_property="embedding",
        text_node_property="text",
    )
    vector_retriever = vector_store.as_retriever()
    print("Neo4j Vector Index connected.")
except Exception as e:
    print(f"Failed to connect to Vector Index: {e}")
    exit()


# --- 3. HELPER FUNCTIONS ---

def get_last_bot_message(user_id):
    """Henter det sidste svar botten gav (til kontekst-forståelse)."""
    if not USE_USER_MODELING:
        return "Ingen kontekst tilgængelig."

    query = """
    MATCH (u:User {userId: $user_id})-[:ASKED]->(:Question)-[:RECEIVED_ANSWER]->(a:Answer)
    RETURN a.text AS last_answer
    ORDER BY a.timestamp DESC LIMIT 1
    """
    with neo4j_driver.session() as session:
        result = session.run(query, user_id=user_id).single()
        return result["last_answer"] if result else "Ingen tidligere besked."


def get_student_profile(user_id):
    """
    Hjernen bag den adaptive læring.
    Henter statistik (succesrate) og nylige emner (variation).
    """
    if not USE_USER_MODELING:
        return "Ingen elevprofil tilgængelig. Antag at eleven er på et gennemsnitligt niveau og giv tilfældige opgaver."

    # 1. Statistik: Hvor god er eleven til hvert emne?
    stats_query = """
    MATCH (u:User {userId: $user_id})-[:ASKED]->(q:Question)-[:CONCERNS_TOPIC]->(t:InteractionTopic)
    WHERE q.is_correct IS NOT NULL
    WITH t.name AS topic, 
         count(q) AS total, 
         sum(CASE WHEN q.is_correct = true THEN 1 ELSE 0 END) AS correct
    RETURN topic, total, correct
    """

    # 2. Variation: Hvad har vi snakket om for nylig?
    recency_query = """
    MATCH (u:User {userId: $user_id})-[:ASKED]->(q:Question)-[:CONCERNS_TOPIC]->(t:InteractionTopic)
    RETURN t.name AS topic
    ORDER BY q.timestamp DESC LIMIT 3
    """

    stats_summary = []
    recent_topics = []

    with neo4j_driver.session() as session:
        # Kør statistik
        stat_res = session.run(stats_query, user_id=user_id)
        for r in stat_res:
            percent = int((r["correct"] / r["total"]) * 100)
            stats_summary.append(f"- {r['topic']}: {percent}% rigtige ({r['correct']}/{r['total']} opgaver)")

        # Kør historik
        recent_res = session.run(recency_query, user_id=user_id)
        recent_topics = [r["topic"] for r in recent_res]

    # Byg teksten til LLM'en
    profile_text = "ELEV STATISTIK (Succesrate):\n" + (
        "\n".join(stats_summary) if stats_summary else "Ingen data endnu (ny elev).")

    if recent_topics:
        profile_text += f"\n\nSENESTE 3 EMNER (Vigtigt for variation!): {', '.join(recent_topics)}"

    return profile_text


def get_personalized_context(input_dict):
    """Henter både bog-viden og chat-historik."""
    question = input_dict["question"]
    user_id = input_dict["user_id"]

    # 1. Vector Retrieval (Altid aktiv - det er lærebogen)
    vector_docs = vector_retriever.invoke(question)
    vector_context = "\n".join([doc.page_content for doc in vector_docs])

    # 2. Chat Historik (Kun aktiv hvis modeling er True)
    history_context = "INGEN HISTORIK (Kontrolgruppe)."

    if USE_USER_MODELING:
        history_query = """
        MATCH (u:User {userId: $user_id})-[:ASKED]->(q:Question)-[:RECEIVED_ANSWER]->(a:Answer)
        RETURN q.text AS question, a.text AS answer
        ORDER BY q.timestamp DESC LIMIT 5
        """
        with neo4j_driver.session() as session:
            results = session.run(history_query, user_id=user_id)
            past_interactions = [f"AI: {r['answer']}\nElev: {r['question']}" for r in results]
            if past_interactions:
                history_context = "\n".join(reversed(past_interactions))

    return f"CHAT HISTORIK:\n{history_context}\n\nVIDEN FRA LÆREBOG:\n{vector_context}"


# --- 4. THE ANALYZER CHAIN ---
# Analyserer emne og korrekthed

analyzer_template = """
You are a Metadata Extractor for a Math Learning App.
Analyze the User Input based on the Previous Context.

User Input: {question}
Previous Bot Message: {last_context}

Tasks:
1. **Topic**: Identify the specific math topic ('Addition', 'Multiplication', 'Division', or 'Subtraction'). If unsure/mixed, use 'General'.
2. **Intent**: Is the user ASKING a question or ANSWERING a question? (Return 'question', 'answer', or 'Not applicable').
3. **Correctness**: If Intent is 'answer', is it mathematically correct based on the Previous Bot Message? (Return true, false, or null).

Return ONLY a JSON object in this format:
{{
  "topic": "TopicName",
  "intent": "question",
  "is_correct": null
}}
"""

analyzer_prompt = ChatPromptTemplate.from_template(analyzer_template)
analyzer_chain = analyzer_prompt | llm | JsonOutputParser()


def analyze_interaction(user_id, user_input):
    last_msg = get_last_bot_message(user_id)
    try:
        return analyzer_chain.invoke({"question": user_input, "last_context": last_msg})
    except Exception as e:
        print(f"Analysis Error: {e}")
        return {"topic": "General", "intent": "question", "is_correct": None}


# --- 5. THE LOGGER ---

def log_interaction(driver, user_id, user_text, bot_response, metadata):
    """Logger samtalen til grafen."""

    # Hvis vi kører kontrol-test uden modeling, logger vi stadig (til analyse bagefter),
    # men vi bruger det ikke i 'get_student_profile'.

    topic_name = metadata.get("topic", "General")
    intent = metadata.get("intent", "question")
    is_correct = metadata.get("is_correct")

    query = """
    MERGE (u:User {userId: $user_id})
    MERGE (t:InteractionTopic {name: $topic_name})

    CREATE (q:Question {
        text: $user_text, 
        timestamp: datetime(),
        intent: $intent,
        is_correct: $is_correct
    })

    CREATE (a:Answer {
        text: $bot_response, 
        timestamp: datetime()
    })

    MERGE (u)-[:ASKED {timestamp: datetime()}]->(q)
    MERGE (q)-[:RECEIVED_ANSWER]->(a)
    MERGE (q)-[:CONCERNS_TOPIC]->(t)
    """
    try:
        with driver.session() as session:
            session.run(query, user_id=user_id, topic_name=topic_name,
                        user_text=user_text, intent=intent, is_correct=is_correct,
                        bot_response=bot_response)
        print(f" -> Logged. Topic: {topic_name} | Correct: {is_correct}")
    except Exception as e:
        print(f"Logging Error: {e}")


# --- 6. THE RAG CHAIN (ADAPTIVE TUTOR) ---

rag_template = """
### ROLLE
Du er en venlig og adaptiv matematik-tutor for eleven {user_id}. Du taler dansk.

### DIN VIDEN OM ELEVEN (VIGTIGT)
{student_profile}

### NUVÆRENDE KONTEKST (Historik & Bog)
{context}

### DIN OPGAVE NU
Du skal vælge din handling baseret på elevens profil og svar:

1. **VURDER SVARET (hvis eleven lige har svaret på et regnestykke):**
   - Tjek om svaret er rigtigt.
   - **HVIS RIGTIGT:** Sig kort "Flot!" eller "Rigtigt!". Giv straks en NY opgave.
   - **HVIS FORKERT:** Sig venligt "Ikke helt". Giv et hint (f.eks. "Husk at 7 gange 10 er 70..."). Giv IKKE svaret. Spørg igen.

2. **VÆLG NÆSTE OPGAVE (ADAPTIV LÆRING):**
   - **Sværhedsgrad:** - Er successraten < 50% i dette emne? Gør det MEGET nemmere (små tal).
     - Er successraten > 80%? Gør det lidt sværere (større tal).
   - **Variation:**
     - Se "SENESTE 3 EMNER". Hvis vi har øvet det samme emne 3 gange i træk (f.eks. Multiplikation), så SKIFT emne (f.eks. til Addition).
     - Varier opgavetyperne.

### REGLER
- **INGEN ABSTRAKTE SPØRGSMÅL:** Spørg ALDRIG "Hvilken strategi brugte du?". Stil kun konkrete regnestykker (f.eks. "Hvad er 6 gange 7?").
- **SIG IKKE HEJ IGEN:** Hvis chat historikken ikke er tom, så gå direkte til feedback og næste spørgsmål.

### DATA OM FORRIGE KONTEKST
Du spurgte sidst om: "{last_context}"

### ELEVENS NYE BESKED:
"{question}"

### DIT SVAR:
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)

rag_chain = (
        {
            "context": RunnableLambda(get_personalized_context),
            "question": lambda x: x["question"],
            "user_id": lambda x: x["user_id"],
            "last_context": lambda x: get_last_bot_message(x["user_id"]),
            "student_profile": lambda x: get_student_profile(x["user_id"])
        }
        | rag_prompt
        | llm
        | StrOutputParser()
)


# --- 7. RUN FUNCTIONS ---

def start_session_with_question(user_id):
    """Genererer et start-spørgsmål for at komme i gang."""
    print(" >> Generating starter question...")

    prompt = """
    Du er en dansk matematiklærer. 
    Lav et helt simpelt regnestykke til et barn (f.eks. "Hvad er 2 plus 2?").
    Skriv KUN spørgsmålet. Intet "Hej".
    """
    initial_question = llm.invoke(prompt).content
    print(f"AI (Starter): {initial_question}")

    # Log det som start
    log_interaction(neo4j_driver, user_id, "(Session Start)", initial_question,
                    {"topic": "General", "intent": "question", "is_correct": None})


def run_chat_cycle(user_id, user_input):
    print(f"\n--- User: {user_input} ---")

    # 1. Analyze
    print(" >> Analyzing...")
    metadata = analyze_interaction(user_id, user_input)

    # 2. Generate (RAG)
    print(" >> Thinking...")
    response = rag_chain.invoke({"question": user_input, "user_id": user_id})
    print(f"AI: {response}")

    # 3. Log
    print(" >> Logging...")
    log_interaction(neo4j_driver, user_id, user_input, response, metadata)


# --- 8. MAIN LOOP ---

if __name__ == "__main__":
    # Brug et unikt navn pr. test, så statistikken ikke blandes
    # F.eks. "Jakob_Test1" (med modeling) og "Jakob_Test2" (uden)
    user_id = "Jakob_Adaptive_Test"

    print(f"\n=== MATH TUTOR SESSION: {user_id} ===")

    # Start med et spørgsmål fra botten
    start_session_with_question(user_id)

    print("\nType 'exit', 'quit', or 'q' to end.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Ending session...")
                break
            if not user_input.strip():
                continue

            run_chat_cycle(user_id, user_input)

        except KeyboardInterrupt:
            print("\nExiting...")
            break

    neo4j_driver.close()
    print("--- Done ---")