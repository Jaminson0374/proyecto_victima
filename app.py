import streamlit as st
st.write("Inicio de la aplicación")
st.write("Cargando modelo...")
st.write("Inicio")


st.write("Embeddings OK")


st.write("LLM OK")

import faiss
import json
import sqlite3
import numpy as np
from datetime import datetime

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================

# CONFIGURACIÓN

# =====================================

CONFIG = {
"embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
"llm_model": "Qwen/Qwen2.5-0.5B-Instruct",
"top_k": 2,
"temperature": 0.1,
"max_tokens": 120
}

# =====================================

# INTERFAZ

# =====================================

st.set_page_config(
page_title="Asistente para Víctimas",
page_icon="⚖️",
layout="wide"
)

st.title("⚖️ Asistente para Víctimas del Conflicto Armado")
st.markdown(
"""
Sistema RAG basado en normativa colombiana:
Ley 1448 de 2011,
Ley 2343 de 2023,
Ley 2421 de 2024,
Resolución 1049 de 2019 y
Resolución 582 de 2021.
"""
)

# =====================================

# CARGA DE MODELOS

# =====================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        CONFIG["embedding_model"]
    )

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["llm_model"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["llm_model"],
        device_map="auto"
    )
    return tokenizer, model

embedding_model = load_embedding_model()

tokenizer, model = load_llm()

# =====================================

# CARGA DE FAISS

# =====================================

@st.cache_resource
def load_index():

    return faiss.read_index(
    "faiss_index/knowledge_base.faiss"
)


index = load_index()

# =====================================

# CARGA DE CHUNKS

# =====================================

@st.cache_data
def load_chunks():

    with open(
        "data/chunks.json",
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)

chunks = load_chunks()

# =====================================

# SQLITE

# =====================================

conn = sqlite3.connect(
"victimas_chatbot.db",
check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations(
id INTEGER PRIMARY KEY AUTOINCREMENT,
question TEXT,
answer TEXT,
timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

# =====================================

# FUNCIONES RAG

# =====================================

def search_documents(query, k=2):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding.astype("float32"),
        k
    )

    results = []

    for idx in indices[0]:

        results.append(
            chunks[idx]
        )

    return results

def retrieve_context(question):

    docs = search_documents(
        question,
        CONFIG["top_k"]
    )

    return "\n\n".join(
        [d["text"] for d in docs]
    )


SYSTEM_PROMPT = """
Eres un asistente jurídico especializado
en víctimas del conflicto armado colombiano.

Reglas:

1. Responde únicamente con el contexto.
2. No inventes información.
3. Si no encuentras evidencia documental,
   indícalo expresamente.
4. Cita la norma cuando sea posible.
   """

def ask_rag(question):

    context = retrieve_context(
        question
    )

    prompt = f"""


    {SYSTEM_PROMPT}

    CONTEXTO:
    {context}

    PREGUNTA:
    {question}

    RESPUESTA:
    """


    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=CONFIG["max_tokens"],
        temperature=CONFIG["temperature"],
        do_sample=False
    )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    cursor.execute(
        """
        INSERT INTO conversations(
            question,
            answer
        )
        VALUES (?,?)
        """,
        (
            question,
            response
        )
    )

    conn.commit()

    return response.strip()


# =====================================

# CHAT

# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input(
    "Escriba su consulta..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Consultando normativa..."
        ):

            answer = ask_rag(question)

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )