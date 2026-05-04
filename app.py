import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

import streamlit as st
from rag_pipeline import build_qa_chain, ask_question


st.set_page_config(
    page_title="Pakistani Legal Assistant",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Pakistani Legal AI Assistant")
st.caption("Ask questions about the Constitution of Pakistan in plain English.")
st.divider()

@st.cache_resource
def load_chain():
    with st.spinner("Loading legal knowledge base..."):
        return build_qa_chain()

chain_tuple = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption(f"📎 Sources: Pages {msg['sources']}")

question = st.chat_input("Ask a legal question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the Constitution..."):
            result = ask_question(chain_tuple, question)
        st.write(result["answer"])
        st.caption(f"📎 Sources: Pages {result['source_pages']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["source_pages"]
    })