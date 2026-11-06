"""UI chat Streamlit untuk bot 'What Would Lee Kuan Yew Do?'"""
import os
import sys

# Streamlit Cloud pakai sqlite3 versi lama yang ditolak Chroma.
# Kalau pysqlite3-binary terpasang, pakai itu. Harus sebelum import chromadb.
try:
    import pysqlite3  # noqa: F401

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st

# saat deploy ambil dari secrets, kalau lokal rag.py baca .env
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "GROQ_MODEL" in st.secrets:
        os.environ["GROQ_MODEL"] = st.secrets["GROQ_MODEL"]
except Exception:
    pass

from ingest import build_index
from rag import answer, get_collection

st.set_page_config(page_title="What Would Lee Kuan Yew Do?", page_icon="🇸🇬")


@st.cache_resource(show_spinner="Building the speech index (first run only)...")
def ensure_index():
    try:
        count = get_collection().count()
    except Exception:
        count = 0
    if count == 0:
        build_index()
        count = get_collection().count()
    if count == 0:
        # error tidak di-cache, jadi bisa dicoba lagi setelah pidato ditambah
        raise RuntimeError("No speeches found. Add .txt files to data/speeches/.")
    return count


def render_sources(sources):
    with st.expander("Sources"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {s['title']}** ({s['date']})")
            st.caption(s["text"])


with st.sidebar:
    st.header("About")
    st.write(
        "Ask anything about life, geopolitics or history. The bot answers in the "
        "voice of Lee Kuan Yew, grounded in excerpts from his public speeches "
        "(retrieval-augmented generation)."
    )
    st.warning(
        "This is an AI simulation, not the real Lee Kuan Yew. Answers may be "
        "inaccurate. Check the cited sources."
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("What Would Lee Kuan Yew Do?")

if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it to .env or to Streamlit secrets.")
    st.stop()

try:
    ensure_index()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# tampilkan ulang chat sebelumnya (Streamlit rerun dari atas tiap interaksi)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

if question := st.chat_input("Ask Lee Kuan Yew anything..."):
    # history untuk LLM cuma role + content, tanpa sources
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = answer(question, history)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()
        st.markdown(result["answer"])
        render_sources(result["sources"])

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
