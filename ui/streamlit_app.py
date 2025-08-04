import os
import sys
import streamlit as st
from dotenv import load_dotenv
# ensure modules is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.rag_qa import RAGQA, reload_vectorstore

# load .env
load_dotenv()

# credentials
APP_USERNAME = os.getenv("APP_USERNAME")
APP_PASSWORD = os.getenv("APP_PASSWORD")

# debug missing
if not APP_USERNAME or not APP_PASSWORD:
    st.error("ENV VARS APP_USERNAME/APP_PASSWORD missing.")
    st.stop()

# page config
st.set_page_config(page_title="AI Agent MCP", layout="wide")
st.title("🤖 AI Agent MCP")

# initialize auth
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == APP_USERNAME and pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# instantiate RAGQA
if "rag" not in st.session_state:
    st.session_state.rag = RAGQA()

# Sidebar controls
st.sidebar.header("Vector Store")
if st.sidebar.button("🔁 Refresh Vector Store"):
    with st.spinner("Refreshing…"):
        ok, log = reload_vectorstore()
        if ok:
            st.sidebar.success("Refreshed")
            st.session_state.rag = RAGQA()
        else:
            st.sidebar.error("Failed to refresh")
            st.sidebar.text(log)

# chat UI
query = st.text_input("Ask a question:")
if query:
    with st.spinner("Thinking…"):
        ans, src = st.session_state.rag.query(query)
    if not src:
        st.warning(ans)
    else:
        st.markdown(f"**Answer:** {ans}")
        st.markdown("**Sources:**")
        for s in src:
            st.write(f"- {s}")

st.markdown("---")
st.caption("© AI Agent MCP — Powered by LangChain, Chroma, OpenAI, Streamlit")