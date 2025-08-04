import os
import sys
import streamlit as st
from dotenv import load_dotenv
# Add root of project to sys.path so 'modules' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.rag_qa import RAGQA, reload_vectorstore

load_dotenv()

# Set page config
st.set_page_config(page_title="AI Agent MCP", layout="wide")
st.title("🤖 AI Agent MCP")

# Basic Auth using Streamlit secrets or .env
APP_USERNAME = os.getenv("APP_USERNAME")
APP_PASSWORD = os.getenv("APP_PASSWORD")

def login():
    st.session_state.logged_in = False
    with st.form("Login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if username == APP_USERNAME and password == APP_PASSWORD:
                st.session_state.logged_in = True
            else:
                st.error("Invalid credentials")

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login()
    st.stop()

# App logic
rag_qa = RAGQA()

st.sidebar.header("📂 Vector Store Controls")

# Button to refresh vector DB
if st.sidebar.button("🔁 Refresh Vector Store"):
    with st.spinner("Refreshing vector store..."):
        success, log = reload_vectorstore()
        if success:
            st.sidebar.success("Vector store refreshed!")
        else:
            st.sidebar.error("Failed to refresh vector store.")
            st.sidebar.text_area("Error log", log)

st.markdown("Ask a question based on your ingested documents and URLs.")

user_input = st.text_input("Your Question")

if user_input:
    with st.spinner("Thinking..."):
        answer, sources = rag_qa.query(user_input)

        # Step 2: Check if the answer came from a meaningful source
        if not sources or "source" not in str(sources).lower():
            st.warning("⚠️ No relevant context was found in your documents or URLs.")
            st.info("Try adding more relevant files or URLs, or reloading the vector store.")
        else:
            # Step 3: Show the actual result
            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources")
            for src in sources:
                st.markdown(f"- `{src}`")

st.markdown("---")
st.markdown("© AI Agent MCP – Powered by LangChain, Chroma, OpenAI, and Streamlit.")
