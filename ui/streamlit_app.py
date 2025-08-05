import os
import sys
import streamlit as st
from dotenv import load_dotenv

# ─── Make sure we can import modules from project root ───
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Core modules
from modules.rag_qa import RAGQA, reload_vectorstore
from modules.summarizer import Summarizer
from modules.memory import ChatMemory
from modules.planner import Planner
from modules.fallback import fallback_answer

# Load env
load_dotenv(os.path.join(ROOT, ".env"))

# — Authentication credentials —
USERNAME = os.getenv("APP_USERNAME")
PASSWORD = os.getenv("APP_PASSWORD")
if not USERNAME or not PASSWORD:
    st.error("⚠️  Missing APP_USERNAME or APP_PASSWORD in .env")
    st.stop()

# — Page config —
st.set_page_config(page_title="AI Agent MCP", layout="wide")
st.title("🤖 AI Agent MCP")

# — Session‐state defaults —
if "auth" not in st.session_state:
    st.session_state.auth = False
if "rag" not in st.session_state:
    st.session_state.rag = RAGQA()
if "summ" not in st.session_state:
    st.session_state.summ = Summarizer()
if "plan" not in st.session_state:
    st.session_state.plan = Planner()
if "mem" not in st.session_state:
    st.session_state.mem = ChatMemory()
if "hist" not in st.session_state:
    st.session_state.hist = []

# — Login form —
if not st.session_state.auth:
    st.subheader("🔐 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == USERNAME and pwd == PASSWORD:
            st.session_state.auth = True
            # NO rerun call needed
        else:
            st.error("❌ Invalid credentials")
    # halt everything else until authenticated
    st.stop()

# — Sidebar tools —
st.sidebar.header("Settings & Tools")

if st.sidebar.button("🔁 Refresh Vector Store"):
    ok, msg = reload_vectorstore()
    if ok:
        st.sidebar.success("✅ Vector store refreshed")
        st.session_state.rag = RAGQA(force_reload=True)
    else:
        st.sidebar.error(f"❌ {msg}")

if st.sidebar.button("🔄 Reset Conversation"):
    st.session_state.mem.clear()
    st.session_state.hist = []
    st.session_state.auth = True  # keep user logged in
    st.experimental_rerun()

# GPT‐fallback toggle
if "use_gpt_fallback" not in st.session_state:
    st.session_state.use_gpt_fallback = False
st.session_state.use_gpt_fallback = st.sidebar.checkbox(
    "Enable GPT fallback", value=st.session_state.use_gpt_fallback
)

# — Main interface —
query = st.text_input("Ask a question:")
length = st.slider("Summary Length (max tokens)", min_value=50, max_value=1000, value=300, step=50)

if query:
    try:
        # 1️⃣ Plan
        steps = st.session_state.plan.plan(query)
        st.markdown("### 🔎 Plan")
        st.write(steps)

        # 2️⃣ Memory & history
        st.session_state.mem.add_user_message(query)
        st.session_state.hist.append({"role": "user", "content": query})

        # 3️⃣ RAG
        answer, sources = st.session_state.rag.query(query)

        # 4️⃣ GPT fallback if no RAG hit
        if not answer.strip() and st.session_state.use_gpt_fallback:
            with st.spinner("No RAG hit—asking ChatGPT…"):
                answer = fallback_answer(query)
            sources = [{"metadata": {"source": "💬 ChatGPT (fallback)"}}]

        # 5️⃣ Summarize
        summary = st.session_state.summ.summarize(answer, max_tokens=length)

        # 6️⃣ Record AI response
        st.session_state.mem.add_ai_message(summary)
        st.session_state.hist.append({"role": "assistant", "content": summary})

        # 7️⃣ Display
        st.markdown("### 💬 Answer")
        st.write(summary)
        if sources:
            st.markdown("### 📚 Sources")
            for doc in sources:
                src = doc.metadata.get("source", "unknown")
                st.write(f"- {src}")

    except Exception as e:
        st.error(f"🚨 {e}")

# — Chat History viewer —
if st.checkbox("Show Chat History"):
    st.markdown("### 📝 History")
    for msg in st.session_state.hist:
        who = "You" if msg["role"] == "user" else "AI"
        st.write(f"**{who}:** {msg['content']}")

st.caption("© AI RAG Agent - powered with Streamlit, Chroma, OpenAI amd LangChain")
