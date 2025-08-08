import os
import sys
import streamlit as st
from dotenv import load_dotenv

# ─── Load environment ───
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))

# ─── App Modules ───
from modules.rag_qa import RAGQA
from modules.summarizer import Summarizer
from modules.memory import ChatMemory
from modules.planner import Planner
from modules.fallback import fallback_answer
from modules.rag_ingest import ingest_documents

# ─── Auth Setup ───
USERNAME = os.getenv("APP_USERNAME")
PASSWORD = os.getenv("APP_PASSWORD")
if not USERNAME or not PASSWORD:
    st.error("⚠️ Missing APP_USERNAME or APP_PASSWORD in .env")
    st.stop()

# ─── Streamlit Page Config ───
st.set_page_config(page_title="AI Agent MCP", layout="wide")
st.title("🤖 AI Agent MCP")

# ─── Session State Init ───
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
if "use_gpt_fallback" not in st.session_state:
    st.session_state.use_gpt_fallback = False

# ─── Login Form ───
if not st.session_state.auth:
    st.subheader("🔐 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == USERNAME and pwd == PASSWORD:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("❌ Invalid credentials")
    st.stop()

# ─── Sidebar Tools ───
st.sidebar.header("Settings & Tools")

if st.sidebar.button("🔁 Refresh Vector Store"):
    try:
        # Step 1: Manually remove RAG instance (free up Chroma lock)
        st.session_state.rag = None  

        # Step 2: Delete old vectorstore dir
        import shutil
        from modules.config import PERSIST_DIR
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)

        # Step 3: Run ingestion pipeline (rebuild vectorstore)
        with st.spinner("♻️ Rebuilding vector store…"):
            rag_ingest.ingest_documents()

        # Step 4: Reinitialize RAG module
        st.session_state.rag = RAGQA(force_reload=True)
        st.sidebar.success("✅ Vector store refreshed successfully")

    except Exception as e:
        st.sidebar.error(f"❌ Ingestion error: {e}")

if st.sidebar.button("🔄 Reset Conversation"):
    st.session_state.mem.clear()
    st.session_state.hist = []
    st.session_state.auth = True
    st.experimental_rerun()

st.session_state.use_gpt_fallback = st.sidebar.checkbox(
    "Enable GPT fallback", value=st.session_state.use_gpt_fallback
)

# ─── Main Interaction ───
query = st.text_input("Ask a question:")
length = st.slider("Summary Length (max tokens)", min_value=50, max_value=1000, value=300, step=50)

if query:
    try:
        # 1️⃣ Plan
        steps = st.session_state.plan.plan(query)
        st.markdown("### 🔎 Plan")
        st.write(steps)

        # 2️⃣ Add user message
        st.session_state.mem.add_user_message(query)
        st.session_state.hist.append({"role": "user", "content": query})

        # 3️⃣ RAG query
        answer, sources = st.session_state.rag.query(query)

        # 4️⃣ Fallback if no RAG result
        if (not isinstance(answer, str) or not answer.strip()) and st.session_state.use_gpt_fallback:
            with st.spinner("💬 No RAG hit — asking ChatGPT…"):
                fallback_response = fallback_answer(query)
                if callable(fallback_response):
                    answer = "[Invalid fallback result: function instead of string]"
                else:
                    answer = str(fallback_response)
                sources = [{"metadata": {"source": "💬 ChatGPT (fallback)"}}]

        # 5️⃣ Summarize
        summary = st.session_state.summ.summarize(answer, max_tokens=length)

        # 6️⃣ Add AI message
        st.session_state.mem.add_ai_message(summary)
        st.session_state.hist.append({"role": "assistant", "content": summary})

        # 7️⃣ Display
        st.markdown("### 💬 Answer")
        st.write(summary)

        if sources:
            st.markdown("### 📚 Sources")
            for doc in sources:
                src = doc.metadata.get("source", "unknown")
                st.write(f"- {os.path.basename(src)}")

    except Exception as e:
        st.error(f"🚨 {e}")

# ─── Chat History ───
if st.checkbox("Show Chat History"):
    st.markdown("### 📝 History")
    for msg in st.session_state.hist:
        who = "You" if msg["role"] == "user" else "AI"
        st.write(f"**{who}:** {msg['content']}")

st.caption("© AI RAG Agent - powered with Streamlit, Chroma, OpenAI, and LangChain")
