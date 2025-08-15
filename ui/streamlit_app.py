# ui/streamlit_app.py

import os
import sys
import time
import gc
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv

# ─── Project root on sys.path ───
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─── Load env ───
load_dotenv(os.path.join(ROOT, ".env"))

# ─── App modules ───
from modules.rag_qa import RAGQA
from modules.summarizer import Summarizer
from modules.memory import ChatMemory
from modules.planner import Planner
from modules.fallback import fallback_answer
import modules.rag_ingest as rag_ingest
from modules.config import PERSIST_DIR
from modules.config import OPENAI_MODEL_FALLBACK as FALLBACK_MODEL

# ─── Auth ───
USERNAME = os.getenv("APP_USERNAME")
PASSWORD = os.getenv("APP_PASSWORD")
if not USERNAME or not PASSWORD:
    st.error("⚠️ Missing APP_USERNAME or APP_PASSWORD in .env")
    st.stop()

# Fallback model label (for badge); must match what fallback.py uses
# FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4")

# ─── Page config ───
st.set_page_config(page_title="AI Agent MCP", layout="wide")
st.title("🤖 AI Agent MCP")

# ─── Session state init ───
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

# ─── Login ───
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

# ─── Sidebar ───
st.sidebar.header("Settings & Tools")

# Refresh Vector Store (temp-dir build + atomic swap)
if st.sidebar.button("🔁 Refresh Vector Store"):
    try:
        # 1) Release current RAG/Chroma so SQLite files are unlocked
        st.session_state.rag = None
        gc.collect()
        time.sleep(0.5)

        # 2) Build in a temp directory first (avoids writing into a locked dir)
        with st.spinner("♻️ Rebuilding vector store…"):
            tmp_dir = tempfile.mkdtemp(prefix="vectorstore_")
            rag_ingest.ingest_documents(force_reload=True, output_dir=tmp_dir)

        # 3) Replace old store with new one
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        shutil.move(tmp_dir, PERSIST_DIR)

        # 4) Reload RAG
        st.session_state.rag = RAGQA(force_reload=True)
        st.sidebar.success("✅ Vector store refreshed and reloaded.")
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
        # 1) Plan (placeholder)
        steps = st.session_state.plan.plan(query)
        st.markdown("### 🔎 Plan")
        st.write(steps)

        # 2) Track user message
        st.session_state.mem.add_user_message(query)
        st.session_state.hist.append({"role": "user", "content": query})

        # 3) RAG
        with st.spinner("🔎 Searching documents…"):
            answer, sources = st.session_state.rag.query(query)

        # 4) Decide if RAG “hit” is good enough
        rag_hit = isinstance(answer, str) and bool(answer.strip()) and bool(sources)
        provenance = "RAG"  # default; will flip if we use GPT

        # 5) If RAG missed, optionally use GPT (based on checkbox)
        if not rag_hit:
            if st.session_state.use_gpt_fallback:
                with st.spinner(f"💬 No RAG hit — asking ChatGPT ({FALLBACK_MODEL})…"):
                    fb = fallback_answer(query)   # MUST be called
                    # Normalize fallback result to string
                    if callable(fb):
                        answer = "[Invalid fallback result: function]"
                    elif isinstance(fb, str):
                        answer = fb
                    else:
                        answer = str(fb)
                    sources = [{"metadata": {"source": f"💬 ChatGPT (fallback · {FALLBACK_MODEL})"}}]
                    provenance = "GPT"
            else:
                # Fallback off → provide a helpful message and keep sources empty
                answer = ("No relevant context found in your documents. "
                          "Enable **GPT fallback** in the sidebar to answer using general knowledge.")
                sources = []
                provenance = "NONE"

        # 6) Normalize answer BEFORE summarizing
        if callable(answer):
            answer = "[Internal error: answer was a function]"
        elif not isinstance(answer, str):
            answer = str(answer)

        # 7) Summarize
        summary = st.session_state.summ.summarize(answer, max_tokens=length)

        # 8) Track AI response
        st.session_state.mem.add_ai_message(summary)
        st.session_state.hist.append({"role": "assistant", "content": summary})

        # 9) Display with provenance badge
        st.markdown("### 💬 Answer")
        if provenance == "RAG":
            badge = "🧠 **RAG**"
        elif provenance == "GPT":
            badge = f"💬 **GPT fallback · model: {FALLBACK_MODEL}**"
        else:
            badge = "⚠️ **No context**"
        st.markdown(badge)
        st.write(summary)

        # 10) Sources (if any)
        if sources:
            st.markdown("### 📚 Sources")
            for doc in sources:
                # doc could be a langchain Document or dict; be defensive
                md = getattr(doc, "metadata", None) or getattr(doc, "__dict__", {}).get("metadata", {}) or {}
                src = md.get("source", "unknown")
                # For the fallback, we’ll show the label as-is
                if isinstance(src, str) and src.lower().endswith(".pdf"):
                    src = os.path.basename(src)
                st.write(f"- {src}")

    except Exception as e:
        st.error(f"🚨 {e}")

# ─── Chat History ───
if st.checkbox("Show Chat History"):
    st.markdown("### 📝 History")
    for msg in st.session_state.hist:
        who = "You" if msg["role"] == "user" else "AI"
        st.write(f"**{who}:** {msg['content']}")

st.caption("© AI RAG Agent - powered with Streamlit, Chroma, OpenAI, and LangChain")
