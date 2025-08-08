# demo_workflow.py
import os
import time
import argparse
import shutil

from modules.context import AgentContext
from modules.rag_ingest import ingest_documents
from modules.rag_qa import RAGQA
from modules.fallback import fallback_answer
from modules.config import PERSIST_DIR as CHROMA_DB_DIR

def main():
    parser = argparse.ArgumentParser(description="AI Agent MCP CLI Demo")
    parser.add_argument(
        "--gpt-fallback", action="store_true",
        help="Enable GPT fallback when RAG returns nothing"
    )
    parser.add_argument(
        "--rebuild-db", action="store_true",
        help="Force rebuild of the vector DB before starting"
    )
    args = parser.parse_args()

    # Optional: wipe vectorstore if requested
    if args.rebuild_db and os.path.exists(CHROMA_DB_DIR):
        print("🧹 Wiping existing vector store...")
        shutil.rmtree(CHROMA_DB_DIR)

    use_fallback = args.gpt_fallback or \
        os.getenv("FALLBACK_WEB_SEARCH", "false").lower() == "true"

    # 0️⃣ Ingest
    print("📥 Ingesting documents…")
    ingest_documents(force_reload=False)

    # 1️⃣ Initialize
    ctx = AgentContext()
    rag = RAGQA(force_reload=False)
    print("🤖 AI Agent MCP Initialized.\n")

    # 2️⃣ Chat loop
    while True:
        q = input("🗨️  Your question (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        ctx.add_chat("user", q)

        # 3️⃣ RAG
        print("🔎 Searching documents…")
        try:
            ans, srcs = rag.query(q)
        except Exception as e:
            ans, srcs = f"[Error: {e}]", []

        # Normalize the answer
        normalized_ans = ans.strip().lower() if ans else ""

        # List of fallback-worthy phrases
        fallback_phrases = [
            "i'm sorry, but the provided context",
            "the text does not provide information",
            "the provided context does not include",
            "no relevant documents found",
            "no relevant context was found",
        ]

        # Check if RAG answer is meaningful or fallback-worthy
        if normalized_ans and not any(p in normalized_ans for p in fallback_phrases):
            print(f"📄 RAG Answer:\n{ans}\n")
            ctx.add_chat("agent", ans)
        else:
            # Trigger fallback
            if use_fallback:
                print("💬 No useful RAG answer — asking ChatGPT (fallback)…")
                try:
                    fallback = fallback_answer(q)
                    print(f"💬 ChatGPT:\n{fallback}\n")
                    ctx.add_chat("agent", fallback)
                except Exception as e:
                    print(f"[Fallback Error: {e}]")
            else:
                print("⚠️  No useful RAG answer and fallback is OFF.\n")

        
        # 4️⃣ GPT fallback
        if use_fallback:
            print("💬 No RAG hit — ChatGPT fallback…")
            try:
                ans = fallback_answer(q)
            except Exception as e:
                ans = f"[GPT fallback error: {e}]"
            print(f"💬 ChatGPT:\n{ans}\n")
            ctx.add_chat("agent", ans)
        else:
            print("⚠️  No answer found and fallback is OFF.\n")

        time.sleep(0.5)

if __name__ == "__main__":
    main()
