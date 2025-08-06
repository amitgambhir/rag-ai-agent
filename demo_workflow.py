#!/usr/bin/env python3
import os
import time
import argparse

from modules.context import AgentContext
from modules.rag_ingest import ingest_documents
from modules.rag_qa import RAGQA
from modules.fallback import fallback_answer

def main():
    parser = argparse.ArgumentParser(description="AI Agent MCP CLI Demo")
    parser.add_argument(
        "--gpt-fallback", action="store_true",
        help="Enable GPT fallback when RAG returns nothing"
    )
    args = parser.parse_args()
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

        if ans.strip():
            print(f"📄 RAG Answer:\n{ans}\n")
            ctx.add_chat("agent", ans)
            continue

        # 4️⃣ GPT fallback
        if use_fallback:
            print("💬 No RAG hit—ChatGPT fallback…")
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
