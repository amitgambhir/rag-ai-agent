# main.py

from context import AgentContext
from modules import rag_ingest, rag_qa, web_search, planner  # planner is empty now
import time

def main():
    # Initialize shared context
    ctx = AgentContext()
    print("🤖 AI Agent Initialized.")

    # Step 1: Ingest Documents (PDFs + URLs)
    print("\n📥 Ingesting documents...")
    rag_ingest.ingest_documents()
    ctx.set_task("document_ingestion_complete")

    # Step 2: Ask the user a question
    while True:
        print("\n🗨️ Ask me a question (or type 'exit' to quit):")
        user_input = input(">> ")

        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        ctx.add_chat("user", user_input)

        # Step 3: Answer from PDF vector store
        print("🔎 Searching ingested documents...")
        try:
            response = rag_qa.query_ingested_docs(user_input)
        except Exception as e:
            response = f"[Error during document query: {e}]"

        if response:
            print(f"📄 Answer from documents:\n{response}\n")
            ctx.add_chat("agent", response)
            continue

        # Step 4: If no good RAG answer, do web search
        print("🌐 No answer found. Searching the web...")
        try:
            web_summary = web_search.search_and_summarize(user_input)
        except Exception as e:
            web_summary = f"[Error during web search: {e}]"

        print(f"🌐 Web summary:\n{web_summary}\n")
        ctx.add_chat("agent", web_summary)

        time.sleep(1)  # Throttle loop for readability

if __name__ == "__main__":
    main()
