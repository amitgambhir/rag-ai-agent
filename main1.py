#main.py

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

# Add project root to Python path
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Import project modules
from modules.context import AgentContext
from modules.planner import Planner
from modules import rag_ingest, rag_qa, web_search
from modules.rag_qa import RAGQA

def main():
    # Initialize shared context and planner
    ctx = AgentContext()
    planner = Planner()
    rag = RAGQA()

    print("🤖 AI Agent Initialized.")

    # Define the workflow steps
    planner.add_task("ingest_documents")
    planner.add_task("query_documents")
    planner.add_task("web_search")

    # Loop until all tasks are done or user exits
    while True:
        current_task = planner.get_next_task()
        if not current_task:
            print("All tasks completed or waiting for user input.")
            break

        if current_task == "ingest_documents":
            print("\n📥 Ingesting documents...")
            rag_ingest.ingest_documents()
            ctx.set_task("document_ingestion_complete")
            planner.mark_task_completed("ingest_documents")

        elif current_task == "query_documents":
            print("\n🗨️ Ask me a question (or type 'exit' to quit):")
            user_input = input(">> ")

            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break

            ctx.add_chat("user", user_input)

            print("🔎 Searching ingested documents...")
            try:
                answer, sources = rag.query(user_input)
                if answer:
                    print(f"📄 Answer from documents:\n{answer}\n")
                    ctx.add_chat("agent", answer)
                else:
                    print("No relevant document answer found.")
                    planner.add_task("web_search")
                planner.mark_task_completed("query_documents")
            except Exception as e:
                print(f"[Error during document query: {e}]")
                planner.add_task("web_search")
                planner.mark_task_completed("query_documents")

        elif current_task == "web_search":
            print("🌐 Searching the web...")
            try:
                web_summary = web_search.search_and_summarize(user_input)
                print(f"🌐 Web summary:\n{web_summary}\n")
                ctx.add_chat("agent", web_summary)
            except Exception as e:
                print(f"[Error during web search: {e}]")
            planner.mark_task_completed("web_search")

        time.sleep(0.5)

if __name__ == "__main__":
    main()
