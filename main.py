from modules.planner import Planner
import sys
import os
import time
from dotenv import load_dotenv
load_dotenv()


def main():
    # Initialize shared context
    ctx = AgentContext()
    planner = Planner()

    print("🤖 AI Agent Initialized.")

    # Define the workflow steps
    planner.add_task("ingest_documents")
    planner.add_task("query_documents")
    planner.add_task("web_search")

    # Loop until all tasks done or user exits
    while True:
        current_task = planner.next_task()
        if not current_task:
            print("All tasks completed or waiting for user input.")
            break

        if current_task == "ingest_documents":
            print("\n📥 Ingesting documents...")
            rag_ingest.ingest_documents()
            ctx.set_task("document_ingestion_complete")
            planner.complete_task("ingest_documents")

        elif current_task == "query_documents":
            print("\n🗨️ Ask me a question (or type 'exit' to quit):")
            user_input = input(">> ")

            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break

            ctx.add_chat("user", user_input)

            print("🔎 Searching ingested documents...")
            try:
                response = rag_qa.query_ingested_docs(user_input)
                if response:
                    print(f"📄 Answer from documents:\n{response}\n")
                    ctx.add_chat("agent", response)
                else:
                    print("No relevant document answer found.")
                    planner.add_task("web_search")
                planner.complete_task("query_documents")
            except Exception as e:
                print(f"[Error during document query: {e}]")
                planner.add_task("web_search")
                planner.complete_task("query_documents")

        elif current_task == "web_search":
            print("🌐 Searching the web...")
            try:
                web_summary = web_search.search_and_summarize(user_input)
                print(f"🌐 Web summary:\n{web_summary}\n")
                ctx.add_chat("agent", web_summary)
            except Exception as e:
                print(f"[Error during web search: {e}]")
            planner.complete_task("web_search")

        # Pause or add other logic here if needed

        # Optional: small delay to keep CLI responsive
        time.sleep(0.5)
