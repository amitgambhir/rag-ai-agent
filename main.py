# main entry point
import os
from dotenv import load_dotenv
from modules import planner, rag_qa, calculator, web_search, summarizer, memory
from langchain.schema import Document

load_dotenv()

def main():
    print("=== AI Agent MCP Workflow ===")
    rag = rag_qa.RAGQA()
    mem = memory.MemoryManager()

    while True:
        try:
            query = input("\nAsk your question (or type 'exit' or 'help'): ").strip()
            if query.lower() == 'exit':
                print("Goodbye!")
                break
            elif query.lower() == 'help':
                print("\nInstructions:\n- Use math ops like '+', '-', '*', '/' for calculator\n- Type 'search <query>' to run a web search\n- Ask anything else to use the document-based QA (RAG)")
                continue

            math_ops = ['+', '-', '*', '/', '**']
            if any(op in query for op in math_ops):
                print("[Calculator Module]")
                result = calculator.eval_expr(query)
                print(f"Result: {result}")
                continue

            if "search" in query.lower():
                print("[Web Search Module]")
                query_text = query.lower().replace("search", "").strip()
                summary = web_search.search_and_summarize(query_text)
                print(f"Summary:\n{summary}")
                continue

            print("[RAG QA Module]")
            try:
                answer, sources = rag.query(query)
                print(f"Answer:\n{answer}")
                print(f"Sources: {sources}")
                mem.add_memory([Document(page_content=f"Q: {query}\nA: {answer}")])
            except Exception as e:
                print(f"[Error] {e}")

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break

if __name__ == "__main__":
    main()