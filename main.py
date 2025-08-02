# main entry point
import os
from modules import planner, rag_qa, calculator, web_search, summarizer, memory
from dotenv import load_dotenv

load_dotenv()

def main():
    print("=== AI Agent MCP Workflow ===")
    rag = rag_qa.RAGQA()
    mem = memory.MemoryManager()

    while True:
        query = input("\nAsk your question (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Goodbye!")
            break

        # Simple logic: if query contains math ops, use calculator
        math_ops = ['+', '-', '*', '/', '**']
        if any(op in query for op in math_ops):
            print("[Calculator Module]")
            result = calculator.eval_expr(query)
            print(f"Result: {result}")
            continue

        # If query contains 'search' keyword, do web search + summarize
        if "search" in query.lower():
            print("[Web Search Module]")
            query_text = query.lower().replace("search", "").strip()
            summary = web_search.search_and_summarize(query_text)
            print(f"Summary:\n{summary}")
            continue

        # Otherwise use RAG QA
        print("[RAG QA Module]")
        answer, sources = rag.query(query)
        print(f"Answer:\n{answer}")
        print(f"Sources: {sources}")

        # Save to memory
        from langchain.schema import Document
        mem.add_memory([Document(page_content=query)])
        mem.add_memory([Document(page_content=answer)])

if __name__ == "__main__":
    main()
