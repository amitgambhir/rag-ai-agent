# web search module
import requests
from duckduckgo_search import ddg
from . import summarizer

def search_and_summarize(query):
    results = ddg(query, max_results=5)
    if not results:
        return "No results found."
    combined_text = " ".join([item["body"] for item in results])
    return summarizer.summarize_text(combined_text)


    combined_text = ""
    for idx, res in enumerate(results):
        combined_text += f"Result {idx+1} Title: {res['title']}\n"
        combined_text += f"Snippet: {res['body']}\n"
        combined_text += f"URL: {res['href']}\n\n"

    prompt = f"Summarize the following search results for the query: {query}\n\n{combined_text}"

    messages = [HumanMessage(content=prompt)]
    summary = llm(messages).content
    return summary

if __name__ == "__main__":
    test_query = "Latest advancements in AI in 2025"
    print(search_and_summarize(test_query))
