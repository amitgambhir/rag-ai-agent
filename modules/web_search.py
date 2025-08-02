# modules/web_search.py

from duckduckgo_search import ddg
from . import summarizer

def search_and_summarize(query):
    try:
        results = ddg(query, max_results=5)
        if not results:
            return "No results found."
        combined_text = " ".join([item.get("body", "") for item in results])
        return summarizer.summarize_text(combined_text)
    except Exception as e:
        return f"Error during search: {e}"
