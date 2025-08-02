# modules/web_search.py
from duckduckgo_search import DDGS
from . import summarizer

def search_and_summarize(query: str) -> str:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(r.get("body", ""))
    full_text = " ".join(results)
    if not full_text.strip():
        return "No results found."
    return summarizer.summarize_text(full_text)

