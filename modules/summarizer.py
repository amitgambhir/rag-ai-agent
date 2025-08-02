# summarizer module
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(temperature=0.3)

def summarize_text(text, max_length=300):
    """
    Summarize the input text into a concise form.
    """
    prompt = (
        f"Please provide a concise summary (under {max_length} words) of the following text:\n\n{text}"
    )
    messages = [HumanMessage(content=prompt)]
    summary = llm(messages).content
    return summary

if __name__ == "__main__":
    sample_text = (
        "Artificial Intelligence (AI) has made tremendous progress in recent years, "
        "transforming many industries including healthcare, finance, and transportation. "
        "The ability of machines to learn and perform complex tasks continues to evolve rapidly."
    )
    print(summarize_text(sample_text))
