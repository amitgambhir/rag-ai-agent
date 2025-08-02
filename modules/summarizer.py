# summarizer module

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0.3, model_name="gpt-4")  # Or use gpt-3.5-turbo for cost efficiency

template = """
You are an expert summarizer.

Summarize the following text briefly and clearly:

{text}
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)

summarize_chain = LLMChain(llm=llm, prompt=prompt)

def summarize_text(text: str) -> str:
    return summarize_chain.run(text)

if __name__ == "__main__":
    sample_text = "LangChain enables building powerful LLM-powered applications..."
    print("Summary:", summarize_text(sample_text))
