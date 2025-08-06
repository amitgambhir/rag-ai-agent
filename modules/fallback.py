# modules/fallback.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

# System prompt: instruct GPT to answer as best as it can, citing it's a fallback
SYSTEM = """You are a knowledgeable assistant. 
If the user’s question cannot be answered from the provided documents, answer using your own knowledge. 
Be concise and clear."""

PROMPT = PromptTemplate.from_template(
    """{system}

User Question:
{question}

Your Answer:"""
)

class GPTFallback:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.0):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    def answer(self, question: str) -> str:
        prompt = PROMPT.format(system=SYSTEM, question=question)
        # invoke returns a AIMessage with .text
        resp = self.llm.invoke({"prompt": prompt})
        return getattr(resp, "text", str(resp)).strip()


# Convenience function
def fallback_answer(question: str) -> str:
    return GPTFallback().answer(question)
