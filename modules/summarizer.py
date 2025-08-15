import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage
from langchain.schema.runnable import RunnableSequence

from modules.config import OPENAI_MODEL_SUMMARY, TEMP_SUMMARY

load_dotenv()

class Summarizer:
    def __init__(self, temperature: float | None = None, model_name: str | None = None):
        self.llm = ChatOpenAI(
            temperature=TEMP_SUMMARY if temperature is None else temperature,
            model_name=model_name or OPENAI_MODEL_SUMMARY,
        )
        template = """
Summarize the following text briefly but thoroughly.
Keep the summary under {max_tokens} tokens.

Text:
{text}

Summary:
"""
        self.prompt = PromptTemplate(
            input_variables=["text", "max_tokens"],
            template=template.strip()
        )
        self.chain = RunnableSequence(self.prompt, self.llm)

    def summarize(self, text: str, max_tokens: int = 300) -> str:
        # Defensive normalization
        if callable(text):
            text = "[Internal error: received a function instead of text]"
        elif not isinstance(text, str):
            text = str(text)

        try:
            output = self.chain.invoke({"text": text, "max_tokens": max_tokens})
        except Exception as e:
            return f"[Summarizer Error: {e}]"

        # Normalize outputs from LangChain
        if isinstance(output, AIMessage):
            content = getattr(output, "content", None)
            if isinstance(content, str):
                return content.strip()
            return str(output).strip()

        if isinstance(output, list) and output and isinstance(output[0], AIMessage):
            content = getattr(output[0], "content", None)
            if isinstance(content, str):
                return content.strip()
            return str(output[0]).strip()

        if callable(output):
            return "[Summarizer Error: output was a function]"

        return str(output).strip()


def summarize_text(text: str, response_length: int = 300) -> str:
    return Summarizer().summarize(text, max_tokens=response_length)
