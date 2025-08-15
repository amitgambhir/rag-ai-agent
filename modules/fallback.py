from langchain_openai import ChatOpenAI
from modules.config import OPENAI_MODEL_FALLBACK, TEMP_FALLBACK

SYSTEM = "You are a helpful AI assistant. If documents are unavailable, answer using your general knowledge."

PROMPT = """System: {system}
User: {question}
Assistant:"""

class GPTFallback:
    def __init__(self, model_name: str | None = None, temperature: float | None = None):
        self.llm = ChatOpenAI(
            model_name=model_name or OPENAI_MODEL_FALLBACK,
            temperature=TEMP_FALLBACK if temperature is None else temperature,
        )

    def answer(self, question: str) -> str:
        prompt = PROMPT.format(system=SYSTEM, question=question)
        try:
            resp = self.llm.invoke(prompt)
            content = getattr(resp, "content", None)
            if not isinstance(content, str):
                return "[Fallback Error: Invalid response content]"
            return content.strip()
        except Exception as e:
            return f"[Fallback Error: {e}]"


def fallback_answer(question: str) -> str:
    return GPTFallback().answer(question)
