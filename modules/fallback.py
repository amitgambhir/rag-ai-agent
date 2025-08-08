from langchain_openai import ChatOpenAI

# System prompt
SYSTEM = "You are a helpful AI assistant."
PROMPT = """System: {system}
User: {question}
Assistant:"""

class GPTFallback:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.0):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    def answer(self, question: str) -> str:
        prompt = PROMPT.format(system=SYSTEM, question=question)
        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", None)
            if not isinstance(content, str):
                return "[Fallback Error: Invalid response content]"
            return content.strip()
        except Exception as e:
            return f"[Fallback Error: {e}]"

# Convenience method
def fallback_answer(question: str) -> str:
    return GPTFallback().answer(question)
