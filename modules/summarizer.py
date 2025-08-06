import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv
load_dotenv()

class Summarizer:
    def __init__(self, temperature=0.0, model_name="gpt-4"):
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)
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
        # Invoke the sequence
        output = self.chain.invoke({"text": text, "max_tokens": max_tokens})

        # Normalize whether it's an AIMessage or list of them
        if isinstance(output, AIMessage):
            msg = output
        elif isinstance(output, list) and output and isinstance(output[0], AIMessage):
            msg = output[0]
        else:
            # Fallback: treat output as list of strings
            msg = output[0] if isinstance(output, list) else output

        # Extract text attribute if available, else string-cast
        return getattr(msg, "text", str(msg)).strip()

# Convenience wrapper
def summarize_text(text: str, response_length: int = 300) -> str:
    summarizer = Summarizer()
    return summarizer.summarize(text, max_tokens=response_length)
