from langchain.memory import ConversationBufferMemory
# If LangChain breaks this in future, change to langchain_core.memory
from langchain_openai import ChatOpenAI

class ChatMemory:
    """
    Wrapper around LangChain's ConversationBufferMemory.
    Stores chat history for conversation context.
    """

    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def add_user_message(self, message: str):
        self.memory.chat_memory.add_user_message(message)

    def add_ai_message(self, message: str):
        self.memory.chat_memory.add_ai_message(message)

    def get_memory(self):
        return self.memory

    def clear(self):
        self.memory.chat_memory.messages = []