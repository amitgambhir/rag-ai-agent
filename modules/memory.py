# memory management
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

MEMORY_DB_DIR = "data/memory_store"

class MemoryManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.memory_db = Chroma(persist_directory=MEMORY_DB_DIR, embedding_function=self.embeddings)
    
    def add_memory(self, documents):
        """
        Add a list of Langchain Document objects to memory.
        """
        self.memory_db.add_documents(documents)
        self.memory_db.persist()
    
    def query_memory(self, query, k=5):
        """
        Query memory for relevant documents.
        """
        results = self.memory_db.similarity_search(query, k=k)
        return results

if __name__ == "__main__":
    # Example usage:
    from langchain.schema import Document

    memory = MemoryManager()
    docs = [
        Document(page_content="Remember this important fact about AI.", metadata={"source": "user"}),
        Document(page_content="The AI agent can learn from previous conversations.", metadata={"source": "user"})
    ]
    memory.add_memory(docs)
    results = memory.query_memory("facts about AI", k=2)
    for doc in results:
        print(doc.page_content)
