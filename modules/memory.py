from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

VECTOR_DB_DIR = "memory_db"

class MemoryManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=self.embeddings
        )

    def add_memory(self, documents):
        """
        documents: list of langchain.schema.Document
        """
        self.vectordb.add_documents(documents)
        self.vectordb.persist()

    def query_memory(self, query, k=10):
        retriever = self.vectordb.as_retriever(search_kwargs={"k": k})
        results = retriever.get_relevant_documents(query)
        return results
