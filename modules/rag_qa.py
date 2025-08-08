import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "..", "vectorstore")

class RAGQA:
    def __init__(self, force_reload=False, temperature=0.0, retriever_k=3):
        self.temperature = temperature
        self.retriever_k = retriever_k
        self.embeddings = OpenAIEmbeddings()
        self.vectordb = None
        self.retriever = None
        self.qa = None

        # Build or load vector DB
        if force_reload:
            from modules.rag_ingest import ingest_documents
            ingest_documents()

        if os.path.exists(PERSIST_DIR):
            self._load_vectorstore()
        else:
            raise FileNotFoundError(f"❌ Vector store not found at {PERSIST_DIR}. Please run ingestion first.")

        self._build_chain()

    def _load_vectorstore(self):
        self.vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": self.retriever_k})

    def _build_chain(self):
        llm = ChatOpenAI(temperature=self.temperature, model_name="gpt-4")
        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )

    def update_model_settings(self, temperature=None, retriever_k=None):
        if temperature is not None:
            self.temperature = temperature
        if retriever_k is not None:
            self.retriever_k = retriever_k
        if self.vectordb:
            self.retriever = self.vectordb.as_retriever(search_kwargs={"k": self.retriever_k})
        self._build_chain()

    def query(self, question: str) -> tuple[str, list]:
        if not question:
            return "", []

        try:
            result = self.qa.invoke({"query": question})
        except Exception as e:
            return f"[RAG Query Error: {e}]", []

        # Parse result safely
        answer = ""
        sources = []

        if isinstance(result, dict):
            answer = result.get("result", "")
            sources = result.get("source_documents", [])
        else:
            answer = result or ""
            sources = []

        # Normalize edge cases
        if callable(answer):
            answer = "[Invalid result: received a function instead of a string]"

        return str(answer).strip(), sources


# For Streamlit UI
def reload_vectorstore():
    try:
        import shutil
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        return True, "Vector store directory cleared."
    except Exception as e:
        return False, str(e)
