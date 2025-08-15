import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

from modules.config import (
    PERSIST_DIR,
    OPENAI_MODEL_CHAT,
    TEMP_CHAT,
    RETRIEVER_K,
    SIMILARITY_THRESHOLD,
)

load_dotenv()

# Heuristics for detecting “polite non-answers”
NON_ANSWER_PHRASES = (
    "does not include information",
    "doesn't include information",
    "does not contain information",
    "doesn't contain information",
    "no information about",
    "not in the provided context",
    "not found in the context",
    "the text does not provide",
    "the provided context does not",
    "the given context does not",
)

def _looks_like_non_answer(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip().lower()
    return any(p in t for p in NON_ANSWER_PHRASES)


class RAGQA:
    """
    RAG pipeline wrapper.
    - Loads persisted Chroma DB from PERSIST_DIR
    - Uses a score-gated retriever to avoid weak matches blocking fallback
    - Returns (answer, sources) where answer is always a string
    """

    def __init__(self, force_reload: bool = False, temperature: float | None = None, retriever_k: int | None = None):
        self.temperature = TEMP_CHAT if temperature is None else temperature
        self.retriever_k = RETRIEVER_K if retriever_k is None else retriever_k
        self.embeddings = OpenAIEmbeddings()
        self.vectordb = None
        self.retriever = None
        self.qa = None

        if not os.path.exists(PERSIST_DIR):
            raise FileNotFoundError(
                f"❌ Vector store not found at {PERSIST_DIR}. Please run ingestion first."
            )

        self._load_vectorstore()
        self._build_chain()

    def _load_vectorstore(self):
        self.vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=self.embeddings,
        )

        # Score-gated retriever to reject weak/irrelevant matches
        self.retriever = self.vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.retriever_k,
                "score_threshold": SIMILARITY_THRESHOLD,  # e.g., 0.2–0.4
            },
        )

    def _build_chain(self):
        llm = ChatOpenAI(temperature=self.temperature, model_name=OPENAI_MODEL_CHAT)
        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
        )

    def update_model_settings(self, temperature: float | None = None, retriever_k: int | None = None):
        if temperature is not None:
            self.temperature = temperature
        if retriever_k is not None:
            self.retriever_k = retriever_k

        if self.vectordb:
            self.retriever = self.vectordb.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.retriever_k,
                    "score_threshold": SIMILARITY_THRESHOLD,
                },
            )
        self._build_chain()

    def query(self, question: str) -> tuple[str, list]:
        """
        Return (answer, sources). If no sufficiently relevant docs or the chain
        produces a "polite non-answer", return ('', []) so the UI can trigger fallback.
        """
        if not question:
            return "", []

        try:
            # Pre-check with explicit scored similarity to guard further
            scored = self.vectordb.similarity_search_with_relevance_scores(
                question, k=self.retriever_k
            )
            prelim_docs = [doc for (doc, score) in scored if (score or 0) >= SIMILARITY_THRESHOLD]

            if not prelim_docs:
                return "", []

            result = self.qa.invoke({"query": question})
        except Exception as e:
            return f"[RAG Query Error: {e}]", []

        if isinstance(result, dict):
            answer = result.get("result", "") or ""
            sources = result.get("source_documents", []) or prelim_docs
        else:
            answer = result or ""
            sources = prelim_docs

        if _looks_like_non_answer(answer):
            return "", []

        if callable(answer):
            answer = "[Invalid result: received a function instead of string]"

        return str(answer).strip(), sources


def reload_vectorstore():
    # Kept for compatibility; UI handles rebuild using rag_ingest
    try:
        import shutil
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        return True, "Vector store directory cleared. Re-ingest to rebuild."
    except Exception as e:
        return False, str(e)
