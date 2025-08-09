# modules/rag_qa.py

import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "..", "vectorstore")

# ---- Heuristics for detecting "polite non-answers" from the chain ----
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

    def __init__(self, force_reload: bool = False, temperature: float = 0.0, retriever_k: int = 3):
        self.temperature = temperature
        self.retriever_k = retriever_k
        self.embeddings = OpenAIEmbeddings()
        self.vectordb = None
        self.retriever = None
        self.qa = None

        # We assume ingestion is handled externally (CLI or Streamlit).
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
                "score_threshold": 0.2,  # tweak as needed (0.2-0.4 are reasonable)
            },
        )

    def _build_chain(self):
        llm = ChatOpenAI(temperature=self.temperature, model_name="gpt-4")
        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,  # ensure we get sources back
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
                    "score_threshold": 0.2,
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
            # 1) Pre-check with explicit scored similarity to guard further
            scored = self.vectordb.similarity_search_with_relevance_scores(
                question, k=self.retriever_k
            )
            # Keep only docs above the same threshold we used for the retriever
            score_threshold = 0.2
            prelim_docs = [doc for (doc, score) in scored if (score or 0) >= score_threshold]

            if not prelim_docs:
                # No meaningful grounding → let UI trigger GPT fallback
                return "", []

            # 2) Ask the chain (constructed with return_source_documents=True)
            result = self.qa.invoke({"query": question})

        except Exception as e:
            return f"[RAG Query Error: {e}]", []

        # 3) Normalize outputs from chain
        if isinstance(result, dict):
            answer = result.get("result", "") or ""
            sources = result.get("source_documents", []) or prelim_docs
        else:
            answer = result or ""
            sources = prelim_docs

        # 4) If the chain gave a "polite non-answer", force fallback
        if _looks_like_non_answer(answer):
            return "", []

        if callable(answer):
            answer = "[Invalid result: received a function instead of string]"

        return str(answer).strip(), sources


# Optional helper retained for compatibility (UI now handles refresh+ingest)
def reload_vectorstore():
    try:
        import shutil
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        return True, "Vector store directory cleared. Re-ingest to rebuild."
    except Exception as e:
        return False, str(e)
