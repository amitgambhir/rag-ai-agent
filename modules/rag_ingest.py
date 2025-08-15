# modules/rag_ingest.py

import os
import shutil
import logging
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

# LangChain loaders & vector store
from langchain_community.document_loaders import (
    PlaywrightURLLoader,
    WebBaseLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Project config (single source of truth)
try:
    from modules.config import PERSIST_DIR
except Exception:
    # Fallback: vectorstore at project root if config import fails
    PERSIST_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "vectorstore")
    )

# ─── Logging ───
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_urls(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _load_pdfs(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    pdfs = []
    for name in os.listdir(directory):
        if name.lower().endswith(".pdf"):
            pdfs.append(os.path.join(directory, name))
    return pdfs


def _ingest_url(url: str) -> List:
    """
    Try Playwright first (DOM-aware), then fallback to WebBaseLoader (requests + bs4)
    if Playwright isn't available/blocked. Keeps logs clean across versions.
    """
    docs = []
    # 1) Playwright attempt (no goto_options to keep compat across versions)
    try:
        pw_loader = PlaywrightURLLoader(
            urls=[url],
            remove_selectors=["header", "footer", "nav"],
        )
        docs = pw_loader.load()
        if docs:
            log.info(f"   ✔ (Playwright) {url} → {len(docs)} doc(s)")
            return docs
    except Exception as e:
        log.warning(f"   ⚠️ Playwright failed for {url}: {e}")

    # 2) Fallback: WebBaseLoader with User-Agent
    try:
        ua = os.getenv(
            "USER_AGENT",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
        wb_loader = WebBaseLoader(
            url,
            requests_kwargs={"headers": {"User-Agent": ua}},
        )
        docs = wb_loader.load()
        if docs:
            log.info(f"   ✔ (WebBaseLoader) {url} → {len(docs)} doc(s)")
            return docs
        else:
            log.warning(f"   ⚠️ No content parsed from {url} via WebBaseLoader")
    except Exception as e:
        log.warning(f"   ❌ WebBaseLoader failed for {url}: {e}")

    return []

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def ingest_documents(force_reload: bool = True, output_dir: Optional[str] = None) -> bool:
    """
    Build embeddings and a Chroma vectorstore at `output_dir` (or PERSIST_DIR).
    Returns True on success (exceptions bubble up to caller).
    """
    persist_dir = output_dir or PERSIST_DIR

    # Clean target if asked
    if force_reload and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    # Ensure target exists & is writable
    os.makedirs(persist_dir, exist_ok=True)
    try:
        os.chmod(persist_dir, 0o775)
    except Exception:
        pass

    documents = []

    log.info("📦 Starting ingestion pipeline...")

    # ── URLs (resilient path)
    url_file = os.path.join("documents", "urls.txt")
    urls = _load_urls(url_file)
    if urls:
        log.info("🔗 Loading web documents...")
        total = 0
        for url in urls:
            try:
                docs = _ingest_url(url)
                if docs:
                    documents.extend(docs)
                    total += len(docs)
            except Exception as e:
                log.warning(f"   ❌ Error fetching or processing {url}: {e}")
        log.info(f"   → URL ingestion complete. Loaded {total} document(s) total.")

    # ── PDFs
    log.info("📄 Loading PDF documents...")
    pdfs = _load_pdfs("documents")
    for pdf in pdfs:
        try:
            log.info(f"   → Reading: {pdf}")
            loader = PyPDFLoader(pdf)
            docs = loader.load()
            documents.extend(docs)
            log.info(f"     ✔ Loaded {len(docs)} pages.")
        except Exception as e:
            log.warning(f"     ❌ Error loading PDF {pdf}: {e}")

    # ── Chunk
    log.info("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    log.info(f"   → Total chunks created: {len(chunks)}")

    # ── Vectorstore
    log.info("🧠 Generating embeddings and building vectorstore...")
    embeddings = OpenAIEmbeddings()

    # Chroma 0.4+ persists automatically when persist_directory is set
    Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    log.info(f"💾 Vectorstore saved at: {persist_dir}")
    log.info("✅ Ingestion completed successfully.")
    return True


if __name__ == "__main__":
    ingest_documents(force_reload=True)
