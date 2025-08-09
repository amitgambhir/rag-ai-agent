import os
import shutil
import logging

from dotenv import load_dotenv
load_dotenv()

# LangChain / loaders
from langchain_community.document_loaders import PlaywrightURLLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Project config
try:
    from modules.config import PERSIST_DIR  # preferred single source of truth
except Exception:
    # Fallback: vectorstore at project root if config import fails
    PERSIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vectorstore"))

# Logging (simple)
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _load_urls(file_path: str):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def _load_pdfs(directory: str):
    if not os.path.isdir(directory):
        return []
    pdfs = []
    for name in os.listdir(directory):
        if name.lower().endswith(".pdf"):
            pdfs.append(os.path.join(directory, name))
    return pdfs


def ingest_documents(force_reload: bool = True, output_dir: str | None = None) -> bool:
    """
    Build embeddings and a Chroma vectorstore at `output_dir` (or PERSIST_DIR).
    Returns True on success (exceptions will bubble to caller).
    """
    persist_root = output_dir or PERSIST_DIR

    # Ensure clean & writable target directory
    if force_reload and os.path.exists(persist_root):
        shutil.rmtree(persist_root)

    os.makedirs(persist_root, exist_ok=True)
    try:
        # make sure we can write here (macOS sometimes preserves weird perms)
        os.chmod(persist_root, 0o775)
    except Exception:
        pass

    documents = []

    log.info("📦 Starting ingestion pipeline...")

    # ── URLs
    url_file = os.path.join("documents", "urls.txt")
    urls = _load_urls(url_file)
    if urls:
        log.info("🔗 Loading web documents...")
        try:
            loader = PlaywrightURLLoader(urls, remove_selectors=["header", "footer", "nav"])
            docs = loader.load()
            documents.extend(docs)
            log.info(f"   ✔ Loaded {len(docs)} documents from URLs.")
        except Exception as e:
            log.warning(f"   ❌ URL ingestion warning: {e}")

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

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_root,
    )

    # Force persistence & drop references to release SQLite handles
    try:
        vectordb.persist()
    finally:
        del vectordb

    log.info(f"💾 Vectorstore saved at: {persist_root}")
    log.info("✅ Ingestion completed successfully.")
    return True


if __name__ == "__main__":
    ingest_documents(force_reload=True)
