import os
import logging
import shutil

from langchain_community.document_loaders import PlaywrightURLLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ─── Constants ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "..", "documents")
URL_FILE = os.path.join(DOCS_DIR, "urls.txt")
PERSIST_DIR = os.path.join(BASE_DIR, "..", "vectorstore")

# ─── Logging ───
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ─── Loaders ───

def load_urls(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def load_pdfs(directory):
    if not os.path.exists(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".pdf")
    ]

def ingest_documents(force_reload=True):
    if force_reload and os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    documents = []

    log.info("📦 Starting ingestion pipeline...")

    # Load URLs
    log.info("🔗 Loading web documents...")
    urls = load_urls(URL_FILE)
    for url in urls:
        try:
            log.info(f"   → Fetching: {url}")
            loader = PlaywrightURLLoader([url], remove_selectors=["header", "footer", "nav"])
            docs = loader.load()
            documents.extend(docs)
            log.info(f"     ✔ Loaded {len(docs)} documents.")
        except Exception as e:
            log.warning(f"     ❌ Error loading {url}: {e}")

    # Load PDFs
    log.info("📄 Loading PDF documents...")
    pdfs = load_pdfs(DOCS_DIR)
    for pdf in pdfs:
        try:
            log.info(f"   → Reading: {pdf}")
            loader = PyPDFLoader(pdf)
            docs = loader.load()
            documents.extend(docs)
            log.info(f"     ✔ Loaded {len(docs)} pages.")
        except Exception as e:
            log.warning(f"     ❌ Error loading PDF {pdf}: {e}")

    if not documents:
        log.warning("⚠️ No documents found to ingest. Skipping vectorstore creation.")
        return False

    # Chunking
    log.info("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = splitter.split_documents(documents)
    log.info(f"   → Total chunks created: {len(docs_split)}")

    # Embedding
    log.info("🧠 Generating embeddings and building vectorstore...")
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        docs_split,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    log.info(f"💾 Vectorstore saved at: {PERSIST_DIR}")
    log.info("✅ Ingestion completed successfully.")

    return True

if __name__ == "__main__":
    ingest_documents()
