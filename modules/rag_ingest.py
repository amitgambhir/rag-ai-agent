import sys
import os
import logging
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.document_loaders import PlaywrightURLLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from modules.rag_qa import PERSIST_DIR, OpenAIEmbeddings
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def load_urls(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def load_pdfs(directory):
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

    # Load web pages using Playwright
    log.info("🔗 Loading web documents...")
    url_file = os.path.join("documents", "urls.txt")
    urls = load_urls(url_file)
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
    pdfs = load_pdfs("documents")
    for pdf in pdfs:
        try:
            log.info(f"   → Reading: {pdf}")
            loader = PyPDFLoader(pdf)
            docs = loader.load()
            documents.extend(docs)
            log.info(f"     ✔ Loaded {len(docs)} pages.")
        except Exception as e:
            log.warning(f"     ❌ Error loading PDF {pdf}: {e}")

    # Chunk documents
    log.info("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = splitter.split_documents(documents)
    log.info(f"   → Total chunks created: {len(docs_split)}")

    # Build vectorstore
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
