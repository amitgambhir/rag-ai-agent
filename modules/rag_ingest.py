import os
import glob
from dotenv import load_dotenv

# Load .env early
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
# Use WebBaseLoader for robust URL ingestion
try:
    from langchain.document_loaders import WebBaseLoader
except ImportError:
    from langchain_community.document_loaders import UnstructuredURLLoader as WebBaseLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

DATA_PATH = "documents"
PERSIST_DIRECTORY = "vectorstore"


def ingest_pdfs():
    """Load and return documents from PDF files."""
    docs = []
    for path in glob.glob(os.path.join(DATA_PATH, "*.pdf")):
        print(f"Loading PDF: {path}")
        loader = PyPDFLoader(path)
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = os.path.basename(path)
        docs.extend(loaded)
    return docs


def ingest_urls():
    """Fetch and return documents from URLs listed in urls.txt."""
    file_path = os.path.join(DATA_PATH, "urls.txt")
    if not os.path.exists(file_path):
        print(f"No URLs file at {file_path}, skipping URL ingestion.")
        return []

    with open(file_path) as f:
        urls = [u.strip() for u in f if u.strip()]
    if not urls:
        print("URLs file empty, skipping URL ingestion.")
        return []

    docs = []
    for url in urls:
        print(f"Loading URL: {url}")
        try:
            loader = WebBaseLoader(url)
            loaded = loader.load()
            # Assign metadata source per document
            for doc in loaded:
                doc.metadata["source"] = url
            docs.extend(loaded)
        except Exception as e:
            print(f"Failed to load URL {url}: {e}")
    return docs


def ingest_documents():
    """Orchestrate ingestion of PDFs and URLs into a persistent vector store."""
    print("=== Ingestion Start ===")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    pdf_docs = ingest_pdfs()
    url_docs = ingest_urls()

    all_docs = pdf_docs + url_docs
    print(f"Total docs loaded: {len(all_docs)}")
    if not all_docs:
        print("No documents to ingest, exiting.")
        return

    # Optional: preview first doc content
    print(f"Sample content: {all_docs[0].page_content[:200]}
...")

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks to embed: {len(chunks)}")

    # Embed and persist
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    # Chroma auto-persists; outdated .persist() removed
    print("=== Ingestion Complete ===")


if __name__ == "__main__":
    ingest_documents()
```python
import os
import glob
from dotenv import load_dotenv

# Load .env early
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

DATA_PATH = "documents"
PERSIST_DIRECTORY = "vectorstore"


def ingest_pdfs():
    docs = []
    for path in glob.glob(os.path.join(DATA_PATH, "*.pdf")):
        print(f"Loading PDF: {path}")
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    return docs


def ingest_urls():
    file = os.path.join(DATA_PATH, "urls.txt")
    if not os.path.exists(file):
        print(f"No URLs file at {file}, skipping.")
        return []
    with open(file) as f:
        urls = [u.strip() for u in f if u.strip()]
    if not urls:
        print("URLs file empty, skipping.")
        return []
    print(f"Loading URLs: {urls}")
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    # add metadata
    for doc, url in zip(docs, urls):
        doc.metadata['source'] = url
    return docs


def ingest_documents():
    print("=== Ingestion Start ===")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    pdfs = ingest_pdfs()
    urls = ingest_urls()
    all_docs = pdfs + urls
    print(f"Total docs loaded: {len(all_docs)}")
    if not all_docs:
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks: {len(chunks)}")

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    # persistence now automatic
    print("=== Ingestion Complete ===")


if __name__ == "__main__":
    ingest_documents()