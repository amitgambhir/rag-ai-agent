# RAG ingestion script
import os
from langchain.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Paths & settings
DOCUMENTS_DIR = "documents"
VECTOR_DB_DIR = "data/vector_store"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def ingest_pdfs():
    pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".pdf")]
    docs = []
    for pdf in pdf_files:
        loader = PyPDFLoader(os.path.join(DOCUMENTS_DIR, pdf))
        pages = loader.load()
        docs.extend(pages)
    return docs

def ingest_urls():
    urls_file = os.path.join(DOCUMENTS_DIR, "urls.txt")
    if not os.path.exists(urls_file):
        return []
    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f.readlines() if line.strip()]
    if not urls:
        return []
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    return docs

def ingest_documents():
    print("Starting ingestion of PDFs and URLs...")
    pdf_docs = ingest_pdfs()
    url_docs = ingest_urls()
    all_docs = pdf_docs + url_docs

    if not all_docs:
        print("No documents found to ingest.")
        return

    print(f"Ingesting {len(all_docs)} documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    split_docs = text_splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )
    vectordb.add_documents(split_docs)
    vectordb.persist()

    print("Ingestion complete and vector store persisted.")

if __name__ == "__main__":
    ingest_documents()
