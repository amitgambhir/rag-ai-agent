from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import os
import glob

DATA_PATH = "documents"
PERSIST_DIRECTORY = "vectorstore"

def ingest_pdfs():
    documents = []
    for filepath in glob.glob(os.path.join(DATA_PATH, "*.pdf")):
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        documents.extend(docs)
    return documents

def ingest_urls():
    urls_file = os.path.join(DATA_PATH, "urls.txt")
    if not os.path.exists(urls_file):
        return []

    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    loader = UnstructuredURLLoader(urls=urls)
    return loader.load()

def ingest_documents():
    print("Starting ingestion of PDFs and URLs...")

    pdf_docs = ingest_pdfs()
    url_docs = ingest_urls()

    all_docs = pdf_docs + url_docs
    print(f"Ingesting {len(all_docs)} documents...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    vectordb.persist()
    print("Ingestion complete and vector store persisted.")

if __name__ == "__main__":
    ingest_documents()
