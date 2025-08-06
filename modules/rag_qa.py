import os

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

PERSIST_DIR = "./vectorstore"

class RAGQA:
    def __init__(self, force_reload=False, temperature=0.0, retriever_k=3):
        self.temperature = temperature
        self.retriever_k = retriever_k
        self.embeddings = OpenAIEmbeddings()            # picks up OPENAI_API_KEY from env
        self.vectordb = None
        self.retriever = None
        self.qa = None

        # build or load vectorstore
        if force_reload or not os.path.exists(PERSIST_DIR):
            self._build_vectorstore()
        else:
            self._load_vectorstore()

        # build QA chain
        self._build_chain()

    def _load_vectorstore(self):
        self.vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": self.retriever_k})

    def _build_vectorstore(self):
        documents = []

        # Ingest from URLs
        for url in URLS:
            loader = WebBaseLoader(url)
            documents.extend(loader.load())

        # Ingest from PDFs
        for pdf in PDFS:
            loader = PyPDFLoader(pdf)
            documents.extend(loader.load())

        # Chunk documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs_split = splitter.split_documents(documents)

        # Create & persist vectorstore
        vectordb = Chroma.from_documents(
            docs_split,
            embedding=self.embeddings,
            persist_directory=PERSIST_DIR
        )
        vectordb.persist()

        self.vectordb = vectordb
        self.retriever = vectordb.as_retriever(search_kwargs={"k": self.retriever_k})

    def _build_chain(self):
        llm = ChatOpenAI(temperature=self.temperature, model_name="gpt-4")
        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever
        )

    def update_model_settings(self, temperature=None, retriever_k=None):
        if temperature is not None:
            self.temperature = temperature
        if retriever_k is not None:
            self.retriever_k = retriever_k

        # Update retriever top-k
        if self.vectordb:
            self.retriever = self.vectordb.as_retriever(search_kwargs={"k": self.retriever_k})

        # Rebuild QA chain with new temperature
        self._build_chain()

    def query(self, question: str):
        docs = self.retriever.get_relevant_documents(question)
        if not docs:
            return "No relevant documents found.", []
        result = self.qa.invoke({"query": question})
        answer = result["result"].strip()
        return answer, result.get("source_documents", docs)

def reload_vectorstore():
    try:
        import shutil
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        return True, "Vector store directory cleared. Will reload on next query."
    except Exception as e:
        return False, str(e)
