import subprocess
import os
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate

PERSIST_DIRECTORY = "vectorstore"

class RAGQA:
    def __init__(self):
        # Initialize the vector store using persisted data
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )

        # Strict prompt: Don't answer if context is insufficient
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an intelligent assistant helping users strictly based on the provided context below.

If the context does not contain enough information to answer the question, clearly respond:
"I’m sorry, but I can’t find enough information in the available documents to answer that."

Do not use any external or prior knowledge.

Context:
{context}

Question:
{question}

Answer:
""",
        )

        # Create the retrieval-based QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": 10}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template},
        )

    def query(self, question):
        result = self.qa_chain(question)
        answer = result["result"]
        sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
        return answer, sources

def reload_vectorstore():
    """Re-run the ingestion pipeline to refresh the vector DB and reload in memory."""
    try:
        result = subprocess.run(
            ["python", "modules/rag_ingest.py"],
            capture_output=True,
            text=True,
            check=True
        )
        print("Vector DB reloaded successfully:\n", result.stdout)

        # Count number of documents reloaded
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )
        doc_count = vectordb._collection.count()

        return True, f"{result.stdout}\n\nTotal documents loaded: {doc_count}"
    except subprocess.CalledProcessError as e:
        print("Error reloading vector DB:\n", e.stderr)
        return False, e.stderr
