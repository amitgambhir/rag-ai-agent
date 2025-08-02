# RAG QA module
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_DIR = "data/vector_store"

class RAGQA:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=self.embeddings)
        self.llm = ChatOpenAI(temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

    def query(self, question):
        result = self.qa_chain({"query": question})
        answer = result["result"]
        sources = [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
        return answer, sources

if __name__ == "__main__":
    rag_qa = RAGQA()
    question = "What is artificial intelligence?"
    answer, sources = rag_qa.query(question)
    print(f"Answer:\n{answer}\n")
    print(f"Sources:\n{sources}")
