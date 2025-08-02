from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

PERSIST_DIRECTORY = "vectorstore"

def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

def create_qa_chain():
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an intelligent assistant helping answer questions based on provided context.

Context: {context}

Question: {question}

Answer concisely and accurately. If the context does not contain the answer, say "I don't know".
        """
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-4")  # You can change this to gpt-3.5-turbo if needed
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain

def answer_question(question: str):
    qa_chain = create_qa_chain()
    return qa_chain.run(question)

if __name__ == "__main__":
    while True:
        question = input("Ask a question (or 'exit' to quit): ")
        if question.lower() == "exit":
            break
        answer = answer_question(question)
        print("Answer:", answer)
