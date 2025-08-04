
import os
import logging
from typing import Dict, List, Any
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGWithMemory:
    """
    RAG (Retrieval-Augmented Generation) system with conversation memory.
    
    This class combines document retrieval with OpenAI's language model
    and maintains conversation history for context-aware responses.
    """
    
    def __init__(self, vectorstore_path="./vectorstore", memory_length=5):
        """
        Initialize the RAG system with memory.
        
        Args:
            vectorstore_path (str): Path to the Chroma vector store
            memory_length (int): Number of previous conversations to remember
        """
        self.vectorstore_path = vectorstore_path
        self.memory_length = memory_length
        
        # Initialize conversation memory
        self.memory = ConversationBufferWindowMemory(
            k=memory_length,  # Remember last 5 exchanges by default
            return_messages=True,
            memory_key="chat_history",
            input_key="question",
            output_key="answer"
        )
        
        # Initialize the QA chain
        self.qa_chain = None
        self._setup_qa_chain()
        
        logger.info(f"RAG system initialized with memory length: {memory_length}")
    
    def _setup_qa_chain(self):
        """
        Set up the conversational retrieval chain with memory.
        This is called during initialization.
        """
        try:
            # Check if vector store exists
            if not os.path.exists(self.vectorstore_path):
                logger.error(f"Vector store not found at {self.vectorstore_path}")
                raise FileNotFoundError(f"Vector store not found. Please run document ingestion first.")
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings()
            
            # Load the vector store
            vectorstore = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=embeddings
            )
            
            # Check if vector store has any documents
            collection = vectorstore._collection
            doc_count = collection.count()
            if doc_count == 0:
                logger.warning("Vector store is empty. Add some documents first.")
            else:
                logger.info(f"Vector store loaded with {doc_count} documents")
            
            # Initialize the language model
            llm = ChatOpenAI(
                temperature=0.7,
                model_name="gpt-3.5-turbo",
                max_tokens=1000
            )
            
            # Create a custom prompt template that uses memory
            prompt_template = """Use the following pieces of context and the conversation history to answer the question at the end.

Conversation History:
{chat_history}

Context from documents:
{context}

Current Question: {question}

Instructions:
1. Use the conversation history to understand the context of the current question
2. Base your answer primarily on the provided document context
3. If the document context doesn't contain enough information, say so clearly
4. Reference previous parts of our conversation when relevant
5. Be helpful and conversational

Answer:"""

            # Note: ConversationalRetrievalChain handles the prompt internally
            # We'll use the default prompt for now, but you can customize it later
            
            # Create the conversational retrieval chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}  # Retrieve top 5 relevant documents
                ),
                memory=self.memory,
                return_source_documents=True,  # This gives us the source documents
                verbose=True  # Set to False in production
            )
            
            logger.info("QA chain setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup QA chain: {str(e)}")
            raise
    
    def get_answer(self, question: str) -> Dict[str, Any]:
        """
        Get an answer to a question using RAG with conversation memory.
        
        Args:
            question (str): The user's question
            
        Returns:
            Dict containing:
                - answer: The AI's response
                - source_documents: List of relevant documents used
                - chat_history: Current conversation history
                - success: Boolean indicating if the operation was successful
        """
        try:
            # Validate input
            if not question or not question.strip():
                return {
                    "answer": "Please ask a valid question.",
                    "source_documents": [],
                    "chat_history": [],
                    "success": False
                }
            
            # Clean and sanitize the input
            clean_question = self._sanitize_input(question)
            
            logger.info(f"Processing question: {clean_question[:100]}...")
            
            # Check if QA chain is initialized
            if not self.qa_chain:
                logger.error("QA chain not initialized")
                return {
                    "answer": "System error: QA chain not initialized. Please contact support.",
                    "source_documents": [],
                    "chat_history": [],
                    "success": False
                }
            
            # Get response from the chain
            result = self.qa_chain({
                "question": clean_question
            })
            
            # Extract source documents information
            source_docs = []
            for doc in result.get("source_documents", []):
                source_docs.append({
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown")
                })
            
            # Get current chat history
            chat_history = []
            for message in self.memory.chat_memory.messages:
                chat_history.append({
                    "type": message.__class__.__name__,
                    "content": message.content
                })
            
            logger.info(f"Successfully generated answer using {len(source_docs)} source documents")
            
            return {
                "answer": result["answer"],
                "source_documents": source_docs,
                "chat_history": chat_history,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try again or contact support if the problem persists.",
                "source_documents": [],
                "chat_history": [],
                "success": False
            }
    
    def _sanitize_input(self, text: str) -> str:
        """
        Clean and sanitize user input to prevent issues.
        
        Args:
            text (str): Raw user input
            
        Returns:
            str: Cleaned and sanitized text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove potential script injections (basic protection)
        text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        # Limit length to prevent very long inputs
        if len(text) > 1000:
            text = text[:1000]
            logger.warning("Input truncated to 1000 characters")
        
        return text
    
    def clear_memory(self):
        """
        Clear the conversation memory.
        This will start a fresh conversation.
        """
        try:
            self.memory.clear()
            logger.info("Conversation memory cleared")
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current memory state.
        
        Returns:
            Dict with memory information
        """
        try:
            messages = self.memory.chat_memory.messages
            return {
                "total_messages": len(messages),
                "memory_length_setting": self.memory_length,
                "has_conversation": len(messages) > 0
            }
        except Exception as e:
            logger.error(f"Error getting memory summary: {str(e)}")
            return {
                "total_messages": 0,
                "memory_length_setting": self.memory_length,
                "has_conversation": False
            }

# Backward compatibility: Create an alias for the old class name if you had one
QASystem = RAGWithMemory

# Example usage (you can remove this in production)
if __name__ == "__main__":
    # This will only run if you execute this file directly
    try:
        # Initialize the system
        rag_system = RAGWithMemory()
        
        # Test question
        response = rag_system.get_answer("What documents do you have information about?")
        print(f"Answer: {response['answer']}")
        print(f"Used {len(response['source_documents'])} sources")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure you have:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Run document ingestion to create the vector store")
        print("3. Installed all required dependencies")