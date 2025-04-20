from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import faiss

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '.env.backend'))

class Retriever:
    def __init__(self, index_dir: str):
        """Initialize the retriever with vector store."""
        self.index_dir = index_dir
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        self.vector_store = self._load_or_create_vector_store()

    def _load_or_create_vector_store(self) -> FAISS:
        """Load existing vector store or create a new one."""
        try:
            os.makedirs(self.index_dir, exist_ok=True)
            index_path = os.path.join(self.index_dir, "index.faiss")
            store_path = os.path.join(self.index_dir, "store.pkl")
            
            if os.path.exists(index_path) and os.path.exists(store_path):
                return FAISS.load_local(
                    self.index_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                # Create a new vector store with a sample document
                texts = ["Initial document for vector store setup"]
                vector_store = FAISS.from_texts(texts, self.embeddings)
                vector_store.save_local(self.index_dir)
                return vector_store
                
        except Exception as e:
            print(f"Error in vector store operation: {str(e)}")
            # Create a new vector store as fallback
            texts = ["Initial document for vector store setup"]
            vector_store = FAISS.from_texts(texts, self.embeddings)
            vector_store.save_local(self.index_dir)
            return vector_store

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store."""
        try:
            docs = [
                Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"]
                )
                for doc in documents
            ]
            
            self.vector_store.add_documents(docs)
            self.vector_store.save_local(self.index_dir)
        except Exception as e:
            raise Exception(f"Error adding documents: {str(e)}")

    def retrieve_relevant_documents(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """Retrieve relevant documents based on similarity search."""
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            raise Exception(f"Error retrieving documents: {str(e)}")

    def save_vector_store(self):
        """Save the current state of the vector store."""
        try:
            self.vector_store.save_local(self.index_dir)
        except Exception as e:
            raise Exception(f"Error saving vector store: {str(e)}") 