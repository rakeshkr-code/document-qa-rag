"""
Manages vector database operations: creation, persistence, and retrieval.
"""
from typing import List, Optional
from langchain.schema import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import (
    VECTORSTORE_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K_RESULTS
)


class VectorStoreManager:
    """Handles all vector database operations."""
    
    def __init__(self):
        """Initialize embedding model and vector store."""
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}  # Improves retrieval
        )
        print(f"✓ Embedding model loaded: {EMBEDDING_MODEL}")
        
        self.vectorstore = None
    
    def create_vectorstore(
        self,
        documents: List[Document],
        persist: bool = True
    ) -> Chroma:
        """
        Create new vector store from documents.
        
        Args:
            documents: List of document chunks
            persist: Whether to save to disk
            
        Returns:
            Chroma vector store instance
        """
        print(f"\nCreating vector store with {len(documents)} documents...")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTORSTORE_DIR) if persist else None
        )
        
        print(f"✓ Vector store created and saved to {VECTORSTORE_DIR}")
        return self.vectorstore
    
    def load_vectorstore(self) -> Chroma:
        """
        Load existing vector store from disk.
        
        Returns:
            Chroma vector store instance
        """
        print("Loading existing vector store...")
        
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )
        
        print(f"✓ Vector store loaded from {VECTORSTORE_DIR}")
        return self.vectorstore
    
    def get_retriever(self, k: int = TOP_K_RESULTS):
        """
        Create a retriever from the vector store.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever object for RAG chain
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call load_vectorstore() first.")
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        return retriever
    
    def similarity_search(self, query: str, k: int = TOP_K_RESULTS) -> List[Document]:
        """
        Perform similarity search (for testing).
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of most similar documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def delete_collection(self):
        """Delete the entire collection (use with caution)."""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            print(f"✓ Collection '{COLLECTION_NAME}' deleted")


# Standalone testing
if __name__ == "__main__":
    from src.document_loader import DocumentProcessor
    
    # Process documents
    processor = DocumentProcessor()
    chunks = processor.process_documents()
    
    # Create vector store
    vs_manager = VectorStoreManager()
    vectorstore = vs_manager.create_vectorstore(chunks)
    
    # Test search
    query = "What is this document about?"
    results = vs_manager.similarity_search(query, k=2)
    
    print(f"\nTest Query: '{query}'")
    print("\nTop Results:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.page_content[:150]}...")
        print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
