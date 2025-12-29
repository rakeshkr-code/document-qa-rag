"""
Script to process and index documents into the vector database.
Run this once to set up your knowledge base.
"""
import sys
from src.document_loader import DocumentProcessor
from src.vectorstore import VectorStoreManager
from src.config import DOCUMENTS_DIR


def main():
    """Main ingestion pipeline."""
    print("\n" + "=" * 60)
    print(" DOCUMENT INGESTION PIPELINE ".center(60, "="))
    print("=" * 60 + "\n")
    
    try:
        # Step 1: Process documents
        processor = DocumentProcessor()
        chunks = processor.process_documents()
        
        if not chunks:
            print("❌ No documents found. Please add PDF files to:", DOCUMENTS_DIR)
            sys.exit(1)
        
        # Step 2: Create vector store
        vs_manager = VectorStoreManager()
        vs_manager.create_vectorstore(chunks, persist=True)
        
        print("\n" + "=" * 60)
        print(" ✓ INGESTION COMPLETE ".center(60, "="))
        print("=" * 60)
        print(f"\nIndexed {len(chunks)} document chunks")
        print(f"Vector store saved to: {vs_manager.vectorstore._persist_directory}")
        print("\nYou can now run: python app.py")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
