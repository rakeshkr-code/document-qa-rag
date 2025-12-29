"""
Handles PDF loading, text extraction, and chunking.
"""
from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """Processes documents for RAG pipeline."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Split on paragraphs first
        )
    
    def load_single_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Document objects with page content and metadata
        """
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {Path(pdf_path).name}")
        return documents
    
    def load_all_pdfs(self, directory: str = None) -> List[Document]:
        """
        Load all PDF files from directory.
        
        Args:
            directory: Path to directory containing PDFs (default: data/documents/)
            
        Returns:
            List of all documents
        """
        if directory is None:
            directory = str(DOCUMENTS_DIR)
        
        loader = DirectoryLoader(
            directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        print(f"\nLoaded {len(documents)} total pages from {directory}")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked documents with preserved metadata
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Add chunk IDs to metadata
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
            
        return chunks
    
    def process_documents(self, directory: str = None) -> List[Document]:
        """
        Complete pipeline: load PDFs and chunk them.
        
        Args:
            directory: Path to directory containing PDFs
            
        Returns:
            List of processed document chunks
        """
        print("=" * 50)
        print("DOCUMENT PROCESSING PIPELINE")
        print("=" * 50)
        
        # Load documents
        documents = self.load_all_pdfs(directory)
        
        if not documents:
            raise ValueError(f"No PDF files found in {directory or DOCUMENTS_DIR}")
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Display sample
        print("\n" + "=" * 50)
        print("SAMPLE CHUNK:")
        print("=" * 50)
        print(f"Content: {chunks[0].page_content[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")
        print("=" * 50 + "\n")
        
        return chunks


# Standalone testing
if __name__ == "__main__":
    processor = DocumentProcessor()
    chunks = processor.process_documents()
    print(f"✓ Successfully processed {len(chunks)} chunks")
