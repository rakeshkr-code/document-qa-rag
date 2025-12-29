"""
Configuration settings for the RAG system.
Centralized location for all hyperparameters and paths.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# Create directories if they don't exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# Model Configuration
OLLAMA_MODEL = "llama3.2"  # Options: llama3.2, llama3.2:1b, mistral
OLLAMA_BASE_URL = "http://localhost:11434"

# Embedding Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, 384 dimensions
# Alternative: "sentence-transformers/all-mpnet-base-v2" (768d, slower but better)

# Text Splitting Configuration
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks (prevents context loss)

# Retrieval Configuration
TOP_K_RESULTS = 4  # Number of relevant chunks to retrieve
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score (0-1)

# ChromaDB Configuration
COLLECTION_NAME = "document_qa_collection"

# LLM Generation Parameters
TEMPERATURE = 0.2  # Lower = more focused, Higher = more creative (0-1)
MAX_TOKENS = 512  # Maximum response length

# Prompt Template
PROMPT_TEMPLATE = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.

If you don't know the answer based on the context, just say "I don't have enough information in the provided documents to answer this question." Don't try to make up an answer.

Context:
{context}

Question: {question}

Helpful Answer:"""
