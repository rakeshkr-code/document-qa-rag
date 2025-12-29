"""
Constructs the RAG chain that combines retrieval and generation.
"""
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    TEMPERATURE,
    PROMPT_TEMPLATE
)
from src.vectorstore import VectorStoreManager


class RAGChain:
    """Manages the complete RAG pipeline."""
    
    def __init__(self, vectorstore_manager: VectorStoreManager):
        """
        Initialize RAG chain with vector store.
        
        Args:
            vectorstore_manager: Initialized VectorStoreManager instance
        """
        self.vectorstore_manager = vectorstore_manager
        
        # Initialize LLM
        print(f"Connecting to Ollama model: {OLLAMA_MODEL}...")
        self.llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE
        )
        print("✓ LLM connected")
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        
        # Build the chain
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """
        Build the retrieval-generation chain.
        
        Returns:
            Complete RAG chain
        """
        # Get retriever
        retriever = self.vectorstore_manager.get_retriever()
        
        # Create document chain (combines docs with LLM)
        document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        
        # Create retrieval chain (retrieves docs + generates answer)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
    
    def query(self, question: str) -> dict:
        """
        Ask a question and get an answer with sources.
        
        Args:
            question: User's question
            
        Returns:
            Dict with 'answer' and 'context' (source documents)
        """
        response = self.chain.invoke({"input": question})
        
        return {
            "answer": response["answer"],
            "context": response["context"],
            "question": question
        }
    
    def format_sources(self, context: list) -> str:
        """
        Format source documents for display.
        
        Args:
            context: List of retrieved documents
            
        Returns:
            Formatted string with sources
        """
        if not context:
            return "No sources found."
        
        sources = []
        for i, doc in enumerate(context, 1):
            source_file = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            content_preview = doc.page_content[:150]
            
            sources.append(
                f"[{i}] File: {source_file} | Page: {page}\n"
                f"    Content: {content_preview}...\n"
            )
        
        return "\n".join(sources)


# Standalone testing
if __name__ == "__main__":
    # Load vector store
    vs_manager = VectorStoreManager()
    vs_manager.load_vectorstore()
    
    # Create RAG chain
    rag = RAGChain(vs_manager)
    
    # Test query
    question = "What is the main topic of the document?"
    result = rag.query(question)
    
    print("\n" + "=" * 50)
    print("QUESTION:", question)
    print("=" * 50)
    print("\nANSWER:", result["answer"])
    print("\n" + "=" * 50)
    print("SOURCES:")
    print("=" * 50)
    print(rag.format_sources(result["context"]))
