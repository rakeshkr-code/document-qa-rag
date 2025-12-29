"""
Utility functions for the RAG system.
"""
import os
from pathlib import Path
from typing import List


def count_pdf_files(directory: str) -> int:
    """Count PDF files in directory."""
    path = Path(directory)
    return len(list(path.glob("**/*.pdf")))


def get_pdf_files(directory: str) -> List[str]:
    """Get list of all PDF file paths."""
    path = Path(directory)
    return [str(p) for p in path.glob("**/*.pdf")]


def check_ollama_running() -> bool:
    """Check if Ollama server is running."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def validate_environment():
    """Validate that all required components are available."""
    from src.config import DOCUMENTS_DIR, VECTORSTORE_DIR, OLLAMA_BASE_URL
    
    issues = []
    
    # Check directories
    if not DOCUMENTS_DIR.exists():
        issues.append(f"Documents directory missing: {DOCUMENTS_DIR}")
    
    pdf_count = count_pdf_files(DOCUMENTS_DIR)
    if pdf_count == 0:
        issues.append(f"No PDF files found in {DOCUMENTS_DIR}")
    
    # Check Ollama
    if not check_ollama_running():
        issues.append(f"Ollama not running. Start it with: ollama serve")
    
    return issues


if __name__ == "__main__":
    issues = validate_environment()
    if issues:
        print("❌ Environment Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Environment validated successfully")
