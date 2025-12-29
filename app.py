"""
Gradio web interface for the Document Q&A chatbot.
"""
import gradio as gr
from src.vectorstore import VectorStoreManager
from src.retrieval_chain import RAGChain
from src.config import DOCUMENTS_DIR


# Initialize RAG system
print("Initializing RAG system...")
try:
    vs_manager = VectorStoreManager()
    vs_manager.load_vectorstore()
    rag_chain = RAGChain(vs_manager)
    print("✓ RAG system ready!\n")
except Exception as e:
    print(f"❌ Error loading system: {e}")
    print(f"Make sure you've run: python ingest.py")
    exit(1)


def answer_question(question: str, show_sources: bool = True) -> tuple:
    """
    Process user question and return answer with sources.
    
    Args:
        question: User's question
        show_sources: Whether to display source documents
        
    Returns:
        Tuple of (answer, sources)
    """
    if not question.strip():
        return "Please enter a question.", ""
    
    try:
        # Get answer
        result = rag_chain.query(question)
        answer = result["answer"]
        
        # Format sources
        if show_sources:
            sources = rag_chain.format_sources(result["context"])
        else:
            sources = "Sources hidden (enable 'Show Sources' to view)"
        
        return answer, sources
    
    except Exception as e:
        return f"Error: {str(e)}", ""


# Build Gradio Interface
with gr.Blocks(title="Document Q&A Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📚 Document Q&A Assistant with RAG
        
        Ask questions about your documents and get AI-powered answers with source citations.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What are the key findings in the document?",
                lines=2
            )
            
            show_sources_checkbox = gr.Checkbox(
                label="Show source documents",
                value=True
            )
            
            submit_btn = gr.Button("Get Answer", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown(
                f"""
                ### ℹ️ System Info
                - **Documents**: `{DOCUMENTS_DIR}`
                - **Model**: Llama 3.2
                - **Vector DB**: ChromaDB
                
                **Tips:**
                - Ask specific questions
                - Questions should relate to document content
                - Check sources for verification
                """
            )
    
    answer_output = gr.Textbox(
        label="Answer",
        lines=8,
        interactive=False
    )
    
    sources_output = gr.Textbox(
        label="Source Documents",
        lines=6,
        interactive=False
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["What is this document about?"],
            ["Summarize the main points"],
            ["What are the key findings?"],
            ["Can you explain the methodology used?"],
        ],
        inputs=question_input
    )
    
    # Event handlers
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, show_sources_checkbox],
        outputs=[answer_output, sources_output]
    )
    
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, show_sources_checkbox],
        outputs=[answer_output, sources_output]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Makes it accessible on your network
        server_port=7860,
        share=False  # Set to True to get a public URL
    )
