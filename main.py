import argparse
import os
import logging
from alpha_factory.graph import AlphaGraphFactory
from alpha_factory import config
from alpha_factory.utils import setup_logging
from pypdf import PdfReader

# Get a logger instance
logger = logging.getLogger(__name__)

def extract_pdf_text(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Failed to extract PDF: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(
        description="AlphaFactory: Autonomous Research Agent using LangGraph. Turns Financial Papers (PDF) into QuantConnect Algorithms."
    )
    parser.add_argument("--paper", required=True, help="Filename of the research paper PDF in alpha_factory/pdf/")
    parser.add_argument("--saveto", help="Optional absolute path to save the generated strategy .py file")
    args = parser.parse_args()

    setup_logging()
    
    # 1. Resolve Paper Path (Support Absolute, Relative, and Tilde paths)
    if os.path.exists(args.paper):
         pdf_path = os.path.abspath(os.path.expanduser(args.paper))
    elif os.path.exists(os.path.expanduser(args.paper)):
         pdf_path = os.path.abspath(os.path.expanduser(args.paper))
    else:
        # Fallback to local 'pdf' folder
        pdf_path = os.path.join(os.path.dirname(__file__), "pdf", args.paper)

    if not os.path.exists(pdf_path):
        logger.error(f"❌ PDF not found at: {pdf_path}")
        return

    logger.info(f"📄 Reading Research Paper: {args.paper}")
    paper_content = extract_pdf_text(pdf_path)

    if not paper_content:
        logger.error("❌ Could not extract text from PDF. Exiting.")
        return
    
    print(f"Starting AlphaFactory Paper Analysis...")
    
    factory = AlphaGraphFactory()
    logger.info(f"Refining Graph...")
    graph = factory.create_graph()
    
    logger.info(f"Starting AlphaFactory Analysis for {args.paper}...")
    
    # Single Execution Run
    initial_state = {
        "generation": 1,
        "iteration_count": 0,
        "syntax_error": None,
        "runtime_error": None,
        "paper_content": paper_content, 
        "paper_name": args.paper,
        "saveto": args.saveto # Pass user desired output path
    }
    
    try:
        # Run the graph
        final_state = graph.invoke(initial_state)
        
        # Check outcome
        if final_state.get("is_saved"):
            logger.info(f"✅ Strategy Generated and Saved to Library.")
        else:
            logger.info(f"❌ Strategy Rejected by Critic/Inspector.")
            if final_state.get("inspector_feedback"):
                logger.error(f"Reason (Inspector): {final_state.get('inspector_feedback')}")
            if final_state.get("critic_feedback"):
                logger.error(f"Reason (Critic): {final_state.get('critic_feedback')}")
            
    except Exception as e:
        logger.exception(f"Graph execution crashed.")

    logger.info("AlphaFactory Run Complete.")

if __name__ == "__main__":
    main()
