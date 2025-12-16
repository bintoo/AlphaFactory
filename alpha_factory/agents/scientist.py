import logging
from typing import Dict, Optional
from .llm import LLMInterface

logger = logging.getLogger(__name__)

class ScientistAgent:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def review_implementation(self, code: str, paper_content: str) -> Dict[str, str]:
        """
        Reviews the code against the paper.
        Returns: {'decision': 'APPROVE'|'REFINE', 'feedback': '...'}
        """
        logger.info("[Scientist] Starting Peer Review (Paper vs Code)...")
        
        prompt = (
            f"You are a Senior Quantitative Scientist acting as a Strict Peer Reviewer.\n"
            f"OBJECTIVE: Verify if the Python Code implementation matches the Research Paper logic.\n\n"
            
            f"--- RESEARCH PAPER ---\n"
            f"{paper_content}\n\n"
            
            f"--- IMPLEMENTED CODE ---\n"
            f"{code}\n\n"
            
            f"--- REVIEW CRITERIA ---\n"
            f"1. Did the developer implement the SPECIFIC formula from the paper? (Check constants, windows, logic).\n"
            f"2. Are there missing components? (e.g. Paper asked for Stop Loss, Code has none).\n"
            f"3. Note: Syntax is ALREADY verified. Focus ONLY on **Scientific Accuracy**.\n\n"
            
            f"--- OUTPUT FORMAT ---\n"
            f"If the code is accurate enough (>=90% match), output ONLY: 'APPROVE'\n"
            f"If changes are needed, output: 'REFINE: [Detailed instructions for the developer]'\n"
        )
        
        response = self.llm.generate(prompt).strip()
        
        if response.startswith("APPROVE"):
            logger.info("[Scientist] ✅ Code Approved. Faithful to Paper.")
            return {"decision": "APPROVE", "feedback": "Code is scientifically accurate."}
        elif response.startswith("REFINE:"):
            feedback = response.replace("REFINE:", "").strip()
            # Downgraded to debug to reduce verbosity
            logger.debug(f"[Scientist] ⚠️ Refinement Requested: {feedback}")
            # Keep a smaller log for user info
            logger.info(f"[Scientist] ⚠️ Refinement Requested (See debug log for details).")
            return {"decision": "REFINE", "feedback": feedback}
        else:
            # Fallback for unclear response
            logger.warning(f"[Scientist] Unclear response, defaulting to Approve: {response}")
            return {"decision": "APPROVE", "feedback": "Auto-approved (Unclear LLM response)"}
