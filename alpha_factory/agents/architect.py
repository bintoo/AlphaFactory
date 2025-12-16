import logging
from typing import Dict, Optional
from .llm import LLMInterface

logger = logging.getLogger(__name__)

class ArchitectAgent:
    def __init__(self, library, llm: LLMInterface):
        self.library = library
        self.llm = llm

    def produce_strategy_concept(self, state: Dict) -> Dict:
        """
        Produces a strategy concept by transcribing the provided research paper PDF.
        """
        generation = state.get('generation', 0)
        logger.info(f"[ArchitectAgent] Generation {generation}...")
        
        # This block forces the LLM to read the PDF and prevents "Momentum Hallucinations"
        paper_content = state.get('paper_content')
        
        if paper_content:
            logger.info("[Architect] Mode: PAPER ANALYSIS (Forensic)")
            mode = "paper_analysis"

            # Full content analysis mode enabled.
            
            prompt = (
                f"You are a Forensic Quantitative Researcher. I have provided a research paper below.\n"
                f"TASK: Transcribe the 'Algorithm' section into a strict coding plan.\n\n"
                
                f"--- CONTAMINATION CHECK (CRITICAL) ---\n"
                f"1. Does the text explicitly mention 'RSI', 'MACD', or 'Moving Average'?\n"
                f"   - If NO, you are FORBIDDEN from suggesting them.\n"
                f"   - If YES, cite the specific sentence where they appear.\n"
                f"2. Does the text describe a 'Grid' (buying specific levels)?\n"
                f"   - If YES, explicitly state: 'STRATEGY_TYPE: GRID'.\n\n"

                f"--- EXTRACTION INSTRUCTIONS ---\n"
                f"IF GRID STRATEGY:\n"
                f"1. Extract the GRID FORMULA (e.g., P * (1+k)^n).\n"
                f"2. Extract the RESET CONDITION (e.g., 'If price > upper_limit, reset').\n"
                f"3. DEFINE the State Variables needed (e.g., 'upper_limit', 'lower_limit', 'grid_center').\n"
                f"4. Do NOT use MarketOrders. Specify using LimitOrders or checking price levels in memory.\n\n"
                
                f"IF SIGNAL STRATEGY:\n"
                f"1. Extract the exact Entry/Exit triggers.\n\n"

                f"--- OUTPUT FORMAT ---\n"
                f"Provide step-by-step logic for the `OnData` method using ONLY the math found in the paper.\n"
                
                f"--- PAPER CONTENT ---\n{paper_content}"
            )

        else:
            logger.error("[Architect] CRITICAL: No PDF Content provided to Architect.")
            raise ValueError("No PDF Content provided. Architect cannot proceed.")

        # --- GENERATE CONCEPT ---
        # Log full prompt for debugging purposes
        logger.debug(f"[Architect] FULL PROMPT:\n{prompt}")

        if not prompt: # Should safely handle error cases if prompt wasn't defined (though it is)
             return { "strategy_concept": "ERROR", "processing_mode": "error" }

        response = self.llm.generate(prompt)

        if response.strip().upper().startswith("IRRELEVANT"):
            logger.warning(f"[Architect] ❌ PDF REJECTED: {response}")
            return {
                "strategy_concept": response,
                "processing_mode": "irrelevant"
            }
        
        return {
            "strategy_concept": response,
            "processing_mode": mode
        }