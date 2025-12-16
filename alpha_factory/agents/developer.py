import logging
import re
from typing import Dict
from .llm import LLMInterface
from alpha_factory import config
from .lean_api import LEAN_API_CHEATSHEET

logger = logging.getLogger(__name__)

class DeveloperAgent:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def _clean_code(self, raw_text: str) -> str:
        """
        Robustly extracts Python code from LLM markdown output.
        """
        match = re.search(r"```python\n(.*?)```", raw_text, re.DOTALL)
        if match: return match.group(1).strip()
        
        match = re.search(r"```(.*?)```", raw_text, re.DOTALL)
        if match: return match.group(1).strip()
            
        return raw_text.strip()

    def write_code(self, state: Dict) -> str:
        concept_text = state.get('strategy_concept', '')
        
        # Check for feedback
        inspector_err = state.get('inspector_feedback')
        runtime_err = state.get('runtime_error')
        syntax_err = state.get('syntax_error')
        scien_err = state.get('scientific_feedback')
        
        error_msg = inspector_err or runtime_err or syntax_err or scien_err
        
        # --- 1. CONFIGURATION INJECTION ---
        start_dt = config.TEST_START_DATE
        end_dt = config.TEST_END_DATE
        
        config_instruction = (
            f"CRITICAL CONFIGURATION:\n"
            f"1. Set Start Date: self.SetStartDate({start_dt.replace('-', ', ')})\n"
            f"2. Set End Date: self.SetEndDate({end_dt.replace('-', ', ')})\n"
            f"3. Set Cash: self.SetCash(100000)\n"
        )

        if error_msg:
            logger.info(f"[DeveloperAgent] FIXING CODE. Feedback: {error_msg[:100]}...")
            
            advice = ""
            if "Zombie State" in error_msg:
                advice = (
                    "\n*** FIXING ZOMBIE STATE ***\n"
                    "1. Define `def RecoverState(self):`.\n"
                    "2. Loop through `self.Portfolio` to rebuild state.\n"
                    "3. Call `self.RecoverState()` in `Initialize`."
                )
            elif "Passive Recorder" in error_msg:
                advice = "\n*** FIXING PASSIVE RECORDER ***\nCall trading logic inside `OnData`."
            elif "Insufficient buying power" in error_msg:
                advice = "\n*** FIXING MARGIN ERROR ***\nReduce position size calculation. Multiply target by 0.95 buffer."
            elif "Cold Start" in error_msg:
                advice = "\n*** FIXING COLD START ***\nDo NOT return early in OnData during warmup. Allow data collection logic to run."
            elif scien_err:
                advice = (
                    "\n*** SCIENTIFIC REFINEMENT ***\n"
                    "You are REFINING existing code to better match the Research Paper. "
                    "DO NOT REWRITE FROM SCRATCH. Apply the feedback precisely."
                )

            task_prompt = (
                f"The previous code FAILED validation.\n"
                f"ERROR: {error_msg}\n"
                f"{advice}\n"
                f"{config_instruction}\n"
                f"Original Concept: {concept_text}\n"
                f"Current Faulty Code: \n{state.get('python_code', '')}\n\n"
                "TASK: Fix the errors while preserving the strategy logic. Return the FULL corrected script."
            )
        else:
            logger.info(f"[DeveloperAgent] Writing NEW code for concept...")
            task_prompt = (
                f"Implement this strategy concept: {concept_text}\n"
                f"{config_instruction}\n"
                "REQUIREMENT: Use REAL mathematical logic (self.History, built-in indicators). No placeholders."
            )

        # --- PDF MODE ASSUMED ---
        
        # --- 2. UNIVERSE INSTRUCTION ---
        if config.UNIVERSE_MODE == "dynamic":
            universe_instruction = "UNIVERSE: Use 'self.AddUniverse' for dynamic selection."
        else:
            tickers = config.TARGET_UNIVERSE 
            universe_instruction = (
                f"UNIVERSE: Use these Tickers: {tickers}.\n"
                f"Loop through them in Initialize: `equity = self.AddEquity(ticker)`."
            )

        # --- 3. SYSTEM PROMPT WITH GROUNDING ---
        resolution_rule = "2. RESOLUTION: Choose the best resolution (Daily, Hour, Minute) suitable for the strategy logic."

        system_rules = (
            "You are an expert QuantConnect Developer. You DO NOT hallucinate.\n"
            "You ONLY use the API defined in the REFERENCE below.\n\n"
            f"{LEAN_API_CHEATSHEET}\n\n"
            
            "CRITICAL: YOU MUST ADAPT TO THE STRATEGY TYPE.\n"
            "--- TYPE A: MOMENTUM / SIGNAL ---\n"
            "If the concept uses Indicators (RSI, MACD), use standard `SetHoldings` logic.\n\n"
        
            "--- TYPE B: GRID / STATE MACHINE (Use this if 'Grid' is mentioned) ---\n"
            "DO NOT use `SetHoldings`. You must manage specific price levels.\n"
            "CODE PATTERN EXAMPLE (Copy this structure):\n"
            "```python\n"
            "def OnData(self, data):\n"
            "    current_price = data[symbol].Close\n"
            "    # 1. Check for Reset\n"
            "    if current_price > self.upper_limit or current_price < self.lower_limit:\n"
            "        self.ResetGrid(current_price)\n"
            "    # 2. Check Levels\n"
            "    if current_price < self.next_buy_level:\n"
            "        self.LimitOrder(symbol, 1, self.next_buy_level)\n"
            "```\n"
            "RULE: If the concept is 'Dynamic Grid', YOU MUST COPY THE PATTERN ABOVE. Do NOT import RSI.\n\n"

            "RULES:\n"
            f"1. CLASS NAME: `{config.TARGET_CLASS_NAME}` inheriting `QCAlgorithm`.\n"
            f"{resolution_rule}\n"
            "3. STATE RECOVERY: Always define `def RecoverState(self)` and call it in `Initialize`.\n"
            "4. ACTIVE GUARD: Always call your exit logic inside `OnData`.\n"
            "5. NO STRINGS: Use `self.symbols` objects in logic.\n"
            "6. WARMUP: Use `self.SetWarmUp`.\n"
            "7. CLEAN CODE: No markdown blocks in output.\n"
            "CRITICAL SAFETY RULES:\n"
            "8. DATAFRAME SAFETY: `self.History` returns a MultiIndex (Symbol, Time). "
            "You MUST use `.loc[symbol]` to isolate the DatetimeIndex before iterating dates. "
            "NEVER iterate `history.index` directly.\n"
            "9. LEVERAGE SAFETY: If your logic uses >1x leverage (e.g. 4x), you MUST "
            "explicitly call `equity.SetLeverage(4)` inside the Initialize loop.\n"
            "10. MARGIN BUFFER: When sizing positions based on leverage, NEVER target 100%. "
            "Multiply your max exposure calculation by 0.95 (e.g., `0.95 * 4.0 * PortfolioValue`). "
            "This prevents 'Insufficient buying power' errors due to small price fluctuations.\n"
            "11. WARMUP DATA: Do NOT return early in `OnData` if you are manually building history (e.g. Opening Range Volume). "
            "Allow data collection to run during warmup. Only wrap *ordering* logic in `if not self.IsWarmingUp:`.\n"
            "13. EXECUTION: For Breakout or Stop-Loss triggers, use `self.MarketOrder`. "
            "Do NOT use `LimitOrder` at the trigger price, as it may not fill if the market gaps. "
            "Use `LimitOrder` ONLY for Grid strategies or profit targets.\n"
            "14. PYTHON DATETIME: Use `.year`, `.month`, `.day` (LOWERCASE). "
            "Do NOT use C# style `.Year`, `.Month`."
        )

        prompt = f"{system_rules}\n{universe_instruction}\nTASK: {task_prompt}\n"

        raw_output = self.llm.generate(prompt)
        return self._clean_code(raw_output)