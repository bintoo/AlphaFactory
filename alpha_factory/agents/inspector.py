import ast
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class InspectorAgent:
    def __init__(self, library, llm):
        self.library = library
        self.llm = llm

    def validate(self, code: str, allow_any_resolution: bool = False) -> Optional[str]:
        logger.info("[Inspector] Starting Ironclad Validation...")

        # --- PHASE 1: SYNTAX (AST) ---
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax Error line {e.lineno}: {e.msg}"

        # --- PHASE 2: ARCHITECTURE & SAFETY (Deterministic) ---
        
        # 1. Resolution Check
        if not allow_any_resolution:
            if "Resolution.Minute" not in code:
                return "Architecture Error: You MUST use `Resolution.Minute` for the Security Subscription."

        # 2. DATA SAFETY CHECK
        error = self._check_data_safety(code)
        if error: return f"Data Safety Error: {error}"

        # 3. Liveness Check
        if "SetHoldings" not in code and "MarketOrder" not in code and "LimitOrder" not in code:
            return "Liveness Error: Code never places a trade (no SetHoldings/MarketOrder/LimitOrder found)."

        # 4. Active Guard Check
        error = self._check_active_guard(tree)
        if error: return f"Architecture Error: {error}"

        # 5. Zombie State Check
        error = self._check_zombie_state(tree, code)
        if error: return f"Safety Error: {error}"

        # 6. Missing Method Definition Check
        error = self._check_undefined_methods(tree)
        if error: return f"Syntax Error: {error}"

        # 7. Variable Consistency Check
        error = self._check_variable_consistency(tree)
        if error: return f"Logic Error: {error}"

        # 8. Cold Start / Warmup Check
        error = self._check_cold_start(tree)
        if error: return f"Logic Error (Cold Start): {error}"

        # 9. Python Syntax Check (New)
        error = self._check_python_syntax_violations(tree)
        if error: return f"Syntax Error (Python): {error}"

        # --- PHASE 3: DUPLICATES ---
        if self.library and self.library.is_duplicate(code):
            return "Duplicate Strategy Detected."

        # --- PHASE 4: LOGIC REVIEW (LLM) ---
        return self._run_llm_review(code)

    def _check_cold_start(self, tree: ast.AST) -> Optional[str]:
        """
        Prevents the strategy from returning early in OnData during Warmup,
        which blocks data collection (e.g. for Opening Range Volume).
        """
        on_data = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'OnData'), None)
        if not on_data: return None

        # Check the first few nodes of OnData
        if len(on_data.body) > 0:
            first_node = on_data.body[0]
            
            # Check for: if self.IsWarmingUp: return
            if isinstance(first_node, ast.If):
                # Check condition: self.IsWarmingUp
                test = first_node.test
                is_warming_check = False
                if isinstance(test, ast.Attribute) and test.attr == 'IsWarmingUp':
                    is_warming_check = True
                
                # Check body: return
                has_return = any(isinstance(n, ast.Return) for n in first_node.body)

                if is_warming_check and has_return:
                    return (
                        "Do not use `if self.IsWarmingUp: return` at the start of OnData. "
                        "This blocks data collection (history building). "
                        "Instead, wrap only the *ordering* logic in `if not self.IsWarmingUp:`."
                    )
        return None

    def _check_data_safety(self, code: str) -> Optional[str]:
        banned_terms = ["BidPrice", "AskPrice", "BidSize", "AskSize", "QuoteBar"]
        for term in banned_terms:
            if term in code:
                return (
                    f"Forbidden Term Detected: `{term}`. "
                    "Your local environment DOES NOT have Quote Data. "
                    "You must use Trade Data (Close, High, Low) only."
                )
        return None

    def _check_undefined_methods(self, tree: ast.AST) -> Optional[str]:
        defined_methods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_methods.add(node.name)

        qc_methods = {
            'SMA', 'EMA', 'RSI', 'ATR', 'MACD', 'BB', 'KCH', 'AROON', 'ADX', 'MOM', 'MAX', 'MIN', 'STD',
            'SetHoldings', 'Liquidate', 'MarketOrder', 'LimitOrder', 'StopMarketOrder', 'StopLimitOrder',
            'Buy', 'Sell', 'SetCash', 'SetStartDate', 'SetEndDate', 'SetWarmUp', 'SetBrokerageModel',
            'AddEquity', 'AddForex', 'AddCrypto', 'AddFuture', 'AddOption', 'AddUniverse', 'AddData',
            'Schedule', 'RegisterIndicator', 'ResolveConsolidator', 'Train', 'SetBenchmark', 'SetTimeZone',
            'Debug', 'Log', 'Error', 'Plot', 'Quit', 'Notify',
            'History', 'GetParameter', 'Securities', 'Portfolio',
            'SetFillModel', 'ImmediateFillModel', 'SetLeverage'
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
                        method_name = node.func.attr
                        if method_name not in defined_methods and method_name not in qc_methods:
                            return (
                                f"Missing Method Definition: You called `self.{method_name}()` "
                                f"but never defined `def {method_name}(self):` in your class."
                            )
        return None

    def _check_active_guard(self, tree: ast.AST) -> Optional[str]:
        on_data = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'OnData'), None)
        if not on_data:
            return "Missing `OnData` method. Minute strategies must check exits in `OnData`."

        has_action = False
        passive_verbs = {'Add', 'Update', 'Plot', 'Log', 'Debug', 'Error', 'Print', 'IsReady', 'WarmUp'}
        
        for node in ast.walk(on_data):
            if isinstance(node, ast.Attribute) and node.attr in ['Invested', 'Portfolio', 'Holdings', 'Quantity']:
                has_action = True
                break

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    method_name = node.func.attr
                    is_self_call = isinstance(node.func.value, ast.Name) and node.func.value.id == 'self'
                    if is_self_call and method_name not in passive_verbs:
                        has_action = True
                        break
                    if method_name in ['Liquidate', 'SetHoldings', 'MarketOrder']:
                        has_action = True
                        break

        if not has_action:
            return "Passive Recorder Detected: `OnData` must call active logic (e.g. `self.ManagePosition()`)."
        return None

    def _check_zombie_state(self, tree: ast.AST, code: str) -> Optional[str]:
            init_node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'Initialize'), None)
            if not init_node: return None

            state_vars = set()
            for node in ast.walk(init_node):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                            if isinstance(node.value, ast.Dict) and target.attr not in ['indicators', 'symbols', 'securities', 'data', 'settings', 'charts']:
                                state_vars.add(target.attr)

            if not state_vars: return None

            methods = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            called_methods = set()
            for node in ast.walk(init_node):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute) and node.func.attr in methods:
                        called_methods.add(node.func.attr)

            has_recovery_logic = False
            for method_name in called_methods:
                method_node = methods[method_name]
                for node in ast.walk(method_node):
                    if isinstance(node, ast.Attribute):
                        if node.attr in ['Portfolio', 'History', 'Invested', 'Quantity', 'Holdings']:
                            has_recovery_logic = True
                            break
                if has_recovery_logic:
                    break

            if not has_recovery_logic:
                for node in ast.walk(init_node):
                    if isinstance(node, ast.For):
                        seg = ast.get_source_segment(code, node)
                        if seg and ("Invested" in seg or "Quantity" in seg or "Holdings" in seg):
                            has_recovery_logic = True
                            break

            if not has_recovery_logic:
                return (
                    f"Zombie State Risk: You defined state variables {state_vars} in `Initialize` but found no recovery logic. "
                    "You must either:\n"
                    "1. Call a method in `Initialize` (e.g. `self.RecoverState()`) that accesses `self.Portfolio`.\n"
                    "2. Iterate over `self.Portfolio` directly in `Initialize` to rebuild state."
                )
                
    def _check_python_syntax_violations(self, tree: ast.AST) -> Optional[str]:
        """
        Detects common Python pitfalls, particularly C#-style DateTime usage.
        Exempts known QuantConnect Enums like Resolution.Minute.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr in ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']:
                    # CHECK PARENT: If it's `Resolution.Minute`, it is VALID.
                    if isinstance(node.value, ast.Name) and node.value.id in ['Resolution', 'TimeRules', 'DateRules']:
                        continue
                        
                    # Otherwise, flag it as likely C# -> Python translation error
                    return (
                        f"Syntax Error: Detected C#-style DateTime property `.{node.attr}`. "
                        f"Python uses lowercase (`.year`, `.month`, `.day`). "
                        f"Please fix `{node.attr}` to `{node.attr.lower()}`."
                    )
        
        # New Check: Dataframe Iteration Safety
        error = self._check_dataframe_iteration_safety(tree)
        if error: return error
        
        return None

    def _check_dataframe_iteration_safety(self, tree: ast.AST) -> Optional[str]:
        """
        Detects unsafe iteration of self.History (MultiIndex) which returns tuples,
        causing 'tuple object has no attribute .year' errors.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ['year', 'month', 'day']:
                # Found .year usage. Check if variable name suggests it might be a tuple.
                # Heuristic: LLMs often iterate 'row' or 'tuple' or 't' or 'bar'.
                # But rigorously, we want to warn if this happens inside a History loop.
                pass
                
        # Better heuristic: Check for `for x in history:` or `for x in history.index:`
        # without explicit unstacking or loc.
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Target variable name
                target_id = None
                if isinstance(node.target, ast.Name): target_id = node.target.id
                elif isinstance(node.target, ast.Tuple): target_id = "tuple_unpack check"

                # Iterated object
                iter_obj = node.iter
                
                # Case 1: Iterating history.index directly
                if isinstance(iter_obj, ast.Attribute) and iter_obj.attr == 'index':
                     # We can't know for sure if it's a dataframe, but if usage of target inside body is `.year`, flag it.
                     if target_id:
                         for subnode in ast.walk(node):
                             if isinstance(subnode, ast.Attribute) and subnode.attr in ['year', 'month', 'day']:
                                 if isinstance(subnode.value, ast.Name) and subnode.value.id == target_id:
                                     return (
                                         "Pandas Safety Error: You are accessing `.year/.month` on an iterated index. "
                                         "If this index comes from `self.History` (MultiIndex), it yields TUPLES `(symbol, time)`, not safe DateTimes. "
                                         "FIX: Use `.loc[symbol]` to isolate the DatetimeIndex BEFORE iterating, or access time via `x[1]`."
                                     )
                                     
        return None

    def _check_variable_consistency(self, tree: ast.AST) -> Optional[str]:
        defined_vars = set()
        used_vars = set()

        init_node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'Initialize'), None)
        if init_node:
            for node in ast.walk(init_node):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                            defined_vars.add(target.attr)

        qc_whitelist = {
            'SMA', 'EMA', 'RSI', 'ATR', 'MACD', 'BB', 'KCH', 'AROON', 'ADX', 'MOM', 'MAX', 'MIN',
            'SetStartDate', 'SetEndDate', 'SetCash', 'AddEquity', 'AddForex', 'AddCrypto', 
            'Schedule', 'Time', 'Portfolio', 'Securities', 'Liquidate', 'SetHoldings', 'MarketOrder',
            'StopMarketOrder', 'LimitOrder', 'Transactions', 'CancelOpenOrders',
            'Debug', 'Log', 'Error', 'TimeRules', 'DateRules', 'History', 'Notify', 'Plot',
            'SubscriptionManager', 'GetParameter', 'LiveMode', 'IsWarmingUp',
            'BrokerageModel', 'SetBrokerageModel', 'SetWarmUp', 'WarmUpPeriod', 'CurrentSlice',
            'UniverseSettings', 'AddUniverse', 'RegisterIndicator', 'ResolveConsolidator', 'Train',
            'ObjectStore', 'PandasConverter', 'Symbol', 'RemoveConsolidator', 'AddConsolidator',
            'SetFillModel', 'TradeBarFillModel', 'SetLeverage','Consolidate','CancelOrder'
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self':
                if node.attr not in qc_whitelist:
                    used_vars.add(node.attr)

        for used in used_vars:
            if used not in defined_vars:
                for defined in defined_vars:
                    if used.lower().replace('_', '') == defined.lower().replace('_', ''):
                        return f"Variable Name Mismatch: Defined `self.{defined}` but used `self.{used}`."
        return None

    def _run_llm_review(self, code: str) -> Optional[str]:
        system_prompt = (
            "You are a Senior QuantConnect Code Reviewer. \n"
            "Architecture and Syntax are verified. \n"
            "YOUR JOB: Check for LOGIC errors.\n"
            "CHECKS:\n"
            "1. **Frequency Check**: Does the exit logic in `OnData` run every minute? Reject if it's gated by a daily flag (e.g. `if self.rebalanceToday:`).\n"
            "2. Infinite loops.\n"
            "3. Logic gaps (e.g. calculating a signal but never using it).\n"
            "4. Math errors (division by zero).\n"
            "5. Look for any look-ahead bias in the code logic. Reject if found and send back to developer to fix with clearly mentioned reason.\n"
            "OUTPUT: Return 'PASS' if good, or a concise error explanation."
        )
        full_prompt = f"{system_prompt}\n\nCODE:\n{code}\n\nREVIEW:"
        review = self.llm.generate(full_prompt).strip()
        
        if "PASS" in review.upper():
            logger.info(f"[Inspector] APPROVED ALGORITHM. (Code hidden for brevity)")
            return None
            
        logger.warning(f"[Inspector] Logic Reject: {review}")
        return f"Logic Review Failed: {review}"