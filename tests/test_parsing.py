
import sys
import os
import re
from typing import Dict, Any, Optional

# Mock logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MockExecutor")

# Copy of the new parse logic
def _parse_cloud_output(output: str) -> Dict[str, Any]:
    metrics = {}
    
    # 1. Backtest URL
    url_match = re.search(r"Backtest url:\s*(https://www\.quantconnect\.com/project/\S+)", output)
    if url_match:
        metrics["BacktestURL"] = url_match.group(1)
        print(f"[Executor] [URL] Report: {metrics['BacktestURL']}")

    # Helper to parse Value from "│ Key │ Value │" or "Key: Value"
    def extract_val(key_pattern, text):
        # Try Table Format: │ Key... │ Value... │
        # Regex: │\s*Key\s*│\s*([^│\n]+)
        table_regex = f"│\\s*{key_pattern}\\s*│\\s*([^│\\n]+)"
        match = re.search(table_regex, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try Simple Format: Key: Value
        simple_regex = f"{key_pattern}:\\s*([^\\n]+)"
        match = re.search(simple_regex, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    # 2. Extract Metrics
    # Sharpe Ratio
    raw_sharpe = extract_val("Sharpe Ratio", output)
    if raw_sharpe:
        try:
            metrics["Sharpe"] = float(re.sub(r"[^\d\.\-\+]", "", raw_sharpe))
        except: pass

    # Return
    raw_return = extract_val("Return", output)
    if raw_return:
        try:
            metrics["Return"] = float(re.sub(r"[^\d\.\-\+]", "", raw_return))
        except: pass

    # Compounding Annual Return
    raw_cagr = extract_val("Compounding Annual Return", output)
    if raw_cagr:
        try:
            metrics["CAGR"] = float(re.sub(r"[^\d\.\-\+]", "", raw_cagr))
            if "Return" not in metrics: metrics["Return"] = metrics["CAGR"]
        except: pass

    # Drawdown
    raw_dd = extract_val("Drawdown", output)
    if raw_dd:
        try:
            metrics["Drawdown"] = float(re.sub(r"[^\d\.\-\+]", "", raw_dd))
        except: pass

    return metrics

# The User's Example Input
EXAMPLE_OUTPUT = """
Successfully compiled project 'AutoTest'
Started backtest named 'Focused Tan Cow' for project 'AutoTest'
Backtest url: https://www.quantconnect.com/project/26800571/f217694bacdabbc24703ce21627ab124
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
┌────────────────────────────┬───────────────────┬─────────────────────────────┬──────────────┐
│ Statistic                  │ Value             │ Statistic                   │ Value        │
├────────────────────────────┼───────────────────┼─────────────────────────────┼──────────────┤
│ Equity                     │ $99,921.04        │ Fees                        │ -$28.00      │
│ Holdings                   │ $0.00             │ Net Profit                  │ $-78.96      │
│ Probabilistic Sharpe Ratio │ 2.499%            │ Return                      │ -0.08 %      │
│ Unrealized                 │ $0.00             │ Volume                      │ $26,233.78   │
├────────────────────────────┼───────────────────┼─────────────────────────────┼──────────────┤
│ Total Orders               │ 28                │ Average Win                 │ 0.01%        │
│ Average Loss               │ -0.02%            │ Compounding Annual Return   │ -0.318%      │
│ Drawdown                   │ 0.100%            │ Expectancy                  │ -0.356       │
│ Start Equity               │ 100000            │ End Equity                  │ 99921.04     │
│ Net Profit                 │ -0.079%           │ Sharpe Ratio                │ -19.106      │
│ Sortino Ratio              │ -13.591           │ Probabilistic Sharpe Ratio  │ 2.499%       │
│ Loss Rate                  │ 64%               │ Win Rate                    │ 36%          │
│ Profit-Loss Ratio          │ 0.80              │ Alpha                       │ -0.02        │
│ Beta                       │ 0                 │ Annual Standard Deviation   │ 0.001        │
│ Annual Variance            │ 0                 │ Information Ratio           │ 0.863        │
│ Tracking Error             │ 0.455             │ Treynor Ratio               │ -136.871     │
│ Total Fees                 │ $28.00            │ Estimated Strategy Capacity │ $13000000.00 │
│ Lowest Capacity Asset      │ MSFT R735QTJ8XC9X │ Portfolio Turnover          │ 0.29%        │
│ Drawdown Recovery          │ 1                 │                             │              │
└────────────────────────────┴───────────────────┴─────────────────────────────┴──────────────┘
Backtest id: f217694bacdabbc24703ce21627ab124
Backtest name: Focused Tan Cow
Backtest url: https://www.quantconnect.com/project/26800571/f217694bacdabbc24703ce21627ab124
"""

if __name__ == "__main__":
    print("Testing Parser...")
    res = _parse_cloud_output(EXAMPLE_OUTPUT)
    print("Parsed Metrics:", res)
    
    # Assertions based on example
    # Sharpe: -19.106
    # Return: -0.08
    # Drawdown: 0.100
    # URL: https://www.quantconnect.com/project/26800571/f217694bacdabbc24703ce21627ab124
    
    if res.get("Sharpe") == -19.106: print("✅ Sharpe Correct")
    else: print(f"❌ Sharpe Mismatch: {res.get('Sharpe')}")
    
    if res.get("Return") == -0.08: print("✅ Return Correct")
    else: print(f"❌ Return Mismatch: {res.get('Return')}")
    
    if res.get("Drawdown") == 0.100: print("✅ Drawdown Correct")
    else: print(f"❌ Drawdown Mismatch: {res.get('Drawdown')}")
    
    if "f217694bacdabbc24703ce21627ab124" in res.get("BacktestURL", ""): print("✅ URL Correct")
    else: print(f"❌ URL Mismatch: {res.get('BacktestURL')}")

