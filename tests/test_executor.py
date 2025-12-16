import pytest
import os
from unittest.mock import MagicMock, patch, mock_open
from alpha_factory.agents import ExecutorAgent

def test_parse_cloud_output():
    executor = ExecutorAgent()
    
    # Mock Output (New Format with Box Drawing chars)
    output = """
    Some logs...
    │ Statistic                  │ Value       │
    ├────────────────────────────┼─────────────┤
    │ Sharpe Ratio               │ 1.512       │
    │ Return                     │ 15.20%      │
    │ Drawdown                   │ 5.10%       │
    │ Compounding Annual Return  │ 12.5%       │
    
    Backtest id: 12345
    Backtest url: https://www.quantconnect.com/project/1/backtest/999
    """
    
    res = executor._parse_cloud_output(output)
    metrics = res["backtest_metrics"]
    
    assert metrics["Sharpe"] == 1.512
    assert metrics["Return"] == 15.20
    assert metrics["Drawdown"] == 5.10
    assert metrics["CAGR"] == 12.5
    assert metrics["BacktestURL"] == "https://www.quantconnect.com/project/1/backtest/999"

def test_parse_cloud_output_fail():
    executor = ExecutorAgent()
    output = "No valid table here."
    res = executor._parse_cloud_output(output)
    
    assert "runtime_error" in res
    assert res["runtime_error"] == "No metrics found in Cloud Output. Check logs."

@patch('subprocess.run')
def test_executor_manual_verify(mock_run):
    """Verify manual verification code path."""
    executor = ExecutorAgent()
    
    # Since we commented out cloud execution code in previous session for manual mode,
    # calling execute should return the manual verify dict immediately.
    
    state = {"generation": 1, "strategy_id": "test"}
    res = executor.execute("print('code')", "test_strategy")
    
    assert res["backtest_metrics"]["manual_verify"] is True
    assert res["runtime_error"] is None
