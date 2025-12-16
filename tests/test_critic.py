import pytest
from alpha_factory.agents import CriticAgent

def test_critic_evaluate_metrics():
    critic = CriticAgent()
    
    # PASS Case
    good_metrics = {
        "Sharpe": 1.5, 
        "Return": 0.40, 
        "Drawdown": 0.10
    }
    result = critic.evaluate(good_metrics)
    assert result["decision"] == "keep"
    
    # FAIL Case (Low Sharpe)
    bad_metrics = {
        "Sharpe": 0.2, 
        "Return": 0.40
    }
    result = critic.evaluate(bad_metrics)
    assert result["decision"] == "kill"

def test_critic_manual_verify():
    critic = CriticAgent()
    
    # Manual Verify Flag present -> Auto Pass
    metrics = {
        "manual_verify": True,
        "Sharpe": 0.0 # Even with bad metrics
    }
    
    result = critic.evaluate(metrics)
    assert result["decision"] == "keep"
