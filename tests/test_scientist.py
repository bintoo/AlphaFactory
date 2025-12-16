import pytest
from unittest.mock import MagicMock
from alpha_factory.agents.scientist import ScientistAgent

def test_scientist_approve():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "APPROVE"
    
    scientist = ScientistAgent(mock_llm)
    result = scientist.review_implementation("code", "paper")
    
    assert result["decision"] == "APPROVE"
    assert "accurate" in result["feedback"]

def test_scientist_refine():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "REFINE: Missing Stop Loss"
    
    scientist = ScientistAgent(mock_llm)
    result = scientist.review_implementation("code", "paper")
    
    assert result["decision"] == "REFINE"
    assert result["feedback"] == "Missing Stop Loss"

def test_scientist_ambiguous():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "I think it is okay"
    
    scientist = ScientistAgent(mock_llm)
    result = scientist.review_implementation("code", "paper")
    
    # Should default to APPROVE
    assert result["decision"] == "APPROVE"
