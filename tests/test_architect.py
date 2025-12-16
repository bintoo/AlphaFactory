import pytest
from unittest.mock import MagicMock
from alpha_factory.agents import ArchitectAgent
from alpha_factory.library import AlphaLibrary
from alpha_factory.agents.llm import LLMInterface

class MockLLM(LLMInterface):
    def generate(self, prompt: str) -> str:
        return "MOCKED_STRATEGY_CONCEPT"

@pytest.fixture
def mock_library():
    return MagicMock(spec=AlphaLibrary)

@pytest.fixture
def architect(mock_library):
    llm = MockLLM()
    # Ensure produce_strategy_concept is called correctly
    return ArchitectAgent(mock_library, llm)

def test_architect_pdf_analysis_mode(architect):
    """Verify that Architect uses PDF content to generate strategy."""
    state = {
        "generation": 1,
        "paper_content": "This is a research paper about Mean Reversion.",
        "paper_name": "test_paper.pdf"
    }
    
    result = architect.produce_strategy_concept(state)
    
    assert result["processing_mode"] == "paper_analysis"
    assert result["strategy_concept"] == "MOCKED_STRATEGY_CONCEPT"
    # Verify no parent IDs since it's from paper
    assert "parent_ids" not in result

def test_architect_missing_paper_error(architect):
    """Verify that Architect raises ValueError if paper content is missing."""
    state = {
        "generation": 1,
        "paper_content": None, # Missing content
        "paper_name": None
    }
    
    with pytest.raises(ValueError, match="No PDF Content"):
        architect.produce_strategy_concept(state)

def test_architect_irrelevant_rejection(architect):
    """Verify Architect rejects irrelevant PDF content."""
    # Mock LLM to return IRRELEVANT
    architect.llm.generate = MagicMock(return_value="IRRELEVANT: This is not a trading paper.")
    
    state = {
        "generation": 1,
        "paper_content": "Some biology text.",
        "paper_name": "bio.pdf"
    }
    
    result = architect.produce_strategy_concept(state)
    assert result["processing_mode"] == "irrelevant"
    assert "IRRELEVANT" in result["strategy_concept"]
