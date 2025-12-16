import pytest
from unittest.mock import MagicMock
from alpha_factory.agents import DeveloperAgent
from alpha_factory.agents.llm import LLMInterface
from alpha_factory import config

class MockLLM(LLMInterface):
    def generate(self, prompt: str) -> str:
        # Echo the prompt partially to verify instructions
        return f"```python\n# Code generated based on prompt\npass\n```"

@pytest.fixture
def developer():
    llm = MockLLM()
    return DeveloperAgent(llm)

def test_developer_clean_code(developer):
    """Verify markdown code block extraction."""
    raw_markdown = "Here is the code:\n```python\ndef test(): pass\n```"
    cleaned = developer._clean_code(raw_markdown)
    assert cleaned == "def test(): pass"
    
    raw_no_lang = "```\ndef test(): pass\n```"
    cleaned = developer._clean_code(raw_no_lang)
    assert cleaned == "def test(): pass"

def test_developer_pdf_mode_instruction(developer):
    """Verify Developer uses specific universe and resolution rules in PDF mode."""
    state = {
        "strategy_concept": "Test Concept",
        "paper_content": "Some paper content", # PDF Mode = True
        "inspector_feedback": None
    }
    
    # We mock LLM to capture prompt using a side_effect or spy, 
    # but here we can just trust the integration or use a spy LLM.
    # Let's inspect the generated prompt by spying on llm.generate
    
    developer.llm.generate = MagicMock(return_value="```python\npass\n```")
    
    developer.write_code(state)
    
    args, _ = developer.llm.generate.call_args
    prompt = args[0]
    
    # Check for PDF-specific instructions
    assert "UNIVERSE: Use these Tickers" in prompt # Proxy list or config list
    assert "RESOLUTION: Choose the best resolution" in prompt # Flexible resolution

def test_developer_fix_mode(developer):
    """Verify Developer includes error feedback in prompt."""
    state = {
        "strategy_concept": "Test Concept",
        "paper_content": "Paper",
        "inspector_feedback": "Syntax Error: Fix this.",
        "python_code": "def old_code(): pass"
    }
    
    developer.llm.generate = MagicMock(return_value="```python\npass\n```")
    developer.write_code(state)
    
    args, _ = developer.llm.generate.call_args
    prompt = args[0]
    
    assert "The previous code FAILED validation." in prompt
    assert "Syntax Error: Fix this." in prompt
