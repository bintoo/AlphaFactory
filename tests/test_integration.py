import pytest
from unittest.mock import MagicMock
from alpha_factory.graph import AlphaGraphFactory

def test_full_graph_happy_path():
    """ Verify the graph runs from start to finish with valid inputs."""
    factory = AlphaGraphFactory()
    
    # Mock Shared LLM to return "PASS" for Inspector Logic Review
    # (Architect and Developer are bypassed by method mocks below)
    factory.llm.generate = MagicMock(return_value="PASS")
    
    # Mock Library duplicate check to prevent "Duplicate Strategy" error loop
    factory.library.is_duplicate = MagicMock(return_value=False)

    # MOCK COMPONENTS to ensure deterministic success involves minimal external dependencies
    # 1. Architect returns concept
    factory.architect.produce_strategy_concept = MagicMock(return_value={
        "strategy_concept": "Buy Low Sell High", 
        "processing_mode": "paper_analysis"
    })
    
    # 2. Developer returns valid code
    valid_code = """
class TestAlgo(QCAlgorithm):
    def Initialize(self):
        self.AddEquity("SPY", Resolution.Minute)
        self.SetFillModel(ImmediateFillModel())
    def OnData(self, data):
        self.SetHoldings("SPY", 1)
    def RecoverState(self): pass
"""
    factory.developer.write_code = MagicMock(return_value=valid_code)
    
    # 3. Inspector returns True (mocking internal check or just trusting the code above is valid)
    # Actually, we can let Inspector run real logic on the valid code above
    # But for Integration speed/isolation, we can mock if needed. 
    # Let's use real Inspector to assert our mock valid_code is actually valid.
    
    # 4. Scientist returns APPROVE (New Step)
    factory.scientist.review_implementation = MagicMock(return_value={"decision": "APPROVE", "feedback": "Good"})

    # 5. Library Save (Publisher)
    factory.library.save_alpha = MagicMock()
    
    graph = factory.create_graph()
    
    initial_state = {
        "generation": 1,
        "iteration_count": 0,
        "paper_content": "Test Paper",
        "paper_name": "test.pdf",
        "syntax_error": None,
        "runtime_error": None
    }
    
    final_state = graph.invoke(initial_state)
    
    assert final_state["is_saved"] is True
    factory.library.save_alpha.assert_called_once()

def test_full_graph_syntax_fix_loop():
    """ Verify the graph loops back to developer on syntax error."""
    factory = AlphaGraphFactory()
    factory.llm.generate = MagicMock(return_value="PASS")
    # Mock Library duplicate check
    factory.library.is_duplicate = MagicMock(return_value=False)
    
    factory.architect.produce_strategy_concept = MagicMock(return_value={
        "strategy_concept": "Concept", 
        "processing_mode": "paper_analysis"
    })
    
    # Mock Developer: First call returns BAD code, Second call returns GOOD code
    valid_code = """
class TestAlgo(QCAlgorithm):
    def Initialize(self):
        self.AddEquity("SPY", Resolution.Minute)
        self.SetFillModel(ImmediateFillModel())
    def OnData(self, data):
        self.SetHoldings("SPY", 1)
        pass
    def RecoverState(self): pass
"""
    
    factory.developer.write_code = MagicMock(side_effect=["syntax error code", valid_code])
    
    # Mock Scientist (approves everything for this test)
    factory.scientist.review_implementation = MagicMock(return_value={"decision": "APPROVE", "feedback": "Good"})

    # Mock Scientist (approves everything for this test)
    factory.scientist.review_implementation = MagicMock(return_value={"decision": "APPROVE", "feedback": "Good"})

    # Library save (Publisher Action)
    factory.library.save_alpha = MagicMock()
    
    graph = factory.create_graph()
    
    initial_state = {
        "generation": 1,
        "iteration_count": 0,
        "paper_content": "Test Paper", 
        "paper_name": "test.pdf",
        "syntax_error": None, 
        "runtime_error": None
    }
    
    final_state = graph.invoke(initial_state)
    
    assert final_state["is_saved"] is True
    # Developer should have been called twice
    assert factory.developer.write_code.call_count == 2
    # Ensure save happened
    factory.library.save_alpha.assert_called_once()
