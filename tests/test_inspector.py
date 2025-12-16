import pytest
import ast
from unittest.mock import MagicMock
from alpha_factory.agents import InspectorAgent
from alpha_factory.library import AlphaLibrary
from alpha_factory.agents.llm import LLMInterface

class MockLLM(LLMInterface):
    def generate(self, prompt: str) -> str:
        return "PASS"

@pytest.fixture
def inspector():
    lib = MagicMock(spec=AlphaLibrary)
    lib.is_duplicate.return_value = False
    llm = MockLLM()
    return InspectorAgent(lib, llm)

def test_inspector_python_syntax_check(inspector):
    """Test the C# style datetime check."""
    
    # CASE 1: Bad C# Style
    bad_code = "time = self.Time.Year"
    tree = ast.parse(bad_code)
    error = inspector._check_python_syntax_violations(tree)
    assert error is not None
    assert "C#-style DateTime" in error

    # CASE 2: Good Python Style
    good_code = "time = self.Time.year"
    tree = ast.parse(good_code)
    error = inspector._check_python_syntax_violations(tree)
    assert error is None

    # CASE 3: Exempt Qualifiers (Resolution.Minute)
    exempt_code = "res = Resolution.Minute"
    tree = ast.parse(exempt_code)
    error = inspector._check_python_syntax_violations(tree)
    assert error is None
    
    # CASE 4: Exempt Qualifiers (TimeRules.Hour)
    exempt_code_2 = "rule = TimeRules.Hour"
    tree = ast.parse(exempt_code_2)
    assert inspector._check_python_syntax_violations(tree) is None

def test_inspector_zombie_state(inspector):
    """Test detection of state variables without recovery logic."""
    
    # CASE 1: Zombie (No recovery)
    zombie_code = """
class Test(QCAlgorithm):
    def Initialize(self):
        self.state = {}
    def OnData(self, data):
        pass
"""
    tree = ast.parse(zombie_code)
    error = inspector._check_zombie_state(tree, zombie_code)
    assert error is not None
    assert "Zombie State Risk" in error

    # CASE 2: Valid (Recovery)
    valid_code = """
class Test(QCAlgorithm):
    def Initialize(self):
        self.state = {}
        self.RecoverState()
    def RecoverState(self):
        for x in self.Portfolio: pass
"""
    tree = ast.parse(valid_code)
    error = inspector._check_zombie_state(tree, valid_code)
    assert error is None

def test_inspector_cold_start(inspector):
    """Test detection of early return during warmup."""
    
    # CASE 1: Bad (Blocks data collection)
    bad_code = """
class Test(QCAlgorithm):
    def OnData(self, data):
        if self.IsWarmingUp: return
        pass
"""
    tree = ast.parse(bad_code)
    error = inspector._check_cold_start(tree)
    assert error is not None
    assert "Do not use `if self.IsWarmingUp: return`" in error

    # CASE 2: Good
    good_code = """
class Test(QCAlgorithm):
    def OnData(self, data):
        self.Log("Collecting Data")
        if not self.IsWarmingUp:
            self.Trade()
"""
    tree = ast.parse(good_code)
    error = inspector._check_cold_start(tree)
    assert error is None

def test_inspector_validate_full(inspector):
    """Test full validation pipeline."""
    valid_code = """
class ValidAlgo(QCAlgorithm):
    def Initialize(self):
        self.AddEquity("SPY", Resolution.Minute)
        self.SetFillModel(ImmediateFillModel())
    def OnData(self, data):
        self.SetHoldings("SPY", 1.0)
    def RecoverState(self):
        pass
"""
    # Should pass
    result = inspector.validate(valid_code, allow_any_resolution=False)
    assert result is None
