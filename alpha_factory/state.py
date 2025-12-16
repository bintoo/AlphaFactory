from typing import TypedDict, List, Optional, Dict

class AgentState(TypedDict):
    # Context
    generation: int
    processing_mode: str       # 'paper_analysis', 'irrelevant', 'error'
    
    # Current Strategy Data
    strategy_id: str
    strategy_concept: str
    python_code: str
    
    # Execution & Feedback
    syntax_error: Optional[str]
    runtime_error: Optional[str]
    inspector_feedback: Optional[str]
    backtest_metrics: Dict     # {Sharpe, Return, Drawdown...}
    critic_feedback: str       # "PASS" or rejection reason
    is_saved: bool             # True if added to library
    
    # Loop Control
    iteration_count: int       # To prevent infinite fix loops
    refinement_count: int      # To prevent infinite refinement loops (Scientist)

    # Context
    paper_content: Optional[str]
    paper_name: Optional[str]
    saveto: Optional[str]

    # Feedback
    inspector_feedback: Optional[str]
    runtime_error: Optional[str]
    critic_feedback: Optional[str]
    scientific_feedback: Optional[str]
