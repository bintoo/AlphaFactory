from typing import TypedDict, Annotated, Dict, Any, Union
import operator
from langgraph.graph import StateGraph, END
from .agents import (
    ArchitectAgent, DeveloperAgent, InspectorAgent, ScientistAgent,
    LLMInterface, MockLLM, OpenAILLM, DeepSeekLLM, GoogleLLM
)
from .library import AlphaLibrary
from . import config

try:
    import logging
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Define the State
from .state import AgentState

class AlphaGraphFactory:
    def __init__(self):
        self.library = AlphaLibrary()
        self.llm = self._create_llm()
        
        # Initialize Agents
        self.architect = ArchitectAgent(self.library, self.llm)
        self.developer = DeveloperAgent(self.llm)
        # Pass library AND llm to inspector
        self.inspector = InspectorAgent(self.library, self.llm)
        self.scientist = ScientistAgent(self.llm)

    def _create_llm(self) -> LLMInterface:
        provider = config.LLM_PROVIDER.lower()
        if provider == "openai":
            return OpenAILLM(api_key=config.OPENAI_API_KEY, model=config.OPENAI_MODEL)
        elif provider == "deepseek":
            return DeepSeekLLM(
                api_key=config.DEEPSEEK_API_KEY, 
                model=config.DEEPSEEK_MODEL, 
                base_url=config.DEEPSEEK_API_BASE
            )
        elif provider == "google":
            return GoogleLLM(api_key=config.GOOGLE_API_KEY, model=config.GOOGLE_MODEL)
        else:
            return MockLLM()

    def create_graph(self):
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("architect", self.architect_node)
        workflow.add_node("developer", self.developer_node)
        workflow.add_node("inspector", self.inspector_node)
        workflow.add_node("scientist", self.scientist_node)
        workflow.add_node("publisher", self.publisher_node)

        # Entry point
        workflow.set_entry_point("architect")

        # Conditional Logic for Architect (Relevance Check)
        workflow.add_conditional_edges(
            "architect",
            self.route_architect,
            {"developer": "developer", "end": END}
        )

        # Standard Linear Edges
        # workflow.add_edge("architect", "developer")
        workflow.add_edge("developer", "inspector")
        
        # Conditional Logic for Inspector (Syntax & Logic Check)
        workflow.add_conditional_edges(
            "inspector",
            self.route_inspector,
            {"developer": "developer", "scientist": "scientist", "publisher": "publisher"}
        )
        # Conditional Logic for Scientist (Scientific Check)
        workflow.add_conditional_edges(
            "scientist",
            self.route_scientist,
            {"developer": "developer", "publisher": "publisher"}
        )
        
        # Publisher -> End
        workflow.add_edge("publisher", END)

        return workflow.compile()

    # Node Functions
    def architect_node(self, state: AgentState) -> AgentState:
        result = self.architect.produce_strategy_concept(state)
        return result 

    def developer_node(self, state: AgentState) -> AgentState:
        # Takes state (concept, errors), returns code and clears errors
        code = self.developer.write_code(state)
        itr = state.get("iteration_count", 0) + 1
        # Clear all error flags
        return {
            "python_code": code, 
            "syntax_error": None, 
            "runtime_error": None, 
            "inspector_feedback": None,
            "iteration_count": itr
        }

    def inspector_node(self, state: AgentState) -> AgentState:
        # Validate returns None or error string
        is_pdf = bool(state.get("paper_content"))
        feedback = self.inspector.validate(state["python_code"], allow_any_resolution=is_pdf)
        return {"inspector_feedback": feedback}

    def scientist_node(self, state: AgentState) -> AgentState:
        # Scientific Check
        result = self.scientist.review_implementation(state["python_code"], state["paper_content"])
        
        if result["decision"] == "REFINE":
            ref_count = state.get("refinement_count", 0) + 1
            return {"scientific_feedback": result["feedback"], "refinement_count": ref_count}
        
        return {"scientific_feedback": None}

    def publisher_node(self, state: AgentState) -> AgentState:
        """
        Final step: Save the approved strategy to the library/disk.
        Replaces legacy Executor/Critic logic.
        """
        logger.info("Strategy Approved. Publishing to Library...")
        self.library.save_alpha(state)
        return {"is_saved": True}

    # Routing Logic
    def route_architect(self, state: AgentState):
        if state.get("processing_mode") == "irrelevant":
            logger.warning("Graph Routing: Concept Rejected (Irrelevant). Ending Workflow.")
            return "end"
        return "developer"

    def route_inspector(self, state: AgentState):
        if state.get("inspector_feedback"):
            # Limit loops
            if state.get("iteration_count", 0) > 10:
                logger.warning("Max repair attempts reached (Inspector). Giving up.")
                return "scientist" # Fail forward to Scientist
            return "developer"
            
        # Optimization: Only ask Scientist ONCE.
        # If we have already refined (count > 0), trust the Developer/Inspector and go to Publisher.
        if state.get("refinement_count", 0) > 0:
            return "publisher"
            
        return "scientist" # First Pass -> Go to Scientist Review

    def route_scientist(self, state: AgentState):
        if state.get("scientific_feedback"):
             if state.get("refinement_count", 0) > 2:
                 logger.warning("Max refinement attempts reached (Scientist). Proceeding to publication.")
                 return "publisher"
             return "developer"
        return "publisher"
