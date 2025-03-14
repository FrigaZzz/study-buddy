from typing import Dict, Any, Optional, List
from langchain.schema import BaseMessage
from langchain_core.language_models import BaseLLM
import langgraph.graph as lg

class BaseAgent:
    """Base class for all agents in the learning framework."""
    
    def __init__(self, llm: BaseLLM, config: Dict[str, Any] = None):
        """
        Initialize the base agent.
        
        Args:
            llm: Language model to use for the agent
            config: Configuration parameters for the agent
        """
        self.llm = llm
        self.config = config or {}
        
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and return an updated state.
        
        Args:
            state: Current state of the conversation
            
        Returns:
            Updated state
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """Format messages for the agent."""
        return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])