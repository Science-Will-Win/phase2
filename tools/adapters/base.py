"""
Base ToolCallAdapter class and registry.
Provides abstraction for model-specific tool call formats.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Unified internal tool call format"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Unified internal tool result format"""
    call_id: str
    name: str
    content: Any
    success: bool = True


class ToolCallAdapter(ABC):
    """Base adapter class for handling model-specific tool call formats"""
    
    name: str = ""
    model_patterns: List[str] = []  # Model name matching patterns
    
    @abstractmethod
    def detect_tool_call(self, text: str) -> bool:
        """Detect if text contains tool call
        
        Args:
            text: LLM output text
            
        Returns:
            True if tool call exists
        """
        pass
    
    @abstractmethod
    def parse_tool_calls(self, text: str) -> Tuple[str, List[ToolCall]]:
        """Parse tool calls from text
        
        Args:
            text: LLM output text
            
        Returns:
            (remaining_text, tool_calls) tuple
        """
        pass
    
    @abstractmethod
    def format_tool_result(self, result: ToolResult) -> Dict[str, Any]:
        """Convert tool result to model-understandable message format
        
        Args:
            result: Unified format tool result
            
        Returns:
            Message dict that model chat template understands
        """
        pass
    
    @abstractmethod
    def format_assistant_with_calls(self, content: str, calls: List[ToolCall]) -> Dict[str, Any]:
        """Format assistant message (with or without tool_calls)
        
        Args:
            content: Assistant response text
            calls: Parsed tool calls
            
        Returns:
            Assistant message dict that model chat template understands
        """
        pass


# Adapter registry
ADAPTER_REGISTRY: Dict[str, ToolCallAdapter] = {}


def register_adapter(adapter_class: type) -> type:
    """Adapter registration decorator
    
    Usage:
        @register_adapter
        class MyAdapter(ToolCallAdapter):
            name = "my_model"
            ...
    """
    instance = adapter_class()
    ADAPTER_REGISTRY[instance.name] = instance
    return adapter_class


def get_adapter_for_model(model_type: str) -> Optional[ToolCallAdapter]:
    """Return adapter matching model type
    
    Args:
        model_type: Model type string (e.g., "ministral_3_3b_instruct")
        
    Returns:
        Matching adapter, or None if not found
    """
    model_type_lower = model_type.lower()
    
    for adapter in ADAPTER_REGISTRY.values():
        for pattern in adapter.model_patterns:
            if pattern in model_type_lower:
                return adapter
    
    # Default: mistral adapter (if available)
    return ADAPTER_REGISTRY.get("mistral")


def get_all_adapters() -> Dict[str, ToolCallAdapter]:
    """Return all registered adapters"""
    return ADAPTER_REGISTRY.copy()
