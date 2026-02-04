"""
Tool call adapters for different model formats.
Provides abstraction layer for model-specific tool call parsing and formatting.
"""

from tools.adapters.base import (
    ToolCallAdapter, ToolCall, ToolResult,
    ADAPTER_REGISTRY, register_adapter, get_adapter_for_model, get_all_adapters
)

# Import adapters to register them
from tools.adapters.mistral import MistralToolCallAdapter

__all__ = [
    'ToolCallAdapter', 'ToolCall', 'ToolResult',
    'ADAPTER_REGISTRY', 'register_adapter', 'get_adapter_for_model', 'get_all_adapters',
    'MistralToolCallAdapter',
]
