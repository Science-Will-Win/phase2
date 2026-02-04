"""
Tool system for LLM agent.
Provides tool registration, execution, and management.
"""

from tools.base import Tool, TOOL_REGISTRY, register_tool, get_tool, get_all_tools, get_tools_schema
from tools.executor import execute_tool_call, parse_tool_calls
from tools.tool_router import ToolRouter, should_use_tools, get_system_prompt_for_request

# Register all tools on import
from tools.biomni import bio_tools
from tools.plan import plan_tools

__all__ = [
    'Tool',
    'TOOL_REGISTRY',
    'register_tool',
    'get_tool',
    'get_all_tools',
    'get_tools_schema',
    'execute_tool_call',
    'parse_tool_calls',
    'ToolRouter',
    'should_use_tools',
    'get_system_prompt_for_request',
]
