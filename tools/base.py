"""
Base Tool class and registry for the agent system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ToolParameter:
    """Represents a tool parameter."""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Any = None


class Tool(ABC):
    """Base class for all tools."""
    
    name: str = ""
    description: str = ""
    parameters: List[ToolParameter] = []
    
    def get_schema(self) -> Dict[str, Any]:
        """Return OpenAI-compatible function schema."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given arguments.
        
        Returns:
            Dict with at least 'success' and 'result' or 'error' keys.
        """
        raise NotImplementedError


# Global tool registry
TOOL_REGISTRY: Dict[str, Tool] = {}


def register_tool(tool_class: type) -> type:
    """Decorator to register a tool class."""
    instance = tool_class()
    TOOL_REGISTRY[instance.name] = instance
    return tool_class


def get_tool(name: str) -> Optional[Tool]:
    """Get a registered tool by name."""
    return TOOL_REGISTRY.get(name)


def get_all_tools() -> Dict[str, Tool]:
    """Get all registered tools."""
    return TOOL_REGISTRY.copy()


def get_tools_schema() -> List[Dict[str, Any]]:
    """Get OpenAI-compatible schema for all registered tools."""
    return [tool.get_schema() for tool in TOOL_REGISTRY.values()]


def get_plan_schema() -> List[Dict[str, Any]]:
    """Get schema for create_plan tool only (for plan creation mode).
    
    This ensures LLM can only call create_plan during plan creation phase.
    """
    create_plan = TOOL_REGISTRY.get('create_plan')
    return [create_plan.get_schema()] if create_plan else []


# Internal tools that should be excluded from user-facing prompts
INTERNAL_TOOLS = {'create_plan', 'execute_step'}


def generate_tools_description() -> str:
    """Generate tool descriptions for prompts from registered tools.
    
    Returns:
        String with format: "- tool_name: description" for each tool
    """
    lines = []
    for name, tool in TOOL_REGISTRY.items():
        if name in INTERNAL_TOOLS:
            continue
        lines.append(f"- {name}: {tool.description}")
    return "\n".join(lines)


def generate_tools_format() -> str:
    """Generate tool call format documentation from registered tools.
    
    Returns:
        String with detailed format for each tool including example args
    """
    import json
    sections = []
    for name, tool in TOOL_REGISTRY.items():
        if name in INTERNAL_TOOLS:
            continue
        
        # Build example args from parameters
        args = {}
        for param in tool.parameters:
            if param.default is not None:
                args[param.name] = param.default
            elif param.type == "string":
                args[param.name] = "..."
            elif param.type == "array":
                args[param.name] = ["..."]
            elif param.type == "number":
                args[param.name] = 0
            elif param.type == "object":
                args[param.name] = {}
            elif param.type == "boolean":
                args[param.name] = True
        
        section = f"## {name}\n{tool.description}\n[TOOL_CALLS]{name}[ARGS]{json.dumps(args, ensure_ascii=False)}"
        sections.append(section)
    
    return "\n\n".join(sections)


def get_tool_default_args(tool_name: str) -> Dict[str, Any]:
    """Get default arguments for a tool based on parameter defaults.
    
    Args:
        tool_name: Name of the registered tool
        
    Returns:
        Dict with parameter names and their default values
    """
    tool = TOOL_REGISTRY.get(tool_name)
    if not tool:
        return {}
    
    defaults = {}
    for param in tool.parameters:
        if param.default is not None:
            defaults[param.name] = param.default
    
    return defaults
