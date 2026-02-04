"""
Tool executor and tool call parsing logic.
Supports adapter-based parsing for different model formats.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from tools.base import get_tool, get_all_tools, get_tools_schema

# Adapter support
_current_adapter = None

def set_adapter(model_type: str):
    """Set the current adapter based on model type.
    
    Args:
        model_type: Model type string (e.g., "ministral_3_3b_instruct")
    """
    global _current_adapter
    try:
        from tools.adapters import get_adapter_for_model
        _current_adapter = get_adapter_for_model(model_type)
        if _current_adapter:
            print(f"[DEBUG] Tool adapter set: {_current_adapter.name}")
    except ImportError:
        print("[Warning] Tool adapters not available")
        _current_adapter = None

def get_current_adapter():
    """Get the current adapter."""
    return _current_adapter


def execute_tool_call(tool_name: str, arguments: Dict[str, Any], debug: bool = True) -> Dict[str, Any]:
    """Execute a single tool call.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Arguments to pass to the tool
        debug: Whether to print debug info to terminal
        
    Returns:
        Tool execution result
    """
    # Handle raw JSON fallback - when JSON parsing failed during tool call extraction
    if 'raw' in arguments and len(arguments) == 1:
        raw_str = arguments['raw']
        if debug:
            print(f"\n[DEBUG] Tool Call: {tool_name}")
            print(f"[DEBUG] Raw argument detected, attempting JSON extraction...")
        try:
            # Find first complete JSON object
            start = raw_str.find('{')
            if start != -1:
                depth = 0
                for i, c in enumerate(raw_str[start:], start):
                    if c == '{': 
                        depth += 1
                    elif c == '}': 
                        depth -= 1
                    if depth == 0:
                        arguments = json.loads(raw_str[start:i+1])
                        if debug:
                            print(f"[DEBUG] Extracted arguments: {arguments}")
                        break
        except Exception as e:
            if debug:
                print(f"[DEBUG] JSON extraction failed: {e}")
            # Keep original raw if extraction fails
    else:
        # DEBUG: Log tool call to terminal
        if debug:
            print(f"\n[DEBUG] Tool Call: {tool_name}")
            print(f"[DEBUG] Arguments: {arguments}")
    
    tool = get_tool(tool_name)
    
    if not tool:
        if debug:
            print(f"[DEBUG] Error: Unknown tool '{tool_name}'")
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}",
            "tool": tool_name
        }
    
    try:
        result = tool.execute(**arguments)
        if debug:
            success = result.get('success', False)
            print(f"[DEBUG] Result: {'Success' if success else 'Failed'}")
        return result
    except Exception as e:
        if debug:
            print(f"[DEBUG] Exception: {e}")
        return {
            "success": False,
            "error": str(e),
            "tool": tool_name
        }


def parse_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse tool calls from LLM output.
    
    Uses adapter if available, otherwise falls back to legacy parsing.
    
    Supports multiple formats:
    1. Adapter-based (model-specific): [TOOL_CALLS]name[ARGS]{...} etc.
    2. JSON block: ```json { "tool_calls": [...] } ```
    3. Function call: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    4. Simple format: [TOOL: tool_name] { arguments }
    
    Args:
        text: LLM output text
        
    Returns:
        Tuple of (remaining_text, list of parsed tool calls)
    """
    # Try adapter first if available
    if _current_adapter is not None:
        try:
            remaining, tool_call_objects = _current_adapter.parse_tool_calls(text)
            # Convert ToolCall objects to dict for backward compatibility
            tool_calls = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in tool_call_objects
            ]
            if tool_calls:
                return remaining, tool_calls
        except Exception as e:
            print(f"[Warning] Adapter parsing failed: {e}")
    
    # Legacy parsing fallback
    return _legacy_parse_tool_calls(text)


def _legacy_parse_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Legacy tool call parsing (multiple formats).
    
    Args:
        text: LLM output text
        
    Returns:
        Tuple of (remaining_text, list of parsed tool calls)
    """
    tool_calls = []
    remaining_text = text
    
    # Pattern 1: JSON block with tool_calls
    json_block_pattern = r'```json\s*(\{[\s\S]*?"tool_calls"[\s\S]*?\})\s*```'
    matches = re.findall(json_block_pattern, text)
    for match in matches:
        try:
            data = json.loads(match)
            if "tool_calls" in data:
                for call in data["tool_calls"]:
                    tool_calls.append({
                        "id": call.get("id", f"call_{len(tool_calls)}"),
                        "name": call.get("function", {}).get("name", call.get("name", "")),
                        "arguments": json.loads(call.get("function", {}).get("arguments", "{}")) 
                                    if isinstance(call.get("function", {}).get("arguments"), str)
                                    else call.get("function", {}).get("arguments", call.get("arguments", {}))
                    })
            remaining_text = remaining_text.replace(f"```json{match}```", "")
        except json.JSONDecodeError:
            pass
    
    # Pattern 2: <tool_call> tags
    tool_call_pattern = r'<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>'
    matches = re.findall(tool_call_pattern, text)
    for match in matches:
        try:
            data = json.loads(match)
            tool_calls.append({
                "id": data.get("id", f"call_{len(tool_calls)}"),
                "name": data.get("name", ""),
                "arguments": data.get("arguments", {})
            })
            remaining_text = remaining_text.replace(f"<tool_call>{match}</tool_call>", "")
        except json.JSONDecodeError:
            pass
    
    # Pattern 3: [TOOL: name] { arguments }
    simple_pattern = r'\[TOOL:\s*(\w+)\]\s*(\{[\s\S]*?\})'
    matches = re.findall(simple_pattern, text)
    for name, args_str in matches:
        try:
            arguments = json.loads(args_str)
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "name": name,
                "arguments": arguments
            })
            remaining_text = re.sub(
                rf'\[TOOL:\s*{name}\]\s*{re.escape(args_str)}',
                "",
                remaining_text
            )
        except json.JSONDecodeError:
            pass
    
    return remaining_text.strip(), tool_calls


def detect_tool_call(text: str) -> bool:
    """Detect if text contains a tool call.
    
    Uses adapter if available.
    
    Args:
        text: LLM output text
        
    Returns:
        True if tool call detected
    """
    if _current_adapter is not None:
        return _current_adapter.detect_tool_call(text)
    
    # Legacy detection
    return (
        '[TOOL_CALLS]' in text or
        '<tool_call>' in text or
        '[TOOL:' in text or
        '"tool_calls"' in text
    )


def format_tool_result_for_llm(tool_result: Dict[str, Any]) -> str:
    """Format tool result as a string for LLM context.
    
    Args:
        tool_result: Result from tool execution
        
    Returns:
        Formatted string for LLM
    """
    if not tool_result.get("success", False):
        return f"[Tool Error] {tool_result.get('error', 'Unknown error')}"
    
    result = tool_result.get("result", {})
    
    lines = []
    lines.append(f"[Tool Result: {tool_result.get('tool', 'unknown')}]")
    
    if tool_result.get("thought"):
        lines.append(f"Thought: {tool_result['thought']}")
    
    if tool_result.get("action"):
        lines.append(f"Action: {tool_result['action']}")
    
    if isinstance(result, dict):
        if result.get("title"):
            lines.append(f"Result: {result['title']}")
        if result.get("details"):
            for detail in result["details"]:
                lines.append(f"  • {detail}")
    else:
        lines.append(f"Result: {result}")
    
    return "\n".join(lines)


def get_tool_call_prompt() -> str:
    """Get the tool call format instructions for LLM.
    
    Returns:
        Instruction string for tool calling
    """
    tools = get_all_tools()
    
    prompt_lines = [
        "You have access to the following tools:",
        ""
    ]
    
    for name, tool in tools.items():
        prompt_lines.append(f"- {name}: {tool.description}")
        if tool.parameters:
            params_str = ", ".join([f"{p.name}: {p.type}" for p in tool.parameters])
            prompt_lines.append(f"  Parameters: {params_str}")
        prompt_lines.append("")
    
    prompt_lines.extend([
        "To use a tool, format your response like this:",
        "<tool_call>",
        '{"name": "tool_name", "arguments": {"param1": "value1"}}',
        "</tool_call>",
        "",
        "After receiving tool results, continue your response based on the information."
    ])
    
    return "\n".join(prompt_lines)


class ToolCallHandler:
    """Handles tool call detection, execution, and response building."""
    
    def __init__(self):
        self.call_history: List[Dict[str, Any]] = []
    
    def process_llm_output(self, text: str) -> Tuple[str, List[Dict[str, Any]], bool]:
        """Process LLM output for tool calls.
        
        Args:
            text: LLM output text
            
        Returns:
            Tuple of (remaining_text, tool_results, has_tool_calls)
        """
        remaining_text, tool_calls = parse_tool_calls(text)
        
        if not tool_calls:
            return text, [], False
        
        tool_results = []
        for call in tool_calls:
            result = execute_tool_call(call["name"], call["arguments"])
            result["call_id"] = call["id"]
            tool_results.append(result)
            
            self.call_history.append({
                "call": call,
                "result": result
            })
        
        return remaining_text, tool_results, True
    
    def build_tool_context(self, tool_results: List[Dict[str, Any]]) -> str:
        """Build context string from tool results for next LLM call.
        
        Args:
            tool_results: List of tool execution results
            
        Returns:
            Context string to append to conversation
        """
        context_parts = []
        for result in tool_results:
            context_parts.append(format_tool_result_for_llm(result))
        
        return "\n\n".join(context_parts)
    
    def reset(self):
        """Reset call history."""
        self.call_history = []


# Global handler instance
tool_handler = ToolCallHandler()
