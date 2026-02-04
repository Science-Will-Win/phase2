"""
Mistral model tool call adapter.
Handles [TOOL_CALLS]name[ARGS]{...} format.
Also supports [TOOL_CALLS]name{...} format (without [ARGS]).
"""

import re
import json
from typing import List, Dict, Any, Tuple

from tools.adapters.base import (
    ToolCallAdapter, ToolCall, ToolResult, register_adapter
)


@register_adapter
class MistralToolCallAdapter(ToolCallAdapter):
    """Handle Mistral model tool call format
    
    Mistral format:
    - Tool call (with ARGS): [TOOL_CALLS]tool_name[ARGS]{"key": "value"}
    - Tool call (without ARGS): [TOOL_CALLS]tool_name{"key": "value"}
    - Tool result: role: "tool" -> template converts to [TOOL_RESULTS]...[/TOOL_RESULTS]
    """
    
    name = "mistral"
    model_patterns = ["mistral", "ministral", "mixtral"]
    
    # Pattern 1: [TOOL_CALLS]name[ARGS]{"key": "value"}
    TOOL_CALL_PATTERN_WITH_ARGS = r'\[TOOL_CALLS\](\w+)\[ARGS\](\{.*?\})(?=\[TOOL_CALLS\]|$|\s*$)'
    
    # Pattern 2: [TOOL_CALLS]name{"key": "value"} (without [ARGS])
    TOOL_CALL_PATTERN_NO_ARGS = r'\[TOOL_CALLS\](\w+)(\{.*?\})(?=\[TOOL_CALLS\]|$|\s*$)'
    
    def detect_tool_call(self, text: str) -> bool:
        """Detect if text contains tool call"""
        # Support both formats: with [ARGS] or without
        if '[TOOL_CALLS]' not in text:
            return False
        # Check for either format
        return ('[ARGS]' in text or 
                re.search(r'\[TOOL_CALLS\]\w+\{', text) is not None)
    
    def parse_tool_calls(self, text: str) -> Tuple[str, List[ToolCall]]:
        """Parse tool calls from text
        
        Args:
            text: LLM output text
            
        Returns:
            (remaining_text, tool_calls) tuple
        """
        tool_calls = []
        remaining = text
        
        # Try pattern with [ARGS] first
        matches = re.findall(self.TOOL_CALL_PATTERN_WITH_ARGS, text, re.DOTALL)
        pattern_used = self.TOOL_CALL_PATTERN_WITH_ARGS
        
        # If no matches, try pattern without [ARGS]
        if not matches:
            matches = re.findall(self.TOOL_CALL_PATTERN_NO_ARGS, text, re.DOTALL)
            pattern_used = self.TOOL_CALL_PATTERN_NO_ARGS
        
        for name, args_str in matches:
            try:
                # Attempt JSON parsing
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                # Store as raw string if JSON parsing fails
                arguments = {"raw": args_str}
            
            tool_calls.append(ToolCall(
                id=f"call_{len(tool_calls)}",
                name=name,
                arguments=arguments
            ))
        
        # Remove tool call parts
        remaining = re.sub(pattern_used, "", remaining, flags=re.DOTALL)
        return remaining.strip(), tool_calls
    
    def format_tool_result(self, result: ToolResult) -> Dict[str, Any]:
        """Convert tool result to Mistral message format
        
        Mistral template (chat_template.jinja) automatically converts
        role: "tool" messages to [TOOL_RESULTS]content[/TOOL_RESULTS]
        """
        # Convert content to JSON string
        if isinstance(result.content, str):
            content = result.content
        else:
            content = json.dumps(result.content, ensure_ascii=False)
        
        # Add instruction to call next tool
        content += "\n\nNow call the next tool. Output ONLY: [TOOL_CALLS]tool_name[ARGS]{...}"
        
        return {
            'role': 'tool',
            'content': content
        }
    
    def format_assistant_with_calls(self, content: str, calls: List[ToolCall]) -> Dict[str, Any]:
        """Format assistant message
        
        Mistral includes [TOOL_CALLS] in text itself,
        so separate tool_calls field is unnecessary
        """
        return {'role': 'assistant', 'content': content}
