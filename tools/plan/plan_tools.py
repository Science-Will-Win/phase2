"""
Plan tools for creating and executing plans.
"""

from typing import Dict, Any, List, Optional
from tools.base import Tool, ToolParameter, register_tool, get_tool


@register_tool
class CreatePlanTool(Tool):
    """Tool for creating a structured plan."""
    
    name = "create_plan"
    description = "Creates a structured plan for research or analysis. Use when breaking down complex tasks into steps."
    parameters = [
        ToolParameter(
            name="goal",
            type="string",
            description="Goal to achieve",
            required=True
        ),
        ToolParameter(
            name="steps",
            type="array",
            description="Plan steps. Each step should have {name, description} format (tool is auto-selected at execution)",
            required=True
        )
    ]
    
    def execute(self, goal: str, steps: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Create a structured plan."""
        # Validate steps
        validated_steps = []
        for i, step in enumerate(steps):
            # Get name with fallback
            name = step.get("name", "").strip() if step.get("name") else ""
            if not name:
                name = f"Step {i + 1}"
            
            # Get description with task fallback (LLM sometimes uses "task" instead of "description")
            description = step.get("description", "") or step.get("task", "")
            
            validated_step = {
                "id": i + 1,
                "name": name,
                "description": description,
                "status": "pending"
            }
            # Note: tool field is intentionally omitted - it will be selected at execution time
            validated_steps.append(validated_step)
        
        return {
            "success": True,
            "tool": self.name,
            "result": {
                "goal": goal,
                "total_steps": len(validated_steps),
                "steps": validated_steps,
                "current_step": 0
            }
        }


@register_tool
class ExecuteStepTool(Tool):
    """Tool for executing a single step in a plan."""
    
    name = "execute_step"
    description = "Executes a specific step in the plan. Calls the tool specified for that step."
    parameters = [
        ToolParameter(
            name="step_id",
            type="number",
            description="Step ID to execute (starts from 1)",
            required=True
        ),
        ToolParameter(
            name="tool_name",
            type="string",
            description="Name of the tool to execute",
            required=True
        ),
        ToolParameter(
            name="tool_args",
            type="object",
            description="Arguments to pass to the tool",
            required=False
        )
    ]
    
    def execute(self, step_id: int, tool_name: str, tool_args: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Execute a specific step by calling its tool."""
        tool = get_tool(tool_name)
        
        if not tool:
            return {
                "success": False,
                "tool": self.name,
                "error": f"Unknown tool: {tool_name}",
                "step_id": step_id
            }
        
        # Execute the tool
        tool_args = tool_args or {}
        tool_result = tool.execute(**tool_args)
        
        return {
            "success": True,
            "tool": self.name,
            "step_id": step_id,
            "executed_tool": tool_name,
            "result": tool_result
        }


class PlanManager:
    """Manages plan state and execution."""
    
    def __init__(self):
        self.current_plan: Optional[Dict[str, Any]] = None
        self.step_results: Dict[int, Dict[str, Any]] = {}
    
    def create_plan(self, goal: str, steps: List[Dict[str, str]]) -> Dict[str, Any]:
        """Create a new plan."""
        create_tool = get_tool("create_plan")
        if create_tool:
            result = create_tool.execute(goal=goal, steps=steps)
            if result.get("success"):
                self.current_plan = result["result"]
                self.step_results = {}
            return result
        return {"success": False, "error": "create_plan tool not found"}
    
    def execute_step(self, step_id: int) -> Dict[str, Any]:
        """Execute a specific step in the current plan."""
        if not self.current_plan:
            return {"success": False, "error": "No active plan"}
        
        steps = self.current_plan.get("steps", [])
        if step_id < 1 or step_id > len(steps):
            return {"success": False, "error": f"Invalid step_id: {step_id}"}
        
        step = steps[step_id - 1]
        tool_name = step.get("tool")
        
        # Get default arguments based on tool type
        tool_args = self._get_default_args(tool_name)
        
        execute_tool = get_tool("execute_step")
        if execute_tool:
            result = execute_tool.execute(
                step_id=step_id,
                tool_name=tool_name,
                tool_args=tool_args
            )
            
            if result.get("success"):
                self.step_results[step_id] = result
                step["status"] = "completed"
                self.current_plan["current_step"] = step_id
            
            return result
        
        return {"success": False, "error": "execute_step tool not found"}
    
    def _get_default_args(self, tool_name: str) -> Dict[str, Any]:
        """Get default arguments for a tool dynamically from tool registry."""
        from tools.base import get_tool_default_args
        return get_tool_default_args(tool_name)
    
    def get_plan_status(self) -> Optional[Dict[str, Any]]:
        """Get current plan status."""
        if not self.current_plan:
            return None
        
        return {
            **self.current_plan,
            "step_results": self.step_results
        }
    
    def reset(self):
        """Reset the plan manager."""
        self.current_plan = None
        self.step_results = {}


# Global plan manager instance
plan_manager = PlanManager()
