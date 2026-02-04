"""
Analysis tools for generating detailed plan explanations.
Uses LLM to analyze and explain research plans.
"""

from typing import Dict, Any, List
from tools.base import Tool, ToolParameter, register_tool
from tools.biomni.bio_tools import generate_with_llm


# System prompt for analyze_plan tool
ANALYZE_PLAN_SYSTEM_PROMPT = """You are a research plan analysis expert.
Provide detailed and clear explanations for the given research goal and each step.

Analysis considerations:
- Explain the purpose and importance of each step
- Include actual results for completed steps
- Describe current status for in-progress steps
- Explain expected results and methodology for pending steps
- Present overall research flow and direction

Output format (markdown):
## Research Goal
(Background and importance of the goal - 2-3 sentences)

## Overall Research Flow

### Step N: step_name (tool_name) [status]
(Purpose, method, and expected results for this step)
- If completed: Summary of actual results
- If in progress: Current work being performed
- If pending: Expected methodology

## Expected Results and Applications
(Final deliverables and application plans - 2-3 sentences)
"""


@register_tool
class AnalyzePlanTool(Tool):
    """Analyze and explain research plans in detail."""
    
    name = "analyze_plan"
    description = "Generates detailed explanations for analysis plans. LLM analyzes the entire plan and provides direction."
    parameters = [
        ToolParameter(
            name="goal",
            type="string",
            description="Research goal",
            required=True
        ),
        ToolParameter(
            name="steps",
            type="array",
            description="List of all steps. Each step has {name, tool, description, status, result} format",
            required=True
        ),
        ToolParameter(
            name="current_step",
            type="number",
            description="Index of the currently running step (0-based)",
            required=False,
            default=0
        )
    ]
    
    def execute(self, goal: str, steps: List[Dict], current_step: int = 0) -> Dict[str, Any]:
        """Analyze the plan and generate detailed explanation.
        
        Args:
            goal: Research goal
            steps: List of step objects with name, tool, description, status, result
            current_step: Index of currently running step
            
        Returns:
            Dict with success status and analysis text
        """
        # Build context for LLM
        steps_info = []
        for i, step in enumerate(steps):
            status = step.get('status', 'pending')
            if i < current_step:
                status = 'completed'
            elif i == current_step:
                status = 'running'
            else:
                status = 'pending'
            
            status_text = {
                'completed': '✓ Completed',
                'running': '● In Progress',
                'pending': '○ Pending'
            }.get(status, status)
            
            step_info = f"Step {i+1}: {step.get('name', 'Unknown')} ({step.get('tool', 'unknown')}) [{status_text}]"
            step_info += f"\n  Description: {step.get('description', '')}"
            
            if step.get('result'):
                result = step['result']
                if isinstance(result, dict):
                    if result.get('title'):
                        step_info += f"\n  Result: {result['title']}"
                    if result.get('details'):
                        details = result['details'][:3]  # First 3 details
                        step_info += "\n  Details: " + ", ".join(str(d) for d in details)
                else:
                    step_info += f"\n  Result: {str(result)[:200]}"
            
            steps_info.append(step_info)
        
        # Build prompt
        prompt = f"""Research Goal: {goal}

Research Steps:
{chr(10).join(steps_info)}

Please provide a detailed analysis and explanation of the above research plan.
Include the purpose, method, and expected results of each step, and explain the overall research direction."""
        
        # Generate analysis using LLM with system prompt
        analysis = generate_with_llm(
            prompt=prompt,
            system_prompt=ANALYZE_PLAN_SYSTEM_PROMPT,
            max_tokens=1000
        )
        
        if not analysis:
            return {
                "success": False,
                "error": "Analysis generation failed",
                "result": None
            }
        
        return {
            "success": True,
            "result": {
                "analysis": analysis,
                "goal": goal,
                "total_steps": len(steps),
                "current_step": current_step
            }
        }
