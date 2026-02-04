"""
Code generation tools.
Uses LLM to generate code based on task description.
"""

import os
from typing import Dict, Any
from tools.base import Tool, ToolParameter, register_tool
from tools.biomni.bio_tools import generate_with_llm


# Load code generation system prompt
def load_code_gen_prompt():
    """Load CODE_GEN_PROMPT.txt if available."""
    prompt_path = os.path.join(os.path.dirname(__file__), "../../prompts/CODE_GEN_PROMPT.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return None


@register_tool
class CodeGenTool(Tool):
    """Generate code based on task description."""
    
    name = "code_gen"
    description = "Generate code based on task description. Supports multiple languages."
    parameters = [
        ToolParameter(
            name="task",
            type="string",
            description="What the code should do (detailed description)",
            required=True
        ),
        ToolParameter(
            name="language",
            type="string",
            description="Programming language (python, r, bash, javascript, etc.)",
            required=False,
            default="python"
        ),
        ToolParameter(
            name="context",
            type="string",
            description="Additional context, requirements, or constraints",
            required=False,
            default=""
        )
    ]
    
    def execute(self, task: str = None, language: str = "python", context: str = "", **kwargs) -> Dict[str, Any]:
        """Generate code using LLM.
        
        Args:
            task: Description of what the code should do
            language: Programming language (default: python)
            context: Additional context or requirements
            **kwargs: Additional arguments (may contain step_description, description)
            
        Returns:
            Dict with success status and generated code
        """
        # Fallback for missing task
        if not task:
            # Try to get from kwargs (step context)
            task = kwargs.get('step_description', '') or kwargs.get('description', '')
            if not task:
                task = "Generate data analysis and visualization code."
        
        # Load system prompt for code generation
        system_prompt = load_code_gen_prompt()
        
        # Build prompt for code generation
        prompt = f"Write {language} code to accomplish the following task:\n\n{task}"
        
        if context:
            prompt += f"\n\nAdditional context/requirements:\n{context}"
        
        prompt += "\n\nOutput ONLY the code. No explanations, no markdown code blocks, just the raw code."
        
        # Generate code using LLM with system prompt
        code = generate_with_llm(prompt, max_tokens=1500, system_prompt=system_prompt)
        
        if not code:
            return {
                "success": False,
                "error": "Failed to generate code",
                "result": None
            }
        
        # Clean up code (remove any markdown formatting if present)
        code = code.strip()
        if code.startswith("```"):
            # Remove markdown code block
            lines = code.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)
        
        return {
            "success": True,
            "result": {
                "language": language,
                "code": code,
                "task": task
            }
        }
