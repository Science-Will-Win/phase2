"""
Plan tools for the agent system.
Handles plan creation and step execution.
"""

import os

from tools.plan.plan_tools import CreatePlanTool, ExecuteStepTool, plan_manager

# Path to the plan system prompt
PLAN_PROMPT_PATH = os.path.join(os.path.dirname(__file__), 'plan_system_prompt.txt')


def load_plan_system_prompt() -> str:
    """Load the plan system prompt from file."""
    try:
        with open(PLAN_PROMPT_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""


__all__ = ['CreatePlanTool', 'ExecuteStepTool', 'plan_manager', 'load_plan_system_prompt']
