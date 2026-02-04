"""
Tool Router - Stage 1 of 2-stage prompt system
Analyzes user input to determine whether a tool call is needed.
"""

import os
import torch
from typing import Optional

from tools.base import generate_tools_description, generate_tools_format


# Default router prompt path
DEFAULT_ROUTER_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "prompts", 
    "TOOL_ROUTER_PROMPT.txt"
)


def _replace_placeholders(prompt: str) -> str:
    """Replace {TOOLS_LIST} and {TOOLS_FORMAT} placeholders with dynamic content."""
    if "{TOOLS_LIST}" in prompt:
        prompt = prompt.replace("{TOOLS_LIST}", generate_tools_description())
    if "{TOOLS_FORMAT}" in prompt:
        prompt = prompt.replace("{TOOLS_FORMAT}", generate_tools_format())
    return prompt


def load_router_prompt(prompt_path: Optional[str] = None) -> str:
    """
    Load the router prompt from file.
    
    Args:
        prompt_path: Path to the router prompt file. If None, uses default.
    
    Returns:
        The router prompt text.
    """
    path = prompt_path or DEFAULT_ROUTER_PROMPT_PATH
    
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
            return _replace_placeholders(prompt)
    
    # Fallback prompt with dynamic tool list
    tools_desc = generate_tools_description()
    return f"""Does this user request need a tool call?

Tool calls are REQUIRED for:
- Planning, designing, creating workflows or experiments
- Using any of these tools:
{tools_desc}

Tool calls are NOT needed for:
- General knowledge questions
- Definitions and explanations
- Simple factual questions

Answer with ONLY one word: YES or NO"""


def should_use_tools(
    model, 
    tokenizer, 
    user_input: str, 
    router_prompt: Optional[str] = None,
    debug: bool = False
) -> bool:
    """
    Stage 1: Determine whether tool call is needed (response within 10 tokens)
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        user_input: The user's input text
        router_prompt: Custom router prompt (optional)
        debug: If True, print debug information
    
    Returns:
        True if tools should be used, False otherwise
    """
    # Load router prompt
    prompt = router_prompt or load_router_prompt()
    
    # Build messages for router
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
    ]
    
    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=True
        )
        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs}
    else:
        # Fallback: simple concatenation
        text = f"{prompt}\n\nUser: {user_input}\nAssistant:"
        inputs = tokenizer(text, return_tensors="pt")
    
    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]
    
    # Generate short response (max 10 tokens)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.0,  # Deterministic
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    
    # Decode response
    response = tokenizer.decode(
        outputs[0][input_length:], 
        skip_special_tokens=True
    ).strip().upper()
    
    if debug:
        print(f"[Router] Input: {user_input[:50]}...")
        print(f"[Router] Response: {response}")
    
    # Check for YES
    result = "YES" in response
    
    if debug:
        print(f"[Router] Use tools: {result}")
    
    return result


def get_system_prompt_for_request(
    model_type: str, 
    use_tools: bool,
    custom_tool_prompt_path: Optional[str] = None,
    model_base_dir: str = "model"
) -> str:
    """
    Return appropriate system prompt based on router result
    
    Args:
        model_type: The model type (e.g., "ministral_3_3b_reasoning")
        use_tools: Whether tools should be used (from should_use_tools)
        custom_tool_prompt_path: Custom path for tool prompt (optional)
        model_base_dir: Base directory for model files
    
    Returns:
        The appropriate system prompt text
    """
    if use_tools:
        # Custom prompt with tool instructions
        tool_prompt_path = custom_tool_prompt_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prompts",
            "PLAN_SYSTEM_PROMPT.txt"
        )
        if os.path.exists(tool_prompt_path):
            with open(tool_prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
                return _replace_placeholders(prompt)
    
    # Model default prompt (no tool instructions)
    model_prompt_path = os.path.join(model_base_dir, model_type, "SYSTEM_PROMPT.txt")
    if os.path.exists(model_prompt_path):
        with open(model_prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    
    # Fallback
    return "You are a helpful AI assistant."


class ToolRouter:
    """
    Tool Router class for managing 2-stage prompt system.
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        model_type: str,
        router_prompt_path: Optional[str] = None,
        tool_prompt_path: Optional[str] = None,
        model_base_dir: str = "model",
        debug: bool = False
    ):
        """
        Initialize the tool router.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            model_type: The model type string
            router_prompt_path: Path to router prompt file
            tool_prompt_path: Path to tool system prompt file
            model_base_dir: Base directory for model files
            debug: Enable debug output
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.router_prompt = load_router_prompt(router_prompt_path)
        self.tool_prompt_path = tool_prompt_path
        self.model_base_dir = model_base_dir
        self.debug = debug
        
        # Cache prompts
        self._tool_prompt = None
        self._default_prompt = None
    
    def route(self, user_input: str) -> tuple[bool, str]:
        """
        Route the user input and return (use_tools, system_prompt).
        
        Args:
            user_input: The user's input text
        
        Returns:
            Tuple of (use_tools: bool, system_prompt: str)
        """
        use_tools = should_use_tools(
            self.model,
            self.tokenizer,
            user_input,
            self.router_prompt,
            self.debug
        )
        
        system_prompt = get_system_prompt_for_request(
            self.model_type,
            use_tools,
            self.tool_prompt_path,
            self.model_base_dir
        )
        
        return use_tools, system_prompt
    
    def get_tool_prompt(self) -> str:
        """Get the tool system prompt (cached)."""
        if self._tool_prompt is None:
            self._tool_prompt = get_system_prompt_for_request(
                self.model_type,
                use_tools=True,
                custom_tool_prompt_path=self.tool_prompt_path,
                model_base_dir=self.model_base_dir
            )
        return self._tool_prompt
    
    def get_default_prompt(self) -> str:
        """Get the default system prompt (cached)."""
        if self._default_prompt is None:
            self._default_prompt = get_system_prompt_for_request(
                self.model_type,
                use_tools=False,
                custom_tool_prompt_path=self.tool_prompt_path,
                model_base_dir=self.model_base_dir
            )
        return self._default_prompt
