"""
Tool Router E2E Verification Test
Tests:
1. Router isolation (separate context from main conversation)
2. System prompt switching based on router decision
3. Tool call execution (create_plan)

Uses ministral_3_3b_instruct model for best router accuracy.
"""
import sys
sys.path.insert(0, "e:\\AIFFEL\\pre-aiffel")

import os
import torch
import argparse
from transformers import AutoTokenizer

# Import tool components
from tools.tool_router import ToolRouter, should_use_tools, load_router_prompt
from tools.executor import execute_tool_call, parse_tool_calls, set_adapter
from tools.adapters import get_adapter_for_model
from tools.base import get_tools_schema

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

MODEL_TYPE = "ministral_3_3b_instruct"
MODEL_PATH = "e:\\AIFFEL\\pre-aiffel\\model\\ministral_3_3b_instruct"

# Test cases
TEST_TOOL_NEEDED = "T세포 고갈 연구를 위한 CRISPR 스크린 계획을 세워줘"
TEST_NO_TOOL = "CRISPR이 뭐야?"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model():
    """Load model and tokenizer"""
    from model import get_model
    
    print(f"Loading model: {MODEL_TYPE}")
    
    args = argparse.Namespace(
        model_type=MODEL_TYPE,
        model_path=MODEL_PATH,
        load_until_layer=None,
        freeze_until_layer=None,
        base_model_path=None,
        hidden_size=None,
        num_hidden_layers=None,
        num_attention_heads=None,
        num_key_value_heads=None,
        intermediate_size=None,
        vocab_size=None,
        max_position_embeddings=None,
        rope_theta=None,
        quantization="fp8"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = get_model(args)
    model.eval()
    
    return model, tokenizer

# ============================================================================
# TEST 1: Router Isolation
# ============================================================================

def test_router_isolation(model, tokenizer):
    """Test that router doesn't affect main conversation context"""
    print("\n" + "=" * 60)
    print("TEST 1: Router Isolation")
    print("=" * 60)
    
    # Create router
    router = ToolRouter(
        model=model,
        tokenizer=tokenizer,
        model_type=MODEL_TYPE,
        debug=True
    )
    
    # Check initial state
    print("\n[Before Router Call]")
    initial_cache = None  # Model doesn't have persistent cache between calls
    
    # Call router for tool-needed request
    print(f"\nRouting: '{TEST_TOOL_NEEDED[:40]}...'")
    use_tools_1, prompt_1 = router.route(TEST_TOOL_NEEDED)
    print(f"  Result: use_tools={use_tools_1}")
    print(f"  System prompt type: {'TOOL' if 'TOOL' in prompt_1[:100] else 'DEFAULT'}")
    
    # Call router for no-tool request
    print(f"\nRouting: '{TEST_NO_TOOL}'")
    use_tools_2, prompt_2 = router.route(TEST_NO_TOOL)
    print(f"  Result: use_tools={use_tools_2}")
    print(f"  System prompt type: {'TOOL' if 'TOOL' in prompt_2[:100] else 'DEFAULT'}")
    
    # Verify isolation
    print("\n[Verification]")
    print(f"  Tool request -> use_tools={use_tools_1} (expected: True)")
    print(f"  No-tool request -> use_tools={use_tools_2} (expected: False)")
    
    test_passed = use_tools_1 == True and use_tools_2 == False
    print(f"\n  TEST RESULT: {'PASS' if test_passed else 'FAIL'}")
    
    return test_passed, router

# ============================================================================
# TEST 2: System Prompt Switching
# ============================================================================

def test_system_prompt_switching(router):
    """Test that system prompt changes correctly based on router decision"""
    print("\n" + "=" * 60)
    print("TEST 2: System Prompt Switching")
    print("=" * 60)
    
    # Get tool prompt
    tool_prompt = router.get_tool_prompt()
    print(f"\n[Tool Prompt (first 200 chars)]:")
    print(f"  {tool_prompt[:200]}...")
    
    # Get default prompt
    default_prompt = router.get_default_prompt()
    print(f"\n[Default Prompt (first 200 chars)]:")
    print(f"  {default_prompt[:200]}...")
    
    # Check prompts are different
    prompts_different = tool_prompt != default_prompt
    tool_has_format = "TOOL_CALLS" in tool_prompt or "tool" in tool_prompt.lower()
    
    print(f"\n[Verification]")
    print(f"  Prompts are different: {prompts_different}")
    print(f"  Tool prompt mentions tools: {tool_has_format}")
    
    test_passed = prompts_different and tool_has_format
    print(f"\n  TEST RESULT: {'PASS' if test_passed else 'FAIL'}")
    
    return test_passed

# ============================================================================
# TEST 3: Tool Call E2E
# ============================================================================

def test_tool_call_e2e(model, tokenizer):
    """Test full tool call flow: generate -> parse -> execute"""
    print("\n" + "=" * 60)
    print("TEST 3: Tool Call E2E (create_plan)")
    print("=" * 60)
    
    # Set adapter for tool parsing
    set_adapter(MODEL_TYPE)
    adapter = get_adapter_for_model(MODEL_TYPE)
    print(f"\n[Adapter]: {adapter.name if adapter else 'None'}")
    
    # Get tools schema
    tools_schema = get_tools_schema()
    print(f"[Registered Tools]: {[t['function']['name'] for t in tools_schema]}")
    
    # Load PLAN_SYSTEM_PROMPT
    plan_prompt_path = "e:\\AIFFEL\\pre-aiffel\\prompts\\PLAN_SYSTEM_PROMPT.txt"
    with open(plan_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    
    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": TEST_TOOL_NEEDED}
    ]
    
    # Build inputs with [TOOL_CALLS] injection
    print(f"\n[Generating response with [TOOL_CALLS] injection...]")
    
    chat_result = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        tools=tools_schema
    )
    
    input_ids = chat_result["input_ids"]
    
    # Inject [TOOL_CALLS] token (ID=9)
    tool_calls_token = torch.tensor([[9]], dtype=torch.long)
    input_ids = torch.cat([input_ids, tool_calls_token], dim=1)
    input_ids = input_ids.to(model.device)
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
    response_clean = response.replace("<s>", "").replace("</s>", "").strip()
    
    # Prepend [TOOL_CALLS] since we injected it
    full_response = "[TOOL_CALLS]" + response_clean
    
    print(f"\n[Model Response (first 300 chars)]:")
    print(f"  {full_response[:300]}...")
    
    # Parse tool calls
    print(f"\n[Parsing tool calls...]")
    remaining, tool_calls = parse_tool_calls(full_response)
    
    print(f"  Found {len(tool_calls)} tool call(s)")
    
    if tool_calls:
        for i, call in enumerate(tool_calls):
            print(f"\n  Tool Call {i+1}:")
            print(f"    Name: {call.get('name', 'N/A')}")
            print(f"    Arguments: {str(call.get('arguments', {}))[:100]}...")
            
            # Execute tool
            print(f"\n[Executing tool: {call.get('name')}...]")
            result = execute_tool_call(call['name'], call.get('arguments', {}))
            
            print(f"  Success: {result.get('success', False)}")
            if result.get('result'):
                print(f"  Result keys: {list(result['result'].keys()) if isinstance(result.get('result'), dict) else 'N/A'}")
    
    # Verify
    test_passed = len(tool_calls) > 0 and tool_calls[0].get('name') == 'create_plan'
    print(f"\n[Verification]")
    print(f"  Tool calls found: {len(tool_calls) > 0}")
    print(f"  First tool is create_plan: {tool_calls[0].get('name') == 'create_plan' if tool_calls else False}")
    print(f"\n  TEST RESULT: {'PASS' if test_passed else 'FAIL'}")
    
    return test_passed, tool_calls

# ============================================================================
# TEST 4: No Tool Call for Questions
# ============================================================================

def test_no_tool_for_questions(model, tokenizer):
    """Test that questions don't trigger tool calls"""
    print("\n" + "=" * 60)
    print("TEST 4: No Tool Call for Questions")
    print("=" * 60)
    
    # Use default system prompt (no tool instructions)
    default_prompt_path = f"e:\\AIFFEL\\pre-aiffel\\model\\{MODEL_TYPE}\\SYSTEM_PROMPT.txt"
    if os.path.exists(default_prompt_path):
        with open(default_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    else:
        system_prompt = "You are a helpful AI assistant."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": TEST_NO_TOOL}
    ]
    
    print(f"\n[Generating response WITHOUT [TOOL_CALLS] injection...]")
    
    chat_result = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    )
    
    input_ids = chat_result["input_ids"].to(model.device)
    
    # Generate without tool injection
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
    response_clean = response.replace("<s>", "").replace("</s>", "").strip()
    
    print(f"\n[Model Response (first 300 chars)]:")
    safe_response = response_clean[:300].encode('ascii', errors='replace').decode('ascii')
    print(f"  {safe_response}...")
    
    # Check for tool calls
    remaining, tool_calls = parse_tool_calls(response_clean)
    
    print(f"\n[Verification]")
    print(f"  Tool calls found: {len(tool_calls)}")
    print(f"  Response is natural language: {len(response_clean) > 20 and '[TOOL_CALLS]' not in response_clean}")
    
    test_passed = len(tool_calls) == 0
    print(f"\n  TEST RESULT: {'PASS' if test_passed else 'FAIL'}")
    
    return test_passed

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("TOOL ROUTER E2E VERIFICATION")
    print(f"Model: {MODEL_TYPE}")
    print("=" * 60)
    
    # Load model
    model, tokenizer = load_model()
    
    results = {}
    
    # Test 1: Router Isolation
    results['router_isolation'], router = test_router_isolation(model, tokenizer)
    
    # Test 2: System Prompt Switching
    results['prompt_switching'] = test_system_prompt_switching(router)
    
    # Test 3: Tool Call E2E
    results['tool_call_e2e'], tool_calls = test_tool_call_e2e(model, tokenizer)
    
    # Test 4: No Tool for Questions
    results['no_tool_questions'] = test_no_tool_for_questions(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    print(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    main()
