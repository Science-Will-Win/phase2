"""
Comprehensive Router Prompt Comparison Test
Tests 3 prompts × 2 models × 16 test cases = 96 total tests
Exports results to CSV files in result/ folder
"""
import sys
sys.path.insert(0, "e:\\AIFFEL\\pre-aiffel")

import os
import torch
import argparse
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer

# ============================================================================
# PROMPTS
# ============================================================================

# 1. Simple YES/NO prompt (current TOOL_ROUTER_PROMPT.txt)
SIMPLE_YESNO_PROMPT = """Does this user request need a tool call?

Tool calls are REQUIRED for:
- Creating plans, research plans, study plans, experiment plans
- Creating workflows or pipelines
- Designing experiments or protocols
- Searching literature (PubMed)
- Getting gene/protein information (NCBI)
- Designing CRISPR guides or sgRNAs
- Building experimental protocols
- Any request that says "create", "make", "design", "plan", "build"

Tool calls are NOT needed for:
- General knowledge questions (What is X? How does X work?)
- Definitions and explanations
- Asking for advice or suggestions without creating something
- Questions about concepts, theories, or mechanisms
- Simple conversational responses

Answer with ONLY one word: YES or NO"""

# 2. Arbitrary [THINK] pattern (previous failed format)
ARBITRARY_THINK_PROMPT = """Does this user request need a tool call?

Tool calls are REQUIRED for:
- Creating plans, workflows, protocols
- Designing experiments
- Searching literature, getting gene info
- Building protocols

Tool calls are NOT needed for:
- General questions
- Definitions and explanations
- Advice without creating something

[THINK]your reasoning here[/THINK]
Then answer: YES or NO"""

# 3. SYSTEM_PROMPT format (model native format)
SYSTEM_FORMAT_PROMPT = """# HOW YOU SHOULD THINK AND ANSWER

First draft your thinking process (inner monologue) until you arrive at a response.

Your thinking process must follow the template below:[THINK]Your thoughts[/THINK]Here, provide your answer.

# TASK

Determine if the user request requires CREATING something (plan, workflow, protocol, design).

- If user wants to CREATE/MAKE/DESIGN/BUILD something → YES
- If user is asking questions, seeking explanations, or wants advice → NO

Answer with only: YES or NO"""

PROMPTS = {
    "simple_yesno": SIMPLE_YESNO_PROMPT,
    "arbitrary_think": ARBITRARY_THINK_PROMPT,
    "system_format": SYSTEM_FORMAT_PROMPT
}

# ============================================================================
# TEST CASES
# ============================================================================

TEST_CASES = [
    # Should be YES (tool needed - creating something)
    ("CRISPR을 사용한 유전자 편집 연구 계획을 세워줘", True),
    ("Create a research plan for cancer immunotherapy", True),
    ("실험 workflow를 만들어줘", True),
    ("Design an experiment protocol for protein expression", True),
    ("Give me a study plan for learning bioinformatics", True),
    ("항암제 개발 프로젝트 계획을 작성해줘", True),
    ("Build a pipeline for RNA-seq analysis", True),
    ("Make a protocol for cell culture experiments", True),
    
    # Should be NO (no tool needed - questions/explanations)
    ("CRISPR이 뭐야?", False),
    ("How does PCR work?", False),
    ("유전자 편집의 원리를 설명해줘", False),
    ("What are the advantages of CRISPR over TALENs?", False),
    ("세포 배양에 대해 조언해줘", False),
    ("Can you explain gene expression regulation?", False),
    ("실험 설계할 때 고려해야 할 점이 뭐야?", False),
    ("What's the difference between RNA and DNA?", False),
]

# ============================================================================
# MODEL CONFIGS
# ============================================================================

MODEL_CONFIGS = {
    "ministral_3_3b_reasoning": {
        "path": "e:\\AIFFEL\\pre-aiffel\\model\\ministral_3_3b_reasoning",
        "quantization": "bf16",
        "max_new_tokens": 2048  # Allow space for [THINK] patterns
    },
    "ministral_3_3b_instruct": {
        "path": "e:\\AIFFEL\\pre-aiffel\\model\\ministral_3_3b_instruct",
        "quantization": "fp8",
        "max_new_tokens": 50  # Short responses expected
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model(model_type: str):
    """Load model and tokenizer"""
    from model import get_model
    
    config = MODEL_CONFIGS[model_type]
    
    args = argparse.Namespace(
        model_type=model_type,
        model_path=config["path"],
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
        quantization=config["quantization"]
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config["path"])
    model = get_model(args)
    model.eval()
    
    return model, tokenizer, config


def extract_answer(response: str) -> str:
    """Extract YES/NO from response"""
    clean = response.replace("<s>", "").replace("</s>", "").strip()
    
    # If [/THINK] present, look at part after it
    if "[/THINK]" in clean:
        final_part = clean.split("[/THINK]")[-1].strip()
    else:
        final_part = clean
    
    final_upper = final_part.upper()
    
    # Check final part first
    if "YES" in final_upper and "NO" not in final_upper:
        return "YES"
    elif "NO" in final_upper and "YES" not in final_upper:
        return "NO"
    
    # Fallback: check entire response
    full_upper = clean.upper()
    if "YES" in full_upper:
        return "YES"
    elif "NO" in full_upper:
        return "NO"
    
    return "UNCLEAR"


def run_single_test(model, tokenizer, prompt: str, user_input: str, max_new_tokens: int) -> tuple:
    """Run a single test case and return (predicted, response)"""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
    ]
    
    chat_result = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    )
    
    input_ids = chat_result["input_ids"].to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
    predicted = extract_answer(response)
    
    return predicted, response


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def run_comprehensive_test():
    """Run all tests and collect results"""
    print("=" * 70)
    print("COMPREHENSIVE ROUTER PROMPT COMPARISON TEST")
    print("=" * 70)
    print(f"Models: {list(MODEL_CONFIGS.keys())}")
    print(f"Prompts: {list(PROMPTS.keys())}")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Total tests: {len(MODEL_CONFIGS) * len(PROMPTS) * len(TEST_CASES)}")
    print("=" * 70)
    
    # Collect all results
    all_results = []
    summary_data = []
    
    for model_type in MODEL_CONFIGS.keys():
        print(f"\n{'='*70}")
        print(f"Loading model: {model_type}")
        print("=" * 70)
        
        model, tokenizer, config = load_model(model_type)
        max_tokens = config["max_new_tokens"]
        
        for prompt_name, prompt_text in PROMPTS.items():
            print(f"\n--- Testing prompt: {prompt_name} ---")
            
            correct = 0
            
            for i, (user_input, expected_yes) in enumerate(TEST_CASES):
                expected = "YES" if expected_yes else "NO"
                predicted, response = run_single_test(
                    model, tokenizer, prompt_text, user_input, max_tokens
                )
                
                is_correct = (predicted == expected)
                if is_correct:
                    correct += 1
                
                # Clean response snippet for CSV
                response_clean = response.replace("<s>", "").replace("</s>", "").strip()
                response_snippet = response_clean.replace("\n", " ")
                
                # Store result
                all_results.append({
                    "model": model_type,
                    "prompt_type": prompt_name,
                    "test_case": user_input,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": is_correct,
                    "response_snippet": response_snippet
                })
                
                # Progress indicator
                status = "OK" if is_correct else "FAIL"
                safe_input = user_input[:30].encode('ascii', errors='replace').decode('ascii')
                print(f"  [{i+1:02d}] {status} | {expected} -> {predicted} | {safe_input}...")
            
            # Calculate accuracy
            accuracy = 100 * correct / len(TEST_CASES)
            print(f"\n  Result: {correct}/{len(TEST_CASES)} = {accuracy:.1f}%")
            
            # Store summary
            summary_data.append({
                "model": model_type,
                "prompt_type": prompt_name,
                "total": len(TEST_CASES),
                "correct": correct,
                "accuracy": accuracy
            })
        
        # Free VRAM before loading next model
        del model
        torch.cuda.empty_cache()
    
    return all_results, summary_data


def export_to_csv(all_results: list, summary_data: list):
    """Export results to CSV files"""
    # Ensure result directory exists
    result_dir = "e:\\AIFFEL\\pre-aiffel\\result"
    os.makedirs(result_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detailed results
    df_detailed = pd.DataFrame(all_results)
    detailed_path = os.path.join(result_dir, f"router_prompt_comparison_{timestamp}.csv")
    df_detailed.to_csv(detailed_path, index=False, encoding="utf-8-sig")
    print(f"\nDetailed results saved to: {detailed_path}")
    
    # Summary results
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(result_dir, f"router_prompt_summary_{timestamp}.csv")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Summary results saved to: {summary_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print("=" * 70)
    
    return detailed_path, summary_path


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    all_results, summary_data = run_comprehensive_test()
    export_to_csv(all_results, summary_data)
