"""
Test router with SYSTEM_PROMPT format for reasoning model.
Uses the model's actual SYSTEM_PROMPT template structure.
"""
import sys
sys.path.insert(0, "e:\\AIFFEL\\pre-aiffel")

import torch
from transformers import AutoTokenizer

# System prompt that follows the model's actual SYSTEM_PROMPT format
ROUTER_PROMPT_SYSTEM_FORMAT = """# HOW YOU SHOULD THINK AND ANSWER

First draft your thinking process (inner monologue) until you arrive at a response.

Your thinking process must follow the template below:[THINK]Your thoughts[/THINK]Here, provide your answer.

# TASK

Determine if the user request requires CREATING something (plan, workflow, protocol, design).

- If user wants to CREATE/MAKE/DESIGN/BUILD something → YES
- If user is asking questions, seeking explanations, or wants advice → NO

Answer with only: YES or NO"""

# 16 test cases
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

def test_with_system_format():
    print("=" * 60)
    print("Testing Router with SYSTEM_PROMPT Format (Reasoning Model)")
    print("=" * 60)
    
    # Load reasoning model
    model_path = "e:\\AIFFEL\\pre-aiffel\\model\\ministral_3_3b_reasoning"
    print(f"\nLoading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loading model...")
    from model import get_model
    import argparse
    args = argparse.Namespace(
        model_type="ministral_3_3b_reasoning",
        model_path=model_path,
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
        quantization="bf16"
    )
    
    model = get_model(args)
    model.eval()
    
    print("\n" + "-" * 60)
    print("Router Prompt (SYSTEM_PROMPT Format):")
    print("-" * 60)
    print(ROUTER_PROMPT_SYSTEM_FORMAT)
    print("-" * 60)
    
    results = []
    correct = 0
    
    for i, (user_input, expected_yes) in enumerate(TEST_CASES):
        messages = [
            {"role": "system", "content": ROUTER_PROMPT_SYSTEM_FORMAT},
            {"role": "user", "content": user_input}
        ]
        
        chat_result = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        )
        
        # BatchEncoding - access input_ids
        input_ids = chat_result["input_ids"].to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=200,  # Allow space for thinking
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
        
        # Clean up response for analysis
        clean_response = response.replace("<s>", "").replace("</s>", "").strip()
        
        # Extract final answer after [/THINK] if present
        if "[/THINK]" in clean_response:
            final_part = clean_response.split("[/THINK]")[-1].strip()
        else:
            final_part = clean_response
        
        # Check for YES/NO in the final part
        final_upper = final_part.upper()
        detected_yes = "YES" in final_upper and "NO" not in final_upper
        detected_no = "NO" in final_upper and "YES" not in final_upper
        
        if detected_yes:
            result = "YES"
        elif detected_no:
            result = "NO"
        else:
            # Fallback: check entire response
            full_upper = clean_response.upper()
            if "YES" in full_upper:
                result = "YES"
            elif "NO" in full_upper:
                result = "NO"
            else:
                result = "UNCLEAR"
        
        expected = "YES" if expected_yes else "NO"
        is_correct = result == expected
        if is_correct:
            correct += 1
        
        status = "OK" if is_correct else "FAIL"
        
        # Safe print for mixed language
        safe_input = user_input.encode('ascii', errors='replace').decode('ascii')
        print(f"\n[{i+1:02d}] {status} | Expected: {expected} | Got: {result}")
        print(f"    Input: {safe_input[:50]}...")
        
        # Show thinking if present
        if "[THINK]" in clean_response:
            think_start = clean_response.find("[THINK]") + 7
            think_end = clean_response.find("[/THINK]") if "[/THINK]" in clean_response else len(clean_response)
            thinking = clean_response[think_start:think_end][:100]
            safe_thinking = thinking.encode('ascii', errors='replace').decode('ascii')
            print(f"    Thinking: {safe_thinking}...")
        
        # Show final answer part
        safe_final = final_part[:80].encode('ascii', errors='replace').decode('ascii')
        print(f"    Final: {safe_final}")
        
        results.append({
            "input": user_input,
            "expected": expected,
            "result": result,
            "correct": is_correct,
            "response": clean_response
        })
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {correct}/{len(TEST_CASES)} correct ({100*correct/len(TEST_CASES):.1f}%)")
    print("=" * 60)
    
    # Show failures
    failures = [r for r in results if not r["correct"]]
    if failures:
        print("\nFailed cases:")
        for f in failures:
            safe_input = f["input"].encode('ascii', errors='replace').decode('ascii')
            print(f"  - {safe_input[:40]}... (expected {f['expected']}, got {f['result']})")
    
    return correct, len(TEST_CASES)

if __name__ == "__main__":
    test_with_system_format()
