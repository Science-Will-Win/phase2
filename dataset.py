import json
import os
import torch
from torch.utils.data import Dataset
from utils import get_file_config


class JsonDataset(Dataset):
    """
    A custom PyTorch Dataset for loading text data from a JSON file.
    Supports instruction/output format for instruction tuning.
    
    Expected JSON format:
    [
        {"instruction": "user question", "output": "assistant response"},
        ...
    ]
    
    Labels are masked for input portions (system + user) to only compute
    loss on the assistant's response.
    """
    
    DEFAULT_SYSTEM_PROMPT = "You are a helpful and harmless AI assistant."
    
    def __init__(self, file_path, tokenizer, max_length=2048, model_type="ministral_3_3b_instruct"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load system prompt: prompts folder (priority) > FileConfig path
        file_config = get_file_config(model_type)
        system_prompt_file = file_config.SYSTEM_PROMPT if file_config and hasattr(file_config, 'SYSTEM_PROMPT') else "SYSTEM_PROMPT.txt"
        
        # 1. Try prompts folder first
        prompts_path = os.path.join(os.path.dirname(__file__), "prompts", system_prompt_file)
        if os.path.exists(prompts_path):
            with open(prompts_path, "r", encoding="utf-8") as f:
                self.DEFAULT_SYSTEM_PROMPT = f.read().strip()
        # 2. Fallback to FileConfig path
        elif file_config and hasattr(file_config, 'BASE_PATH'):
            system_prompt_path = os.path.join(file_config.BASE_PATH, system_prompt_file)
            if os.path.exists(system_prompt_path):
                with open(system_prompt_path, "r", encoding="utf-8") as f:
                    self.DEFAULT_SYSTEM_PROMPT = f.read().strip()
        
        if file_path and isinstance(file_path, str):
            with open(file_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Support multiple formats
        if isinstance(item, str):
            # Legacy: raw string (no masking possible)
            instruction = item
            output = ""
        elif "instruction" in item:
            # New format: instruction/output
            instruction = item.get("instruction", "")
            output = item.get("output", "")
        else:
            # Legacy: text field (no masking possible)
            instruction = item.get("text", "")
            output = ""
        
        # Build conversation with system, user, and assistant
        conversation_full = [
            {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        
        # Build conversation without assistant (for computing prompt length)
        conversation_prompt = [
            {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": instruction}
        ]
        
        try:
            # Full text with assistant response
            full_text = self.tokenizer.apply_chat_template(
                conversation_full, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # Prompt only (system + user) with generation prompt
            # This gives us the exact position where assistant starts
            prompt_text = self.tokenizer.apply_chat_template(
                conversation_prompt, 
                tokenize=False, 
                add_generation_prompt=True  # Adds assistant prefix
            )
        except Exception as e:
            print(f"Warning: apply_chat_template failed: {e}. Using fallback format.")
            full_text = f"{self.DEFAULT_SYSTEM_PROMPT}\n\nUser: {instruction}\nAssistant: {output}"
            prompt_text = f"{self.DEFAULT_SYSTEM_PROMPT}\n\nUser: {instruction}\nAssistant: "
        
        # Tokenize full text
        tokenized_full = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize prompt to find response start position
        tokenized_prompt = self.tokenizer(
            prompt_text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = tokenized_full["input_ids"].squeeze(0)
        attention_mask = tokenized_full["attention_mask"].squeeze(0)
        
        # Create labels with masking
        labels = input_ids.clone()
        
        # Get prompt length (where response starts)
        prompt_length = tokenized_prompt["input_ids"].shape[1]
        total_length = len(input_ids)
        
        # Mask prompt portion with -100 (ignored in loss calculation)
        if prompt_length > 0 and output:  # Only mask if there's actual output
            labels[:prompt_length] = -100
        
        # Also mask padding tokens
        labels[attention_mask == 0] = -100
        
        # Debug: print masking stats (first item only, controlled by class flag)
        if not hasattr(self, '_debug_printed') or not self._debug_printed:
            masked_count = (labels == -100).sum().item()
            response_count = total_length - masked_count
            print(f"[Dataset Debug] prompt_length={prompt_length}, total_length={total_length}")
            print(f"[Dataset Debug] masked_tokens={masked_count}, response_tokens={response_count}")
            self._debug_printed = True
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def get_dataset(args, tokenizer):
    """
    Factory function to return a dataset.
    """
    model_type = getattr(args, 'model_type', 'ministral_3_3b_instruct')
    if args.data_path:
        print(f"Loading custom JsonDataset from {args.data_path}")
        return JsonDataset(args.data_path, tokenizer, model_type=model_type)
    else:
        # Fallback to dummy if no path provided
        return DummyDataset(tokenizer)


class DummyDataset(Dataset):
    """
    Dummy dataset for testing with instruction/output format.
    """
    
    DEFAULT_SYSTEM_PROMPT = "You are a helpful and harmless AI assistant named Ministral."
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = [
            {
                "instruction": "What is 2 + 2?",
                "output": "[THINK]I need to add 2 and 2. 2 + 2 = 4.[/THINK]4"
            },
            {
                "instruction": "What is the capital of Japan?",
                "output": "[THINK]Japan is a country in East Asia. Its capital city is Tokyo.[/THINK]Tokyo"
            }
        ]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item["instruction"]
        output = item["output"]
        
        # Build conversation with system, user, and assistant
        conversation_full = [
            {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        
        # Build conversation without assistant (for computing prompt length)
        conversation_prompt = [
            {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": instruction}
        ]
        
        try:
            full_text = self.tokenizer.apply_chat_template(
                conversation_full, 
                tokenize=False, 
                add_generation_prompt=False
            )
            prompt_text = self.tokenizer.apply_chat_template(
                conversation_prompt, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"Warning: apply_chat_template failed: {e}. Using fallback format.")
            full_text = f"{self.DEFAULT_SYSTEM_PROMPT}\n\nUser: {instruction}\nAssistant: {output}"
            prompt_text = f"{self.DEFAULT_SYSTEM_PROMPT}\n\nUser: {instruction}\nAssistant: "
        
        # Tokenize full text
        tokenized_full = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize prompt to find response start position
        tokenized_prompt = self.tokenizer(
            prompt_text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = tokenized_full["input_ids"].squeeze(0)
        attention_mask = tokenized_full["attention_mask"].squeeze(0)
        
        # Create labels with masking
        labels = input_ids.clone()
        prompt_length = tokenized_prompt["input_ids"].shape[1]
        
        # Mask prompt portion with -100
        labels[:prompt_length] = -100
        
        # Also mask padding tokens
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
