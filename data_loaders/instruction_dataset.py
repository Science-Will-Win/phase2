import json
import os
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
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
        prompts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", system_prompt_file)
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
        
        # Store raw data for stratified split access
        self.raw_data = self.data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Support multiple formats
        if isinstance(item, str):
            instruction = item
            output = ""
        elif "instruction" in item:
            instruction = item.get("instruction", "")
            output = item.get("output", "")
        else:
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
        
        tokenized_full = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        tokenized_prompt = self.tokenizer(
            prompt_text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = tokenized_full["input_ids"].squeeze(0)
        attention_mask = tokenized_full["attention_mask"].squeeze(0)
        
        labels = input_ids.clone()
        
        prompt_length = tokenized_prompt["input_ids"].shape[1]
        total_length = len(input_ids)
        
        if prompt_length > 0 and output:
            labels[:prompt_length] = -100
        
        labels[attention_mask == 0] = -100
        
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
    Factory function for instruction-tuning datasets.
    
    Returns:
        tuple: (train_dataset, eval_dataset) - eval_dataset is None if val_ratio=0
    """
    model_type = getattr(args, 'model_type', 'ministral_3_3b_instruct')
    val_ratio = getattr(args, 'val_ratio', 0.3)
    stratify_key = getattr(args, 'stratify', None)
    
    if not args.data_path:
        raise ValueError(
            "instruction_dataset requires --data_path. "
            "For testing without data, use --dataset_type dummy_dataset."
        )
    
    print(f"Loading JsonDataset from {args.data_path}")
    full_dataset = JsonDataset(args.data_path, tokenizer, model_type=model_type)
    
    if val_ratio > 0:
        indices = list(range(len(full_dataset)))
        
        stratify_labels = None
        if stratify_key and hasattr(full_dataset, 'raw_data'):
            stratify_labels = [item.get(stratify_key, "unknown") 
                               for item in full_dataset.raw_data]
        
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=42,
            stratify=stratify_labels
        )
        
        train_dataset = Subset(full_dataset, train_indices)
        eval_dataset = Subset(full_dataset, val_indices)
        
        split_type = f"stratified by '{stratify_key}'" if stratify_key else "random"
        print(f"[Dataset] {split_type} split: train={len(train_indices)}, val={len(val_indices)}")
        
        return train_dataset, eval_dataset
    else:
        print(f"[Dataset] No validation split (val_ratio=0)")
        return full_dataset, None
