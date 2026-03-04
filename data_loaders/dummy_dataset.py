import torch
from torch.utils.data import Dataset


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
        
        conversation_full = [
            {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        
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
        
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def get_dataset(args, tokenizer):
    """
    Factory function for the dummy test dataset.
    
    Returns:
        tuple: (train_dataset, None) - no validation split for dummy data
    """
    return DummyDataset(tokenizer), None
