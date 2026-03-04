"""
Agent multi-turn dataset with role-based loss masking.

Builds token sequences manually (no apply_chat_template) to handle
the agent conversation structure: system -> user -> assistant/tool loops.

Masking strategy:
  - system, user, tool messages -> masked (IGNORE_INDEX)
  - assistant messages           -> unmasked (loss computed)
  - EOS token                    -> unmasked

Usage:
    --dataset_type agent_dataset --data_path formatted_data.json
"""

import json
import os

import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

from masking import IGNORE_INDEX, build_labels_from_boundaries, apply_padding, debug_masking


class AgentDataset(Dataset):
    """
    Dataset for multi-turn agent conversations with role-based loss masking.

    Input format (from format_agent_data.py):
        [
          {
            "messages": [
              {"role": "system",    "content": "..."},
              {"role": "user",      "content": "..."},
              {"role": "assistant", "content": "[THINK]...[/THINK]\n[EXECUTE]...[/EXECUTE]"},
              {"role": "tool",      "content": "[OBSERVATION]...[/OBSERVATION]"},
              ...
            ]
          },
          ...
        ]
    """

    TRAIN_ROLES = ("assistant",)

    def __init__(self, file_path, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self._validate_data()
        self._resolve_special_ids()

        self._debug_printed = False

    def _validate_data(self):
        """Verify that loaded data has the expected structure."""
        if not isinstance(self.data, list) or len(self.data) == 0:
            raise ValueError("Agent data must be a non-empty JSON array.")
        sample = self.data[0]
        if "messages" not in sample:
            raise ValueError(
                "Each instance must have a 'messages' key. "
                "Run format_agent_data.py first."
            )
        for msg in sample["messages"]:
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    "Each message must have 'role' and 'content' keys."
                )

    def _resolve_special_ids(self):
        """Cache special token IDs from tokenizer vocabulary."""
        tok = self.tokenizer
        vocab = tok.get_vocab()

        def _id(token_str):
            tid = vocab.get(token_str)
            if tid is None:
                tid = tok.convert_tokens_to_ids(token_str)
            if tid is None or tid == tok.unk_token_id:
                raise ValueError(
                    f"Special token '{token_str}' not found in tokenizer. "
                    f"Make sure you are using the agent tokenizer."
                )
            return tid

        self.bos_id = _id("<s>")
        self.eos_id = _id("</s>")
        self.pad_id = _id("<pad>") if "<pad>" in vocab else self.eos_id
        self.sys_start_id = _id("[SYSTEM_PROMPT]")
        self.sys_end_id = _id("[/SYSTEM_PROMPT]")
        self.inst_start_id = _id("[INST]")
        self.inst_end_id = _id("[/INST]")

    def _encode_text(self, text):
        """Tokenize plain text without adding special tokens."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _build_sequence(self, messages):
        """
        Build the full token sequence and track role boundaries.

        Sequence layout:
            <s> [SYSTEM_PROMPT] sys_content [/SYSTEM_PROMPT]
            [INST] user_content [/INST]
            assistant_content          (contains [THINK], [EXECUTE], [SOLUTION])
            tool_content               (contains [OBSERVATION])
            assistant_content
            tool_content
            ...
            final_assistant_content
            </s>

        Returns:
            (input_ids, boundaries)
            boundaries: list of (start, end, role) tuples
        """
        input_ids = []
        boundaries = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            start = len(input_ids)

            if role == "system":
                input_ids.append(self.bos_id)
                input_ids.append(self.sys_start_id)
                input_ids.extend(self._encode_text(content))
                input_ids.append(self.sys_end_id)

            elif role == "user":
                input_ids.append(self.inst_start_id)
                input_ids.extend(self._encode_text(content))
                input_ids.append(self.inst_end_id)

            elif role == "assistant":
                input_ids.extend(self._encode_text(content))

            elif role == "tool":
                input_ids.extend(self._encode_text(content))

            end = len(input_ids)
            boundaries.append((start, end, role))

        eos_start = len(input_ids)
        input_ids.append(self.eos_id)
        boundaries.append((eos_start, len(input_ids), "assistant"))

        return input_ids, boundaries

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]
        input_ids, boundaries = self._build_sequence(messages)

        labels = build_labels_from_boundaries(
            input_ids, boundaries, train_roles=self.TRAIN_ROLES
        )
        attention_mask = [1] * len(input_ids)

        input_ids_t, labels_t, attention_mask_t = apply_padding(
            input_ids, labels, attention_mask,
            self.max_length, self.pad_id,
        )

        if not self._debug_printed:
            self._debug_printed = True
            total = len(input_ids)
            trained = sum(1 for l in labels if l != IGNORE_INDEX)
            print(
                f"[AgentDataset] sample {idx}: "
                f"seq_len={total}, trained={trained}, "
                f"masked={total - trained}, "
                f"ratio={trained / max(total, 1):.1%}"
            )

        return {
            "input_ids": input_ids_t,
            "attention_mask": attention_mask_t,
            "labels": labels_t,
        }


def get_dataset(args, tokenizer):
    """
    Factory function for agent dataset. Conforms to data_loaders/ routing protocol.

    Returns:
        (train_dataset, eval_dataset) — eval_dataset is None when val_ratio=0
    """
    if not args.data_path:
        raise ValueError(
            "agent_dataset requires --data_path pointing to preprocessed JSON. "
            "Run data_formatting/format_agent_data.py first."
        )

    max_length = getattr(args, "max_length", 4096)
    val_ratio = getattr(args, "val_ratio", 0.3)

    print(f"[AgentDataset] Loading from {args.data_path}")
    full_dataset = AgentDataset(args.data_path, tokenizer, max_length=max_length)
    print(f"[AgentDataset] {len(full_dataset)} instances loaded")

    if val_ratio > 0 and len(full_dataset) > 1:
        indices = list(range(len(full_dataset)))
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=42,
        )
        train_dataset = Subset(full_dataset, train_idx)
        eval_dataset = Subset(full_dataset, val_idx)
        print(
            f"[AgentDataset] split: train={len(train_idx)}, "
            f"val={len(val_idx)} (ratio={val_ratio})"
        )
        return train_dataset, eval_dataset
    else:
        print("[AgentDataset] No validation split")
        return full_dataset, None
