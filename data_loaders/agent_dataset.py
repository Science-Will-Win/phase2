"""
Agent multi-turn dataset with turn-by-turn ShareGPT-style masking.

Each multi-turn conversation in the source JSON is automatically expanded into
N sub-instances (one per assistant turn). Each sub-instance contains all prior
turns as context (masked) and only its final assistant message is trainable.

This is the standard ShareGPT-style training pattern, equivalent to feeding
prior turns as the prompt and letting the model generate the current assistant
response — mirroring inference-time multi-turn behavior.

Masking strategy:
  - system, user, tool messages          -> masked (IGNORE_INDEX)
  - prior assistant messages (context)   -> masked (IGNORE_INDEX)  ← key change
  - last assistant message (target turn) -> unmasked (loss computed)
  - EOS token                            -> unmasked

Usage:
    --dataset_type agent_dataset --data_path formatted_data.json
"""

import json
import os
import re

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

    _SOLUTION_TAG_RE = re.compile(r"\[/?SOLUTION\]")

    @classmethod
    def _strip_orphan_solution(cls, content: str) -> str:
        """Remove unmatched [SOLUTION] / [/SOLUTION] tokens via bracket matching.

        Walks through `content` left-to-right matching each [SOLUTION] with the
        next [/SOLUTION]. Balanced pairs are preserved intact; only orphan
        opens (no matching close) and orphan closes (no preceding open) are
        removed.

        Examples:
            "[SOLUTION]A[/SOLUTION]"                    -> unchanged
            "[SOLUTION]A[/SOLUTION] x [SOLUTION]B"      -> "[SOLUTION]A[/SOLUTION] x B"
            "[SOLUTION]A [SOLUTION]B[/SOLUTION]"        -> "A [SOLUTION]B[/SOLUTION]"
            "[/SOLUTION]A[SOLUTION]B[/SOLUTION]"        -> "A[SOLUTION]B[/SOLUTION]"
        """
        matches = list(cls._SOLUTION_TAG_RE.finditer(content))
        if not matches:
            return content

        stack = []  # spans of unmatched [SOLUTION] (open) tokens
        orphan_closes = []  # spans of orphan [/SOLUTION] (close-before-open) tokens

        for m in matches:
            if m.group() == "[SOLUTION]":
                stack.append((m.start(), m.end()))
            else:  # "[/SOLUTION]"
                if stack:
                    stack.pop()  # matched pair, both kept
                else:
                    orphan_closes.append((m.start(), m.end()))

        # Anything left in stack is orphan-open
        to_remove = stack + orphan_closes
        if not to_remove:
            return content

        # Remove from end to start to keep indices valid
        to_remove.sort(key=lambda span: span[0], reverse=True)
        new_content = content
        for start, end in to_remove:
            new_content = new_content[:start] + new_content[end:]
        return new_content

    def __init__(self, file_path, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Expand: 1 conversation -> N sub-instances (one per assistant turn).
        # Sub-instance for assistant at index i contains messages[0..i] inclusive,
        # where messages[i] is the trainable target. Earlier messages (including
        # any prior assistant turns) become context and will be masked in
        # _build_sequence().
        #
        # Additionally clean up orphan [SOLUTION] tokens in intermediate
        # (non-final-of-original-conversation) assistant turns. The final
        # assistant turn of the original conversation keeps its content
        # untouched (it has properly balanced [SOLUTION] tags).
        self.data = []
        n_cleaned_msgs = 0
        for inst in raw:
            msgs = inst.get("messages", [])
            for i, m in enumerate(msgs):
                if m.get("role") != "assistant":
                    continue
                # Build cleaned sub-instance messages (copy + cleanup).
                # Bracket-matching cleanup is safe for all assistant turns —
                # balanced [SOLUTION]X[/SOLUTION] pairs are preserved.
                cleaned_msgs = []
                for j, mm in enumerate(msgs[: i + 1]):
                    if mm.get("role") == "assistant":
                        new_content = self._strip_orphan_solution(mm["content"])
                        if new_content != mm["content"]:
                            n_cleaned_msgs += 1
                        cleaned_msgs.append({**mm, "content": new_content})
                    else:
                        cleaned_msgs.append(mm)
                sub = {k: v for k, v in inst.items() if k != "messages"}
                sub["_turn_idx"] = i
                sub["messages"] = cleaned_msgs
                self.data.append(sub)

        print(
            f"[AgentDataset] {len(raw)} conversations -> "
            f"{len(self.data)} turn-instances (turn-by-turn expansion)"
        )
        print(
            f"[AgentDataset] orphan [SOLUTION] tags stripped from "
            f"{n_cleaned_msgs} intermediate-turn message copies"
        )

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

        Turn-by-turn masking strategy:
            Only the LAST assistant message in `messages` is treated as the
            trainable target (boundary role = "assistant"). Any prior assistant
            messages (from earlier turns) are treated as context and given the
            boundary role "context", which is masked out by
            build_labels_from_boundaries since it is not in TRAIN_ROLES.

        Sequence layout:
            <s> [SYSTEM_PROMPT] sys_content [/SYSTEM_PROMPT]
            [INST] user_content [/INST]
            assistant_content          (prior turn -> context, masked)
            tool_content               (contains [OBSERVATION])
            ...
            final_assistant_content    (target turn -> assistant, trainable)
            </s>

        If multiple system messages appear, <s> (BOS) is only added once
        at the very beginning. Subsequent system messages use
        [SYSTEM_PROMPT]...[/SYSTEM_PROMPT] without an extra BOS.

        Returns:
            (input_ids, boundaries)
            boundaries: list of (start, end, role) tuples
        """
        input_ids = []
        boundaries = []
        has_bos = False

        # Identify the last assistant index (the only trainable target).
        # Earlier assistant messages are context (masked).
        last_assist_idx = None
        for j in range(len(messages) - 1, -1, -1):
            if messages[j].get("role") == "assistant":
                last_assist_idx = j
                break

        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            start = len(input_ids)

            if role == "system":
                if not has_bos:
                    input_ids.append(self.bos_id)
                    has_bos = True
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

            # Only the last assistant turn is "assistant" (trainable).
            # Prior assistant turns become "context" (masked, since "context"
            # is not in TRAIN_ROLES).
            if role == "assistant" and i != last_assist_idx:
                boundaries.append((start, end, "context"))
            else:
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
        # split_random_state: pulled from args (training.py resolves -1 to a
        # fresh random int and persists it via args.json for resume parity).
        # Fallback to 42 only if args has no attribute at all (back-compat).
        split_rs = getattr(args, "split_random_state", 42)
        if split_rs is None or split_rs < 0:
            split_rs = 42
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=split_rs,
        )
        train_dataset = Subset(full_dataset, train_idx)
        eval_dataset = Subset(full_dataset, val_idx)
        print(
            f"[AgentDataset] split: train={len(train_idx)}, "
            f"val={len(val_idx)} (ratio={val_ratio}, random_state={split_rs})"
        )
        return train_dataset, eval_dataset
    else:
        print("[AgentDataset] No validation split")
        return full_dataset, None
