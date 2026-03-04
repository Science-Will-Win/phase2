"""
Role-based loss masking for multi-turn agent training.

Provides incremental tokenization to compute per-message token boundaries,
then builds labels that mask non-training roles with IGNORE_INDEX (-100).
"""

import torch

IGNORE_INDEX = -100


def compute_role_boundaries(tokenizer, messages):
    """
    Compute token start/end indices for each message using incremental tokenization.

    Uses the additive property of apply_chat_template: tokenizing messages[:i+1]
    gives us the cumulative token count up to message i, so the difference between
    consecutive calls gives us the exact token span of each message.

    Args:
        tokenizer: HuggingFace tokenizer with apply_chat_template support.
        messages: List of dicts with 'role' and 'content' keys.

    Returns:
        List of (start_idx, end_idx, role) tuples.
    """
    boundaries = []
    prev_len = 0
    for i in range(len(messages)):
        partial_ids = tokenizer.apply_chat_template(
            messages[:i + 1],
            tokenize=True,
            add_generation_prompt=False
        )
        curr_len = len(partial_ids)
        boundaries.append((prev_len, curr_len, messages[i]["role"]))
        prev_len = curr_len
    return boundaries


def build_labels_from_boundaries(input_ids, boundaries, train_roles=("assistant",)):
    """
    Create labels array where only train_roles tokens have real labels,
    everything else is masked with IGNORE_INDEX.

    Args:
        input_ids: List or tensor of token IDs.
        boundaries: Output of compute_role_boundaries().
        train_roles: Tuple of role strings whose tokens should contribute to loss.

    Returns:
        List of label values (same length as input_ids).
    """
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    labels = [IGNORE_INDEX] * len(input_ids)
    for start, end, role in boundaries:
        if role in train_roles:
            for j in range(start, min(end, len(labels))):
                labels[j] = input_ids[j]
    return labels


def apply_padding(input_ids, labels, attention_mask, max_length, pad_token_id):
    """
    Apply right-padding to input_ids, labels, and attention_mask to reach max_length.
    Truncates if longer than max_length.

    Args:
        input_ids: List of token IDs.
        labels: List of label values.
        attention_mask: List of 0/1 mask values.
        max_length: Target sequence length.
        pad_token_id: Token ID to use for padding.

    Returns:
        Tuple of (input_ids, labels, attention_mask) as torch tensors.
    """
    seq_len = len(input_ids)

    if seq_len > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        attention_mask = attention_mask[:max_length]
    elif seq_len < max_length:
        pad_len = max_length - seq_len
        input_ids = input_ids + [pad_token_id] * pad_len
        labels = labels + [IGNORE_INDEX] * pad_len
        attention_mask = attention_mask + [0] * pad_len

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )


def debug_masking(tokenizer, input_ids, labels, max_display=200):
    """
    Print a debug view showing which tokens are masked vs trained.

    Args:
        tokenizer: HuggingFace tokenizer for decoding.
        input_ids: Tensor of token IDs.
        labels: Tensor of label values (-100 = masked).
        max_display: Max number of tokens to display.
    """
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()

    total = len(input_ids)
    masked = sum(1 for l in labels if l == IGNORE_INDEX)
    trained = total - masked
    pad_count = sum(1 for i, l in zip(input_ids, labels) if l == IGNORE_INDEX and i == tokenizer.pad_token_id)
    content_masked = masked - pad_count

    print(f"\n{'='*60}")
    print(f"[Masking Debug] total={total}, trained={trained}, "
          f"content_masked={content_masked}, padding={pad_count}")
    print(f"[Masking Debug] train_ratio={trained/max(total - pad_count, 1):.1%} "
          f"(excluding padding)")
    print(f"{'='*60}")

    print("\n--- Trained tokens (loss computed) ---")
    trained_ids = [iid for iid, l in zip(input_ids, labels)
                   if l != IGNORE_INDEX]
    if trained_ids:
        trained_text = tokenizer.decode(trained_ids[:max_display])
        print(trained_text[:500])
        if len(trained_ids) > max_display:
            print(f"  ... ({len(trained_ids) - max_display} more tokens)")

    print("\n--- Masked tokens (no loss) ---")
    masked_ids = [iid for iid, l in zip(input_ids, labels)
                  if l == IGNORE_INDEX and iid != tokenizer.pad_token_id]
    if masked_ids:
        masked_text = tokenizer.decode(masked_ids[:max_display])
        print(masked_text[:500])
        if len(masked_ids) > max_display:
            print(f"  ... ({len(masked_ids) - max_display} more tokens)")

    print(f"{'='*60}\n")
