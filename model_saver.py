"""Save trained model in base model format (Mistral3/HuggingFace official).

Converts custom architecture key names back to base model format,
adds missing weights, and saves as sharded safetensors with proper config.

This allows the saved model to be loaded directly by vLLM without conversion.
"""

import json
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


# ── Key renaming rules: custom architecture → base model format ─────
# Mirrors convert_checkpoint.py KEY_RENAME_RULES
KEY_RENAME_RULES = [
    (re.compile(r"^vision_tower\.layers\."), "vision_tower.transformer.layers."),
    (re.compile(r"(vision_tower\.transformer\.layers\.\d+\.)self_attn\."), r"\1attention."),
    (re.compile(r"(vision_tower\.transformer\.layers\.\d+\.)input_layernorm\."), r"\1attention_norm."),
    (re.compile(r"(vision_tower\.transformer\.layers\.\d+\.)post_attention_layernorm\."), r"\1ffn_norm."),
    (re.compile(r"(vision_tower\.transformer\.layers\.\d+\.)mlp\."), r"\1feed_forward."),
    (re.compile(r"^vision_tower\.patch_embedding\.proj\."), "vision_tower.patch_conv."),
    (re.compile(r"^vision_tower\.norm\."), "vision_tower.ln_pre."),
    (re.compile(r"^model\."), "language_model.model."),
    (re.compile(r"^lm_head\."), "language_model.lm_head."),
]

# Keys that may exist only in custom architecture (not in base model)
# Dynamically determined at save time based on model attributes
CUSTOM_ONLY_PREFIXES = ()

# Keys that exist in base model but not in custom architecture
MISSING_KEYS = [
    "multi_modal_projector.norm.weight",
    "multi_modal_projector.patch_merger.merging_layer.weight",
]

# Files to copy from base model directory
BASE_CONFIG_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
]


def _rename_key(key: str) -> str:
    """Apply KEY_RENAME_RULES to convert custom key → base key."""
    for pattern, replacement in KEY_RENAME_RULES:
        key = pattern.sub(replacement, key)
    return key


def _get_missing_weights(base_model_dir: str) -> dict:
    """Load weights that exist in base model but not in custom architecture."""
    base_path = Path(base_model_dir)
    index_path = base_path / "model.safetensors.index.json"

    if not index_path.exists():
        print(f"[model_saver] WARNING: {index_path} not found, skipping missing weights")
        return {}

    with open(index_path) as f:
        index = json.load(f)

    result = {}
    shards_to_load = {}
    for key in MISSING_KEYS:
        shard = index["weight_map"].get(key)
        if shard:
            shards_to_load.setdefault(shard, []).append(key)

    for shard_name, keys in shards_to_load.items():
        shard_path = base_path / shard_name
        if not shard_path.exists():
            print(f"[model_saver] WARNING: shard {shard_path} not found")
            continue
        shard_data = load_file(str(shard_path))
        for key in keys:
            if key in shard_data:
                result[key] = shard_data[key]
                print(f"[model_saver]   + missing weight: {key} {list(shard_data[key].shape)}")

    return result


def save_head_weights(model, base_model_dir):
    """Save trained head weights to base model's heads/ subdirectory.

    Extensible pattern: each custom head gets its own .safetensors file.
    On model load, model_loader.py auto-scans heads/ and loads any found.

    Args:
        model: Trained model (may be PeftModel wrapper)
        base_model_dir: Path to base model directory
    """
    heads_dir = os.path.join(base_model_dir, "heads")
    os.makedirs(heads_dir, exist_ok=True)

    # Unwrap PeftModel to access the actual model
    inner = model
    if hasattr(model, "base_model"):
        inner = model.base_model
    if hasattr(inner, "model"):
        inner = inner.model

    saved_any = False

    # log_variance_head (heteroscedastic uncertainty)
    if hasattr(inner, "log_variance_head"):
        head_state = {
            f"log_variance_head.{k}": v.cpu().contiguous()
            for k, v in inner.log_variance_head.state_dict().items()
        }
        save_path = os.path.join(heads_dir, "log_variance_head.safetensors")
        save_file(head_state, save_path)
        print(f"[model_saver] Saved log_variance_head to {save_path}")
        saved_any = True

    # Future heads can be added here:
    # if hasattr(inner, "token_classification_head"):
    #     ...

    if not saved_any:
        print("[model_saver] No custom heads found to save")


def _split_into_shards(tensors: dict, max_shard_bytes: int) -> list:
    """Split tensors into shards by byte size."""
    shards = []
    current_shard = OrderedDict()
    current_size = 0
    for key in sorted(tensors.keys()):
        tensor = tensors[key]
        tensor_bytes = tensor.nelement() * tensor.element_size()
        if current_size + tensor_bytes > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = OrderedDict()
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor_bytes
    if current_shard:
        shards.append(current_shard)
    return shards


def save_model_as_base_format(
    model,
    tokenizer,
    output_dir: str,
    base_model_dir: str,
    max_shard_bytes: int = 5_000_000_000,
):
    """Save trained model in base model format.

    Args:
        model: Trained PyTorch model (custom architecture)
        tokenizer: Tokenizer to save
        output_dir: Directory to save the converted model
        base_model_dir: Path to base model (for config, missing weights)
        max_shard_bytes: Maximum bytes per shard (~5GB default, matching base)
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[model_saver] Saving model in base format to {output_dir}")

    # Step 1-2: Get state_dict and rename keys
    print("[model_saver] [1/5] Converting state_dict keys...")
    state_dict = model.state_dict()
    renamed = OrderedDict()
    n_renamed = 0
    n_removed = 0

    # Determine which prefixes to skip based on model attributes
    skip_prefixes = list(CUSTOM_ONLY_PREFIXES)
    if hasattr(model, 'log_variance_head'):
        print("[model_saver]   log_variance_head detected, will include")
    else:
        skip_prefixes.append("log_variance_head")
        print("[model_saver]   log_variance_head not in model, will skip")

    # Check tie_word_embeddings: if True, lm_head.weight == embed_tokens.weight
    # Base model doesn't store lm_head separately when tied
    tie_word_embeddings = getattr(model.config, "tie_word_embeddings", False)
    if not tie_word_embeddings:
        text_config = getattr(model.config, "text_config", None)
        if text_config:
            tie_word_embeddings = getattr(text_config, "tie_word_embeddings", False)

    for key, tensor in state_dict.items():
        # Skip weights not present in this model's architecture
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            n_removed += 1
            continue

        # Skip lm_head when tied (base model only stores embed_tokens)
        if tie_word_embeddings and key.startswith("lm_head."):
            print(f"[model_saver]   Skipping {key} (tie_word_embeddings=True)")
            n_removed += 1
            continue

        new_key = _rename_key(key)
        # Move tensors to CPU for saving
        renamed[new_key] = tensor.cpu().contiguous()
        if key != new_key:
            n_renamed += 1

    print(f"[model_saver]   {len(renamed)} weights, {n_renamed} renamed, {n_removed} removed")

    # Step 4: Add missing weights from base model
    if base_model_dir:
        print("[model_saver] [2/5] Adding missing weights from base model...")
        missing = _get_missing_weights(base_model_dir)
        for key, tensor in missing.items():
            renamed[key] = tensor.cpu().contiguous()
        print(f"[model_saver]   Added {len(missing)} missing weights")
    else:
        print("[model_saver] [2/5] No base_model_dir provided, skipping missing weights")

    # Step 5: Save sharded safetensors + index.json
    print("[model_saver] [3/5] Saving sharded safetensors...")
    shards = _split_into_shards(renamed, max_shard_bytes)
    num_shards = len(shards)

    weight_map = {}
    total_size = 0
    for i, shard in enumerate(shards):
        shard_name = f"model-{i+1:05d}-of-{num_shards:05d}.safetensors"
        shard_path = os.path.join(output_dir, shard_name)
        shard_size = sum(t.nelement() * t.element_size() for t in shard.values())
        total_size += shard_size
        print(f"[model_saver]   {shard_name}: {len(shard)} tensors, {shard_size / 1e9:.2f} GB")
        save_file(shard, shard_path)
        for key in shard:
            weight_map[key] = shard_name

    # Save index.json
    total_params = sum(t.nelement() for shard in shards for t in shard.values())
    index = {
        "metadata": {"total_parameters": total_params, "total_size": total_size},
        "weight_map": OrderedDict(sorted(weight_map.items())),
    }
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[model_saver]   Index saved: {total_params:,} params, {total_size / 1e9:.2f} GB total")

    # Step 6: Copy config files from base model
    if base_model_dir:
        print("[model_saver] [4/5] Copying config files from base model...")
        for fname in BASE_CONFIG_FILES:
            src = os.path.join(base_model_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, fname))
                print(f"[model_saver]   Copied {fname}")
    else:
        print("[model_saver] [4/5] No base_model_dir, skipping config copy")

    # Step 7: Save tokenizer
    print("[model_saver] [5/5] Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)

    print(f"[model_saver] Done! Model saved to {output_dir}")
