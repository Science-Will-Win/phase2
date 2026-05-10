"""Convert HF Trainer's pytorch_model_fsdp.bin checkpoint to sharded
vLLM-compatible safetensors format with proper key renaming.

Usage:
    python data_formatting/convert_fsdp_checkpoint.py <checkpoint_dir> <output_dir>

Example:
    python data_formatting/convert_fsdp_checkpoint.py \
        model/train/<exp>/checkpoint-8 \
        model/train/<exp>/converted

This script does NOT need GPU/distributed. It runs on a single process,
reading the FSDP bin file and writing sharded safetensors.
"""
import argparse
import json
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import save_file


# Match model_saver.py's KEY_RENAME_RULES
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

# PEFT prefix to strip
PEFT_PREFIX = "base_model.model."

MISSING_KEYS = [
    "multi_modal_projector.norm.weight",
    "multi_modal_projector.patch_merger.merging_layer.weight",
]


def rename_key(k: str) -> str:
    # Strip PEFT wrapping
    if k.startswith(PEFT_PREFIX):
        k = k[len(PEFT_PREFIX):]
    # Strip LoRA's base_layer wrapper (`q_proj.base_layer.weight` -> `q_proj.weight`)
    k = k.replace(".base_layer.", ".")
    # Apply rename rules
    for pattern, replacement in KEY_RENAME_RULES:
        k = pattern.sub(replacement, k)
    return k


def load_fsdp_bin(ckpt_dir: str) -> dict:
    """Load HF Trainer's FSDP checkpoint bin."""
    bin_path = os.path.join(ckpt_dir, "pytorch_model_fsdp.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"pytorch_model_fsdp.bin not found in {ckpt_dir}")

    print(f"Loading FSDP checkpoint: {bin_path}")
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    print(f"Loaded {len(state)} tensors")
    return state


def filter_lora(state: dict) -> tuple:
    """Split state into (non_lora, lora) dicts."""
    non_lora = OrderedDict()
    lora = OrderedDict()
    for k, v in state.items():
        if "lora_" in k:
            lora[k] = v
        else:
            non_lora[k] = v
    return non_lora, lora


def add_missing_weights(state: dict, base_model_dir: str) -> dict:
    """Load missing weights (multi_modal_projector parts) from base model dir."""
    base_path = Path(base_model_dir)
    index_path = base_path / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"Warning: {index_path} not found, skipping missing weights")
        return state

    with open(index_path) as f:
        index = json.load(f)

    from safetensors.torch import load_file as st_load_file
    shards_to_load = {}
    for key in MISSING_KEYS:
        shard = index["weight_map"].get(key)
        if shard:
            shards_to_load.setdefault(shard, []).append(key)

    for shard_name, keys in shards_to_load.items():
        shard_path = base_path / shard_name
        if not shard_path.exists():
            continue
        shard_data = st_load_file(str(shard_path))
        for k in keys:
            if k in shard_data:
                state[k] = shard_data[k]
                print(f"  + missing: {k} {list(shard_data[k].shape)}")

    return state


def split_into_shards(tensors: dict, max_bytes: int) -> list:
    shards = []
    cur = OrderedDict()
    cur_bytes = 0
    for k in sorted(tensors.keys()):
        t = tensors[k]
        size = t.nelement() * t.element_size()
        if cur_bytes + size > max_bytes and cur:
            shards.append(cur)
            cur = OrderedDict()
            cur_bytes = 0
        cur[k] = t
        cur_bytes += size
    if cur:
        shards.append(cur)
    return shards


def convert(ckpt_dir: str, output_dir: str, base_model_dir: str = None,
            max_shard_bytes: int = 5_000_000_000):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load FSDP checkpoint
    raw = load_fsdp_bin(ckpt_dir)

    # 2. Split LoRA vs non-LoRA
    non_lora, lora = filter_lora(raw)
    print(f"non-LoRA params: {len(non_lora)} tensors")
    print(f"LoRA params:     {len(lora)} tensors")

    # 3. Rename keys (strip PEFT, strip base_layer, apply rename rules)
    renamed = OrderedDict()
    for k, v in non_lora.items():
        new_k = rename_key(k)
        renamed[new_k] = v.to(torch.bfloat16).contiguous()  # ensure bf16

    # 4. Add missing base model weights
    if base_model_dir:
        renamed = add_missing_weights(renamed, base_model_dir)

    # 5. Shard and save
    shards = split_into_shards(renamed, max_shard_bytes)
    n = len(shards)
    weight_map = {}
    total_size = 0
    total_params = 0
    for i, shard in enumerate(shards):
        name = f"model-{i+1:05d}-of-{n:05d}.safetensors"
        path = os.path.join(output_dir, name)
        size = sum(t.nelement() * t.element_size() for t in shard.values())
        params = sum(t.nelement() for t in shard.values())
        total_size += size
        total_params += params
        print(f"Saving {name}: {len(shard)} tensors, {params:,} params, {size/1e9:.2f} GB")
        save_file(shard, path)
        for k in shard:
            weight_map[k] = name

    # 6. index.json
    index = {
        "metadata": {"total_parameters": total_params, "total_size": total_size},
        "weight_map": OrderedDict(sorted(weight_map.items())),
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nTotal: {total_params:,} params, {total_size/1e9:.2f} GB across {n} shard(s)")

    # 7. Save LoRA adapter as single file
    if lora:
        adapter_dir = os.path.join(output_dir, "lora_adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        # Strip PEFT prefix and base_layer for LoRA keys too
        lora_clean = OrderedDict()
        for k, v in lora.items():
            ck = k
            if ck.startswith(PEFT_PREFIX):
                ck = ck[len(PEFT_PREFIX):]
            lora_clean[ck] = v.to(torch.bfloat16).contiguous()
        adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        save_file(lora_clean, adapter_path)
        print(f"LoRA adapter saved: {adapter_path} ({len(lora_clean)} tensors)")

        # Copy adapter_config.json from checkpoint dir if present
        src_cfg = os.path.join(ckpt_dir, "adapter_config.json")
        if os.path.exists(src_cfg):
            shutil.copy2(src_cfg, os.path.join(adapter_dir, "adapter_config.json"))

    # 8. Copy config files from base model dir
    if base_model_dir:
        for fname in ["config.json", "generation_config.json",
                      "preprocessor_config.json", "chat_template.jinja"]:
            src = os.path.join(base_model_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, fname))

    # 9. Copy tokenizer from checkpoint dir
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = os.path.join(ckpt_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    # 10. Copy heads/log_variance_head.safetensors if present
    src_heads = os.path.join(ckpt_dir, "heads")
    if os.path.exists(src_heads):
        dst_heads = os.path.join(output_dir, "heads")
        shutil.copytree(src_heads, dst_heads, dirs_exist_ok=True)
        print(f"heads/ copied to {dst_heads}")

    print(f"\nDone. Converted output: {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", help="Path to checkpoint-X dir with pytorch_model_fsdp.bin")
    ap.add_argument("output_dir", help="Output dir for sharded safetensors")
    ap.add_argument("--base-model-dir", default="model/ministral_3_3b_reasoning_agent",
                    help="Base model dir for missing weights + config files")
    ap.add_argument("--max-shard-bytes", type=int, default=5_000_000_000)
    args = ap.parse_args()

    convert(args.ckpt_dir, args.output_dir, args.base_model_dir, args.max_shard_bytes)


if __name__ == "__main__":
    main()
