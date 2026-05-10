"""Convert training output to inference-ready artifacts (Option A).

Combines:
  - Base model weights (unchanged) from base_model_dir
  - Trained log_variance_head from checkpoint's heads/

Produces:
  - Sharded safetensors of base+head (vLLM-compatible)
  - LoRA adapter copied as separate single file (use --lora_path at inference)
  - Config files + tokenizer

Single-process, no FSDP/distributed required.

Usage:
    python data_formatting/convert_fsdp_checkpoint.py <checkpoint_dir> <output_dir>
"""
import argparse
import json
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file as st_load_file, save_file as st_save_file


def _list_safetensors(d: str):
    return sorted(
        f for f in os.listdir(d)
        if f.startswith("model-") and f.endswith(".safetensors")
    )


def _load_base_full_state(base_model_dir: str) -> OrderedDict:
    """Load all base model shards into one OrderedDict."""
    files = _list_safetensors(base_model_dir)
    if not files:
        raise FileNotFoundError(f"No safetensors shards in {base_model_dir}")

    full = OrderedDict()
    total_bytes = 0
    print(f"Loading base model from {base_model_dir} ({len(files)} shard(s))...")
    for fn in files:
        path = os.path.join(base_model_dir, fn)
        sd = st_load_file(path)
        for k, v in sd.items():
            full[k] = v
            total_bytes += v.nelement() * v.element_size()
        print(f"  {fn}: {len(sd)} tensors")
    print(f"  total: {len(full)} tensors, {total_bytes/1e9:.2f} GB")
    return full


def _load_trained_head(ckpt_dir: str) -> OrderedDict:
    """Load log_variance_head from checkpoint's heads/ dir."""
    head_path = os.path.join(ckpt_dir, "heads", "log_variance_head.safetensors")
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"head not found: {head_path}")
    head = st_load_file(head_path)
    print(f"Loaded trained head from {head_path}: {list(head.keys())}")
    return OrderedDict(head)


def _merge_head_into_base(base: OrderedDict, head: OrderedDict) -> int:
    """Replace base's head tensors with trained ones. Returns count of replacements."""
    replaced = 0
    for hk, hv in head.items():
        # head keys are like 'log_variance_head.weight', 'log_variance_head.bias'
        if hk in base:
            base[hk] = hv.contiguous()
            replaced += 1
            print(f"  replaced: {hk} {list(hv.shape)} {hv.dtype}")
        else:
            # try matching by suffix
            matches = [k for k in base if k.endswith(hk)]
            if len(matches) == 1:
                base[matches[0]] = hv.contiguous()
                replaced += 1
                print(f"  replaced (suffix): {matches[0]} <- {hk}")
            else:
                base[hk] = hv.contiguous()
                replaced += 1
                print(f"  added (no match in base): {hk} {list(hv.shape)}")
    return replaced


def _split_into_shards(tensors: OrderedDict, max_bytes: int):
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


def convert(ckpt_dir: str, output_dir: str, base_model_dir: str,
            max_shard_bytes: int = 5_000_000_000):
    if not base_model_dir or not os.path.exists(base_model_dir):
        raise ValueError(f"base_model_dir required and must exist: {base_model_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load base model full state
    base = _load_base_full_state(base_model_dir)

    # 2. Load trained head and merge into base
    head = _load_trained_head(ckpt_dir)
    n_repl = _merge_head_into_base(base, head)
    print(f"Replaced {n_repl} head tensor(s) in base")

    # 3. Re-shard and save
    shards = _split_into_shards(base, max_shard_bytes)
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
        st_save_file(shard, path)
        for k in shard:
            weight_map[k] = name

    index = {
        "metadata": {"total_parameters": total_params, "total_size": total_size},
        "weight_map": OrderedDict(sorted(weight_map.items())),
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nMain (base + head) total: {total_params:,} params, {total_size/1e9:.2f} GB across {n} shard(s)")

    # 4. Copy config files from base model dir
    for fname in ["config.json", "generation_config.json",
                  "preprocessor_config.json", "chat_template.jinja"]:
        src = os.path.join(base_model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    # 5. Copy tokenizer (prefer ckpt_dir's, fallback to base_model_dir)
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        for src_dir in [ckpt_dir, base_model_dir]:
            src = os.path.join(src_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, fname))
                break

    # 6. Copy LoRA adapter from checkpoint (single file, separate from main)
    src_adapter = os.path.join(ckpt_dir, "adapter_model.safetensors")
    src_adapter_cfg = os.path.join(ckpt_dir, "adapter_config.json")
    if os.path.exists(src_adapter):
        adapter_dst_dir = os.path.join(output_dir, "lora_adapter")
        os.makedirs(adapter_dst_dir, exist_ok=True)
        shutil.copy2(src_adapter, os.path.join(adapter_dst_dir, "adapter_model.safetensors"))
        if os.path.exists(src_adapter_cfg):
            shutil.copy2(src_adapter_cfg, os.path.join(adapter_dst_dir, "adapter_config.json"))
        print(f"LoRA adapter copied to {adapter_dst_dir}")

    # 7. Also copy heads/ for transparency (already merged into main, but kept for ref)
    src_heads = os.path.join(ckpt_dir, "heads")
    if os.path.exists(src_heads):
        dst_heads = os.path.join(output_dir, "heads")
        shutil.copytree(src_heads, dst_heads, dirs_exist_ok=True)

    print(f"\nDone. Converted output: {output_dir}")
    print(f"Inference usage:")
    print(f"  --model_path {output_dir}  (sharded base+head)")
    print(f"  --lora_path {output_dir}/lora_adapter  (optional, attach LoRA)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", help="Path to checkpoint-X dir")
    ap.add_argument("output_dir", help="Output dir for sharded base+head")
    ap.add_argument("--base-model-dir", default="model/ministral_3_3b_reasoning_agent")
    ap.add_argument("--max-shard-bytes", type=int, default=5_000_000_000)
    args = ap.parse_args()

    convert(args.ckpt_dir, args.output_dir, args.base_model_dir, args.max_shard_bytes)


if __name__ == "__main__":
    main()
