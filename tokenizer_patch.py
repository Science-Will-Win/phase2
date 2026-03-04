"""
Standalone tokenizer patching module.

Renames special token slots in tokenizer files (tokenizer.json, tokenizer_config.json)
based on a patches dict {token_id: new_name}. Tracks patch status via
'_agent_patch_version' field in tokenizer_config.json to avoid redundant work.

Can be called programmatically via apply_if_needed() or run directly:
    python tokenizer_patch.py <model_dir> [--patches '{"36":"[EXECUTE]"}'] [--version 1]
"""

import json
import os
import argparse


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def needs_patch(model_dir: str, patch_version: int) -> bool:
    """Check whether tokenizer files need patching."""
    tc_path = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.exists(tc_path):
        return False
    tc = _read_json(tc_path)
    return tc.get("_agent_patch_version") != patch_version


def apply_patches(model_dir: str, patches: dict[int, str], patch_version: int) -> None:
    """
    Apply token renames to tokenizer files.

    Args:
        model_dir: Directory containing tokenizer.json and tokenizer_config.json.
        patches: Mapping of token ID (int) -> new token name (str).
                 e.g. {36: "[EXECUTE]", 37: "[/EXECUTE]", ...}
        patch_version: Version number written to tokenizer_config.json after patching.
    """
    tok_path = os.path.join(model_dir, "tokenizer.json")
    tc_path = os.path.join(model_dir, "tokenizer_config.json")

    id_to_new = {int(k): v for k, v in patches.items()}

    # --- tokenizer.json ---
    tok = _read_json(tok_path)

    # 1) added_tokens array
    for entry in tok.get("added_tokens", []):
        tid = entry.get("id")
        if tid in id_to_new:
            entry["content"] = id_to_new[tid]

    # 2) model.vocab dict  (old_name -> id  =>  new_name -> id)
    vocab = tok.get("model", {}).get("vocab", {})
    for tid, new_name in id_to_new.items():
        old_keys = [k for k, v in vocab.items() if v == tid and k != new_name]
        for old_key in old_keys:
            del vocab[old_key]
        vocab[new_name] = tid

    _write_json(tok_path, tok)

    # --- tokenizer_config.json ---
    tc = _read_json(tc_path)

    for str_id, new_name in ((str(tid), name) for tid, name in id_to_new.items()):
        if str_id in tc.get("added_tokens_decoder", {}):
            tc["added_tokens_decoder"][str_id]["content"] = new_name

    tc["_agent_patch_version"] = patch_version
    _write_json(tc_path, tc)

    print(f"[tokenizer_patch] Applied {len(id_to_new)} token renames (version={patch_version}) in {model_dir}")


def apply_if_needed(model_dir: str, patches: dict[int, str], patch_version: int = 1) -> None:
    """Apply patches only if not already applied at the given version."""
    if not needs_patch(model_dir, patch_version):
        return
    apply_patches(model_dir, patches, patch_version)


# ---- CLI entry point ----

def main():
    parser = argparse.ArgumentParser(description="Patch tokenizer special tokens")
    parser.add_argument("model_dir", help="Path to model directory")
    parser.add_argument(
        "--patches",
        type=str,
        default=None,
        help='JSON string of {token_id: new_name}, e.g. \'{"36":"[EXECUTE]"}\'',
    )
    parser.add_argument("--version", type=int, default=1, help="Patch version number")
    parser.add_argument("--force", action="store_true", help="Apply even if version matches")
    args = parser.parse_args()

    if args.patches:
        patches = {int(k): v for k, v in json.loads(args.patches).items()}
    else:
        try:
            from utils import get_file_config
            config = get_file_config("ministral_3_3b_reasoning_agent")
            if config and hasattr(config, "TOKENIZER_PATCHES"):
                patches = config.TOKENIZER_PATCHES
                args.version = getattr(config, "PATCH_VERSION", args.version)
            else:
                parser.error("No --patches provided and no FileConfig found")
                return
        except ImportError:
            parser.error("No --patches provided and utils module not available")
            return

    if args.force or needs_patch(args.model_dir, args.version):
        apply_patches(args.model_dir, patches, args.version)
    else:
        print(f"[tokenizer_patch] Already at version {args.version}, skipping.")


if __name__ == "__main__":
    main()
