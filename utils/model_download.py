"""
Model auto-download utilities for HuggingFace Hub.
"""

import os
from .config_utils import get_file_config


def get_file_config_for_model(model_base_name):
    """Infer architecture from model name and return its FileConfig."""
    arch_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "architectures")
    if not os.path.exists(arch_dir):
        return None
    available = sorted(
        [f[:-3] for f in os.listdir(arch_dir) if f.endswith(".py") and not f.startswith("_")],
        key=len, reverse=True
    )
    for arch_name in available:
        if model_base_name.startswith(arch_name):
            try:
                return get_file_config(arch_name)
            except Exception:
                continue
    return None


def download_from_hf(repo_id, dest_path, verify_fn=None):
    """Download a model from HuggingFace Hub to dest_path.

    Args:
        repo_id: HuggingFace repository ID (e.g. "mistralai/Ministral-3-3B-Instruct-2512")
        dest_path: Local directory to download into
        verify_fn: Optional callable(path) -> bool to verify the download succeeded.
                   If None, always returns True after download.

    Returns:
        True if download succeeded and verification passed, False otherwise.
    """
    try:
        from huggingface_hub import snapshot_download
        print(f"\nModel not found locally. Downloading from HuggingFace: {repo_id}")
        print(f"Destination: {dest_path}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=dest_path,
        )
        if verify_fn and not verify_fn(dest_path):
            print(f"Download succeeded but verification failed for {dest_path}")
            return False
        print(f"Download complete: {dest_path}")
        return True
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"Failed to download model from {repo_id}: {e}")
        return False


def auto_download_model(model_name, base_dir, has_model_files_fn=None):
    """Try to auto-download a model from HuggingFace if not found locally.

    Args:
        model_name: Model name (e.g. "ministral_3_3b_instruct")
        base_dir: Base directory for models (e.g. "model/")
        has_model_files_fn: Optional callable(path) -> bool to verify model files.

    Returns:
        Downloaded model path, or None if download failed.
    """
    model_base_name = os.path.basename(model_name)
    file_config = get_file_config_for_model(model_base_name)
    if file_config and hasattr(file_config, 'HF_REPO_ID'):
        dest = os.path.join(base_dir, model_name)
        if download_from_hf(file_config.HF_REPO_ID, dest, verify_fn=has_model_files_fn):
            return dest
    return None
