"""
Early GPU Configuration utility.
Must be imported before torch to set CUDA_VISIBLE_DEVICES.
"""
import argparse
import os


def configure_gpu():
    """
    Parse --local and --gpu args, set CUDA_VISIBLE_DEVICES.
    
    GPU defaults:
    - --gpu specified: Use the specified GPU(s)
    - --local flag: Use GPU 0 (single GPU)
    - Server mode (default): Use GPUs 6,7 (multi GPU)
    
    Returns:
        str: GPU IDs string that was set
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--gpu", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    if args.gpu is not None:
        gpu_ids = args.gpu
    elif args.local:
        gpu_ids = "0"  # Local mode: single GPU
    else:
        gpu_ids = "6,7"  # Server mode: multi GPU
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    print(f"[GPU] Using CUDA_VISIBLE_DEVICES={gpu_ids}")
    return gpu_ids
