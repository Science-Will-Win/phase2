"""
Utility functions for dynamic model configuration loading.
"""

import importlib
import inspect


def get_file_config(model_type: str):
    """
    Dynamically retrieve the FileConfig class for the given model_type.
    Requires a *FileConfig class in architectures/{model_type}.py.
    
    Args:
        model_type: Model type (e.g., "ministral_3_3b_instruct")
        
    Returns:
        FileConfig class or None (if not found)
    """
    try:
        model_module = importlib.import_module(f"architectures.{model_type}")
    except ImportError:
        print(f"[WARNING] Could not import architectures.{model_type}")
        return None
    
    for name, obj in inspect.getmembers(model_module, inspect.isclass):
        if name.endswith("FileConfig"):
            return obj
    
    print(f"[WARNING] No FileConfig class found in architectures.{model_type}")
    return None
