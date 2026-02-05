"""
Model utility functions for DataParallel handling.
"""
import torch
import torch.nn as nn


def maybe_wrap_dataparallel(model):
    """
    Wrap model with DataParallel if multiple GPUs are available.
    
    Args:
        model: PyTorch model to potentially wrap
        
    Returns:
        DataParallel-wrapped model if multiple GPUs, otherwise original model
    """
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    return model


def get_model_device(model):
    """
    Get device of a model, handling DataParallel wrapper.
    
    Args:
        model: PyTorch model (may be wrapped in DataParallel)
    
    Returns:
        torch.device: The device the model is on
    """
    if isinstance(model, nn.DataParallel):
        return next(model.module.parameters()).device
    return next(model.parameters()).device


def get_unwrapped_model(model):
    """
    Get the underlying model (unwrap DataParallel if needed).
    
    Args:
        model: PyTorch model (may be wrapped in DataParallel)
    
    Returns:
        The unwrapped model
    """
    if isinstance(model, nn.DataParallel):
        return model.module
    return model
