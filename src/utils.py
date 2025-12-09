"""
Utility functions for model saving/loading and device detection.
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)


def save_model(model: torch.nn.Module, path: str):
    """
    Save model weights to disk.
    
    Args:
        model: PyTorch model to save
        path: Path where to save the model weights
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


def load_model(model: torch.nn.Module, path: str, device: str = None):
    """
    Load model weights from disk.
    
    Args:
        model: PyTorch model instance to load weights into
        path: Path to the saved model weights
        device: Device to load the model to (if None, uses model's current device)
    
    Returns:
        model: Model with loaded weights
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found at {path}")
    
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    logger.info(f"Model loaded from {path}")
    return model


def get_device_info():
    """
    Detect available device (CPU or GPU) and log device information.
    
    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        logger.info("Using CPU (CUDA not available)")
    
    return device

