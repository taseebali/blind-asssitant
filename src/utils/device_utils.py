"""
Utility functions for device management and configuration
"""

import torch

def get_device():
    """
    Get the best available device (CUDA GPU or CPU).
    
    Returns:
        torch.device: The device to use for computations
    """
    if torch.cuda.is_available():
        # Get the best available GPU
        device = torch.device("cuda")
        # Set device settings for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return device
    elif torch.backends.mps.is_available():  # For Apple M1/M2
        return torch.device("mps")
    else:
        return torch.device("cpu")