import random
import numpy as np
import torch

use_cuda = False  # Global flag that indicates whether CUDA is used or not.

# device
def __setup_cuda(to_use_cuda: bool = True):
    """
    Configure the global device setting for Torch tensors.
    If CUDA is available and to_use_cuda=True, sets the default device to 'cuda:0'.
    Otherwise, sets the default device to CPU.
    """
    global use_cuda
    if to_use_cuda and torch.cuda.is_available():
        use_cuda = True
        torch.defaultDevice = torch.device('cuda:0')
    else:
        use_cuda = False
        torch.defaultDevice = torch.device('cpu')
        
    # Set the default device for newly created tensors.
    torch.set_default_device(torch.defaultDevice)
    return


# precision  
def __set_default_float(precision=32):
    """
    Sets the default floating point precision for both NumPy arrays and PyTorch tensors.
    Supported precisions are 32-bit and 64-bit.
    
    Args:
        precision (int): The desired precision. Must be either 32 or 64.
        
    Raises:
        Exception: If an unsupported precision is provided.
    """
    if precision == 32:
        np.defaultReal = np.float32
        torch.defaultReal = torch.float32
    elif precision == 64:
        np.defaultReal = np.float64
        torch.defaultReal = torch.float64
    else:
        raise Exception(f"unknown float type {precision}")
    
    # Set the default dtype for torch operations
    torch.set_default_dtype(torch.defaultReal)
    return


# reproductivity
def __set_random_seed(seed=None):
    """
    Sets all random seeds for Python's random, NumPy, and torch for reproducibility.
    
    Args:
        seed (int): The seed to use. If None, no seed is set (default behavior).
    """
    if seed is not None:
        # Set Python's built-in random seed.
        random.seed(seed)
        
        # Set NumPy's random seed.
        np.random.seed(seed)
        
        # Set PyTorch's random seed for CPU and GPU (if in use).
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def oneKey_configure(use_cuda: bool = None, 
                     precision: int = None, 
                     seed: int = None):
    """
    A convenience function that sets up global configurations in one call.
    
    Args:
        use_cuda (bool, optional): If True, attempt to use CUDA. If None, no change is made.
        precision (int, optional): Set the global float precision (32 or 64). If None, no change.
        seed (int, optional): Seed for all random generators for reproducibility. If None, no change.
    """
    # Configure CUDA usage if requested.
    if use_cuda is not None:
        __setup_cuda(to_use_cuda=use_cuda)
        
    # Set the default floating point precision if requested.
    if precision is not None:
        __set_default_float(precision=precision)
        
    # Set the random seed for reproducibility if requested.
    if seed is not None:
        __set_random_seed(seed=seed)
        
    # Print out the configuration for verification.
    print(f"""
          Configured successfully:
          Device: {torch.defaultDevice}, numpy.dtype: {np.defaultReal}, tensor.dtype: {torch.defaultReal}
          """)


# default configuration:
oneKey_configure(use_cuda=True, precision=32)
