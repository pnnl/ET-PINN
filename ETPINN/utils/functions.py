# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 18:58:16 2023

@author: chen096
"""

from .decorator import treeFunction  
import numpy as np 
import torch  


# Convert a tree-like structure to NumPy arrays with the same architecture 
treeToNumpy = treeFunction(
    lambda x: x.detach().cpu().numpy()  # Detach the tensor from computation graph, move to CPU, and convert to NumPy
    if not isinstance(x, np.ndarray)  # Only apply if the input is not already a NumPy array
    else x  # If the input is already a NumPy array, return it unchanged
)

# Convert a tree-like structure to PyTorch tensors with the same architecture 
treeToTensor = treeFunction(
    lambda x: torch.as_tensor(  # Convert input to a PyTorch tensor
        x, 
        dtype=torch.defaultReal,  # Use the default data type for real numbers
        device=torch.defaultDevice  # Use the default device (e.g., CPU or GPU)
    )
)
