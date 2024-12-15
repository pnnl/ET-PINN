# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:16:51 2023

@author: chen096
"""

import numpy as np

def roulette_selection(p, nChoose, replace=False):
    """
    Implements roulette wheel selection.

    Args:
        p (list or np.ndarray): A list or 1D array of probabilities or weights.
                                The values do not need to sum to 1.
        nChoose (int): The number of indices to select.
        replace (bool): Whether to allow selection with replacement.
                        - If `False`, indices will not repeat.
                        - If `True`, indices can be selected multiple times.
                        
    Returns:
        np.ndarray: Array of selected indices.
    """
    # Convert the input probabilities to a NumPy array 
    ps = np.array(p, dtype=np.float64)
    
    # Normalize the probabilities to ensure they sum to 1
    ps /= ps.sum()
    
    # Create an array of indices corresponding to the probabilities
    ind = list(range(ps.shape[0]))
    
    # Use NumPy's random.choice to perform the selection based on probabilities
    return np.random.choice(ind, size=nChoose, p=ps, replace=replace)
