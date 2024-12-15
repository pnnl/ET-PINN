import numpy as np
import torch

# Global caches:
# _cache: Stores computed partial derivatives keyed by (x, y, component, indx, order_x_partial)
# _cacheNeighbour: Stores neighbours for order_x_partial combinations to find related partial derivatives
_cache = {}
_cacheNeighbour = {}

def getNeighbourA(order_x_partial: tuple):
    """
    Find all A-type neighbours of a given partial-derivative order tuple.
    A-type neighbours are used to deduce new partial derivatives from previously computed ones.
    
    For example:
    If we have order_x_partial = (2,3,2),
    a neighbour might be something like (2,2,3) if it can help deduce (2,3,2).

    Explanation:
    - Given order_x_partial, we try to generate new tuples by decreasing one component of the order
      and increasing another. This might help us reuse previously computed derivatives that are
      stored in the cache, avoiding redundant computations.

    Conditions for a valid neighbour:
    - The decremented component remains non-negative.
    - After decrementing and incrementing, certain trailing components should be zero to ensure uniqueness.
    - The indices iminus and iplus must be different (no increment and decrement on the same index).

    Returns:
        A list of tuples (iminus, iplus, tuple(order_x_neighbour))
        where order_x_neighbour is a neighbour configuration.
    """
    # If we have cached neighbours for this order, return them
    if order_x_partial in _cacheNeighbour:
        return _cacheNeighbour[order_x_partial]

    n = len(order_x_partial)
    neighbours = []
    
    # Try all combinations of decreasing one component (iminus) and increasing another (iplus)
    for iminus in range(n): 
        for iplus in range(n):
            # Convert to numpy array to modify easily
            order_x_neighbour = np.array(order_x_partial)
            # Decrease the iminus-th partial order and increase the iplus-th
            order_x_neighbour[iminus] -= 1
            order_x_neighbour[iplus]  += 1
            
            # Conditions:
            if order_x_neighbour[iminus] >= 0 and np.all(order_x_neighbour[iplus+1:] == 0) and iminus != iplus:
                neighbour = (iminus, iplus, tuple(order_x_neighbour))
                neighbours.append(neighbour)
    
    # Cache the neighbours for future queries
    _cacheNeighbour[order_x_partial] = neighbours
    return neighbours


def partialDerivative(x, y, component=0, indx=None, order_x_partial: tuple = (0,)):
    """
    Compute a partial derivative of y with respect to x, possibly multiple times.
    
    This function can handle multiple partial derivatives based on the 'order_x_partial' tuple, 
    which indicates how many times we differentiate w.r.t. each component of x.

    For example:
    - order_x_partial = (1,) means first derivative w.r.t x[0] once.
    - order_x_partial = (2,1) means second derivative w.r.t x[0] and first derivative w.r.t x[1].

    The code uses caching (_cache) to store intermediate partial derivatives. This makes repeated 
    or higher-order differentiation more efficient.

    Args:
        x: torch.Tensor or a tuple of torch.Tensors - The inputs to the network/operation.
        y: torch.Tensor - The output from which we will differentiate.
        component (int): Which component (index) of y to differentiate w.r.t if y is multi-dimensional.
        indx: Optional index if x is a tuple of tensors. Indicates which tensor in the tuple to differentiate w.r.t.
        order_x_partial (tuple): A tuple of integers indicating the number of times to differentiate 
                                 w.r.t. each dimension of x.

    Returns:
        ys: The computed partial derivative of the specified order.
    """
    # Create a key for caching
    key = (x, y, component, indx, order_x_partial)
    
    # If we've computed this exact partial derivative before, return cached result.
    # When we store partial derivatives, we store them in a multi-dimensional fashion.
    # The code below tries to extract the relevant slice for the particular order_x_partial.
    if key in _cache:
        for i in range(len(order_x_partial)):
            if order_x_partial[i] > 0:
                return _cache[key][..., i:i+1]

    # If not found in cache, we compute step-by-step.
    # Initialize ys as the output component of y we are differentiating.
    ys = y[..., component:component+1]
    
    # order_x_tmp will store intermediate states of differentiation order as we build up.
    order_x_tmp = [0]*len(order_x_partial)

    # For each dimension 'i', we differentiate 'num' times w.r.t that dimension.
    for i, num in enumerate(order_x_partial):
        if num == 0:
            # If no differentiation is required along this dimension, skip.
            continue
        
        # Perform 'num' successive differentiations along the i-th dimension.
        for _ in range(num):
            order_x_tmp[i] += 1
            current_order = tuple(order_x_tmp)
            key_current = (x, y, component, indx, current_order)

            # Try to find a neighbour from which we can deduce the current partial derivative.
            found = False
            for iminus, iplus, order_x_neighbour in getNeighbourA(current_order):
                key_neighbour = (x, y, component, indx, order_x_neighbour)
                if key_neighbour in _cache:
                    # If we found a neighbour from which we can derive this derivative,
                    # use it to avoid redundant computation.
                    ys = _cache[key_neighbour][..., iminus:iminus+1]
                    found = True
                    break
            
            # If we couldn't find it in the cache through neighbours, we must explicitly compute it.
            if not found:
                # Differentiate using PyTorch's autograd.
                if indx is None:
                    # Differentiate w.r.t. the entire x if x is a single tensor.
                    ys_xall = torch.autograd.grad(
                        ys, x, grad_outputs=torch.ones_like(ys),
                        create_graph=True
                    )[0]
                else:
                    # Differentiate w.r.t. a specific tensor in a tuple x if indx is provided.
                    ys_xall = torch.autograd.grad(
                        ys, x[indx], grad_outputs=torch.ones_like(ys),
                        create_graph=True
                    )[0]

                # Cache this new partial derivative result.
                _cache[key_current] = ys_xall
                # Extract the relevant component of ys_xall after differentiation.
                ys = ys_xall[..., i:i+1]

    return ys

def clearCache():
    """
    Clear the global cache of computed partial derivatives.
    This can be useful if you change input data or models and want to ensure no stale
    cached results are used.
    """
    _cache.clear()

# Attach the clearCache function to partialDerivative for convenience.
partialDerivative.clearCache = clearCache

def getOutput(y):
    """
    A simple utility function that just returns y.
    Might be a placeholder for a more complex post-processing step.
    """
    return y
