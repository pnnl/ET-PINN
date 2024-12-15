import torch

def is_BFGS(optimizer):
    """
    Check if the optimizer name indicates a BFGS-based optimization method.
    
    Args:
        optimizer (str): The name of the optimizer.
        
    Returns:
        bool: True if the optimizer is BFGS-based, False otherwise.
    """
    return "bfgs" in optimizer.lower()

def get(params, optimizerOption):
    """
    Given a set of parameters and an optimizer configuration, this function returns 
    a PyTorch optimizer instance and an optional learning rate scheduler.
    
    Args:
        params (iterable): The parameters of the model that the optimizer will update.
        optimizerOption (dict): A dictionary containing the optimizer setup. 
            Keys:
                'name' (str): The name of the optimizer (e.g. "adam", "lbfgs").
                'lr' (float, optional): The learning rate for the optimizer. Required for most optimizers except LBFGS.
                'lr_decay' (callable, optional): A function/lambda for learning rate scheduling.
    
    Returns:
        (optimizer, lr_scheduler):
            optimizer: A PyTorch optimizer instance (e.g., torch.optim.Adam, torch.optim.LBFGS).
            lr_scheduler: A learning rate scheduler instance if 'lr_decay' is provided, otherwise None.
    
    Raises:
        ValueError: If the optimizer requires a learning rate and none is provided.
        NotImplementedError: If the requested optimizer or scheduler is not supported.
    """
    optimizer = optimizerOption['name']
    learning_rate = optimizerOption.get('lr', None)
    decay = optimizerOption.get('lr_decay', None)
    
    # If the provided optimizerOption already includes a PyTorch optimizer instance, just use it.
    if isinstance(optimizer, torch.optim.Optimizer):
        optim = optimizer
    elif is_BFGS(optimizer):
        # BFGS (LBFGS in PyTorch) generally doesn't need a custom learning rate or decay.
        # If specified, they are ignored, but a warning is printed.
        if learning_rate is not None or decay is not None:
            print(f"Warning: learning rate/decay is ignored for {optimizer}")
        optim = torch.optim.LBFGS(
            params,
            lr=1,
            history_size=20,
            tolerance_change=1e-12,
            tolerance_grad=1e-12,
            max_iter=20,
            max_eval=25,
            line_search_fn=None,  # or 'strong_wolfe' if needed
        )
    else:
        # For standard optimizers like Adam, a learning rate must be provided.
        if learning_rate is None:
            raise ValueError(f"No learning rate specified for {optimizer}.")
        elif optimizer.lower() == "adam":
            optim = torch.optim.Adam(params, lr=learning_rate)
        else:
            # Extend here if more optimizers are supported
            raise NotImplementedError(f"{optimizer} not implemented for PyTorch backend.")
    
    # Optionally, get a learning rate scheduler based on the 'lr_decay' function
    lr_scheduler = _get_learningrate_scheduler(optim, decay)
    return optim, lr_scheduler

def _get_learningrate_scheduler(optim, decay):
    """
    Creates and returns a learning rate scheduler if a decay function is provided.
    
    Args:
        optim (torch.optim.Optimizer): The optimizer to which the scheduler will be applied.
        decay (callable or None): If callable, it should accept an integer parameter (epoch) and return a multiplier 
                                  for the learning rate.
                                  
    Returns:
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler or None): Returns a scheduler instance 
        if 'decay' is provided and callable, otherwise None.
    
    Raises:
        NotImplementedError: If 'decay' is not None and not callable.
    """
    if decay is None:
        return None
    if callable(decay):
        # LambdaLR applies a user-defined lambda to each epoch for modifying the learning rate.
        return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=decay)
    else:
        raise NotImplementedError(f"Unknown learning rate scheduler: {decay}")
