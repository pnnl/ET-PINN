import torch

# Dictionary mapping string keys to activation functions
ACTIVATION_DICT = {
    "relu": torch.relu,  # ReLU activation
    "selu": torch.selu,  # SELU activation
    "gelu": torch.nn.GELU(),  # GELU activation
    "sigmoid": torch.sigmoid,  # Sigmoid activation
    "sin": torch.sin,  # Sine activation
    "swish": lambda x: x * torch.sigmoid(x),  # Swish activation: x * sigmoid(x)
    "tanh": torch.tanh,  # Tanh activation
}

# Dictionary mapping string keys to weight initialization functions
INITIALIZER_DICT = {
    "default": None,  # No initialization (default)
    "glorot normal": torch.nn.init.xavier_normal_,  # Xavier normal initialization
    "glorot uniform": torch.nn.init.xavier_uniform_,  # Xavier uniform initialization
    "he normal": torch.nn.init.kaiming_normal_,  # He normal initialization
    "he uniform": torch.nn.init.kaiming_uniform_,  # He uniform initialization
    "zeros": torch.nn.init.zeros_,  # Initialize all weights to zero
}

def get(key, identifier):
    """
    Returns a function corresponding to the given key and identifier.
    
    Args:
        key (str): Specifies the type of function, either 'activation' or 'initializer'.
                   The string is case-insensitive.
        identifier (str or function): The name of the function as a string, or 
                                       the function itself.

    Returns:
        function: The corresponding activation or initializer function.

    Raises:
        Exception: If the identifier is not a valid key in the corresponding dictionary.
        TypeError: If the identifier is not a string or valid function.
    """
    # Ensure the key is either 'ACTIVATION' or 'INITIALIZER'
    assert key.upper() in ['ACTIVATION', 'INITIALIZER'], \
        "Key must be either 'ACTIVATION' or 'INITIALIZER'."
    
    # Select the appropriate dictionary based on the key
    DICT = ACTIVATION_DICT if key.upper() == 'ACTIVATION' else INITIALIZER_DICT

    # If the identifier is a string, look it up in the dictionary
    if isinstance(identifier, str):  
        # Check if the identifier is a valid key in the dictionary
        if identifier.lower() not in DICT:
            raise Exception(
                f"""Only support the following {key.lower()} functions:
                {", ".join(DICT.keys())}"""
            )
        # Return the corresponding function
        return DICT[identifier.lower()]

    # If the identifier is not a string, raise a TypeError
    raise TypeError(
        f"Could not interpret {key.lower()} function identifier: {identifier}"
    )
