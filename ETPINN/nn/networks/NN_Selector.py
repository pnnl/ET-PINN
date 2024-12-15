from .fnn import FNN
from .deeponn import DEEPONN

# A dictionary mapping network types (strings) to their corresponding classes.
# This allows for flexible construction of neural networks based on a given Architecture dict.
NNdict = {
    "FNN": FNN,
    "DEEPONN": DEEPONN
}

def NNBuilder(Architecture):
    """
    Build a neural network instance based on the provided Architecture dictionary.

    Args:
        Architecture (dict): A dictionary containing the required configuration keys.
                             Must include 'NNType' to specify which network class to use.

    Returns:
        An instance of the neural network class specified in Architecture['NNType'].

    Raises:
        AssertionError: If 'NNType' is not in Architecture or is not a valid key in NNdict.
    """
    # Ensure the Architecture dictionary specifies the neural network type.
    assert 'NNType' in Architecture, "Architecture dictionary must contain 'NNType'."
    
    # Retrieve the network class name.
    NNclass = Architecture['NNType']
    
    # Confirm that the specified NN class is known and available.
    assert NNclass in NNdict, f"Unknown NNType '{NNclass}'. Must be one of {list(NNdict.keys())}."
    
    # Instantiate the network using the corresponding class from NNdict.
    NN = NNdict[NNclass](Architecture=Architecture)
    
    return NN
