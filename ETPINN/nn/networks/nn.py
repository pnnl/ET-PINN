import torch
import dill as pickle

# By default, we use torch.defaultDevice as the device, 
device = torch.defaultDevice
class NN(torch.nn.Module):
    """
    Base class for neural network modules. This class provides:
    - Loading and saving network architectures and parameters
    - Input and output transformation hooks
    - Methods to count trainable parameters
    - Basic architecture validation checks

    Args:
        Architecture (dict, optional): A dictionary specifying the network's architecture parameters.
        OldNetfile (str, optional): A path to a saved state dictionary (from .savenet()).
                                    The Architecture will be loaded from the file if provided.
    Raises:
        Exception: If neither `Architecture` nor `OldNetfile` is provided.
    """

    def __init__(self, Architecture=None, OldNetfile=None):
        super().__init__()

        # Initialize input/output transforms as no-ops.
        self._input_fun = lambda x: x
        self._input_params = {}
        self._input_transform = self._input_fun
        self._output_fun = lambda y, x: y
        self._output_params = {}
        self._output_transform = self._output_fun

        # Require at least one: Architecture dict or an existing network file.
        if not (Architecture or OldNetfile):
            raise Exception('At least one of "Architecture" or "OldNetfile" should be provided.')

        if Architecture is not None:
            self.Architecture = Architecture
        else:
            # Load architecture from an old network file
            self.Architecture = torch.load(OldNetfile, map_location=lambda storage, loc: storage)['Architecture']

    def checkArchitecture(self, **kargs):
        """
        Check if the given keys exist in the current Architecture.

        Args:
            **kargs: key-value pairs where the value is a boolean indicating if the key is optional.
                     If optional is False and the key is missing, an error is raised.

        Raises:
            Exception: If required keys are missing from the Architecture.
        """
        valid = True
        not_included = []
        for arg, optional in kargs.items():
            if arg not in self.Architecture and not optional:
                valid = False
                not_included.append(arg)

        if not valid:
            raise Exception(
                f"Architecture for {self.__class__.__name__} should include keys: "
                f"{', '.join(kargs.keys())}. Missing keys: {', '.join(not_included)}."
            )

    def apply_input_transform(self, transform, params):
        """
        Set a transform function for inputs. This transform is applied before the data is fed into the network.

        Args:
            transform (callable): A function that takes inputs (x, **params) and returns transformed inputs.
            params (dict): Additional parameters for the transform function.
        """
        # Convert parameters to tensors on the appropriate device
        self._input_params = {
            k: (torch.as_tensor(p).to(device) if not isinstance(p, torch.Tensor) else p.to(device))
            for k, p in params.items()
        }
        self._input_fun = transform
        self._input_transform = lambda x: transform(x, **self._input_params)

    def apply_output_transform(self, transform, params):
        """
        Set a transform function for outputs. This transform is applied after the network produces its output.

        Args:
            transform (callable): A function that takes (y, x, **params) where y is the output and x the input.
            params (dict): Additional parameters for the transform function.
        """
        self._output_params = {
            k: (torch.as_tensor(p).to(device) if not isinstance(p, torch.Tensor) else p.to(device))
            for k, p in params.items()
        }
        self._output_fun = transform
        self._output_transform = lambda y, x: transform(y, x, **self._output_params)

    def num_trainable_parameters(self):
        """
        Calculate the total number of trainable parameters in the network.

        Returns:
            int: Number of trainable parameters.
        """
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

    def loadnet(self, OldNetfile):
        """
        Load a previously saved network state from a file.

        Args:
            OldNetfile (str): Path to the file containing the saved network state.

        The file should contain:
        - 'Architecture': The network's architecture.
        - 'state_dict': The model's state dictionary.
        - 'input_transform': A tuple (input_transform_function, input_transform_params).
        - 'output_transform': A tuple (output_transform_function, output_transform_params).
        """
        net = torch.load(OldNetfile, map_location=lambda storage, loc: storage, pickle_module=pickle)

        # Update the network architecture and re-setup the network
        self.Architecture = net['Architecture']
        self.setup()  # This method is expected to be defined in subclasses to build the model

        # Load state dictionary (weights)
        self.load_state_dict(net['state_dict'])

        # Re-apply transforms
        self.apply_input_transform(*net['input_transform'])
        self.apply_output_transform(*net['output_transform'])

    def savenet(self, netfile):
        """
        Save the current network state to a file.

        Args:
            netfile (str): The path to the file where the network should be saved.
        """
        torch.save({
            'Architecture': self.Architecture,
            'state_dict': self.state_dict(),
            'input_transform': (self._input_fun, self._input_params),
            'output_transform': (self._output_fun, self._output_params),
        }, netfile, pickle_module=pickle)


class identityNN(torch.nn.Module):
    """
    A simple identity network that returns the input as-is.

    This can be useful as a placeholder network or for debugging.
    """
    def __init__(self):
        super().__init__()
        self.weight = None
        self.bias = None

    def forward(self, x, shift=0):
        """
        Forward pass of the identityNN. Just returns the input without modification

        Args:
            x (torch.Tensor): Input tensor.
            shift (int or float, optional): A shift value that could be added if desired. Currently unused.
        Returns:
            torch.Tensor: The same input tensor.
        """
        return x
