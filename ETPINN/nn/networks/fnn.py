import torch

from .nn import NN, identityNN
from .config import get
from ... import config
from .feature import FourierFeature


class FNN(NN):
    """
    Fully-connected Neural Network (FNN) class that acts as a wrapper for either:
    - FNNwithEncoder: If `useEncoder` is True and there are more than two layers.
    - FNNwithoutEncoder: Otherwise.

    This class determines which internal architecture to use based on the provided `Architecture` dictionary.
    It sets up the network, initializes parameters, and defines the forward pass through `_input_transform`
    and `_output_transform` hooks.
    """

    def __init__(self, Architecture=None, OldNetfile=None):
        super(FNN, self).__init__(Architecture=Architecture, OldNetfile=OldNetfile)
        self.name = "FNN"
        self.setup()

    def setup(self):
        """
        Set up the FNN by:
        1. Validating required keys in the Architecture (e.g., activation, layer_sizes, features).
        2. Determining which FNN variant (with or without an encoder) to use.
        3. Initializing the internal FNN model and parameters.
        
        If the encoder is used and the network is deep (more than two layers), it constructs FNNwithEncoder,
        otherwise, it constructs FNNwithoutEncoder.
        """
        self.checkArchitecture(activation=False, initializer=False, layer_sizes=False,
                               useEncoder=False, features=True)

        useEncoder = self.Architecture["useEncoder"]

        # Decide which sub-class of FNN to instantiate based on useEncoder and layer sizes
        if useEncoder and len(self.Architecture["layer_sizes"]) > 2:
            self.FNN = FNNwithEncoder(name="FNNwithEncoder", Architecture=self.Architecture)
        else:
            self.FNN = FNNwithoutEncoder(name="FNNwithoutEncoder", Architecture=self.Architecture)

        # Move the internal FNN to the default device (CPU or GPU depending on configuration)
        self.FNN = self.FNN.to(torch.defaultDevice)

        # Link init_params to the internal FNN's init_params for convenience
        self.init_params = self.FNN.init_params

        # Input/Output dimensions
        self.inDim = self.Architecture["layer_sizes"][0]
        # If using Fourier features, the effective input dimension might change
        if hasattr(self.FNN.feature, "features"):
            self.inDim = self.FNN.feature.features.shape[1]
        self.outDim = self.Architecture["layer_sizes"][-1]

        # Initialize parameters after setting up the FNN
        self.init_params()

    def init_params(self):   
        # This method is overridden to ensure parameters are re-initialized if needed.
        self.FNN.init_params()

    def forward(self, x, shift=0):
        """
        Forward pass of the FNN:
        1. Apply input transform (e.g., normalization)
        2. Pass through the internal FNN (with or without encoder)
        3. Apply output transform (e.g., inverse normalization)

        Args:
            x (torch.Tensor): Input tensor.
            shift (int or float, optional): A shift for Fourier features if used.

        Returns:
            torch.Tensor: The network output.
        """
        y = self._input_transform(x)
        y = self.FNN(y, shift)
        y = self._output_transform(y, x)
        return y


class FNNwithoutEncoder(NN):
    """
    A simple fully-connected neural network (FNN) without any encoder stage.

    This class builds a standard feed-forward network specified by `layer_sizes` and applies 
    a chosen activation function between layers. It can also integrate Fourier features at the input.
    """

    def __init__(self, name="FNN", Architecture=None, OldNetfile=None):
        super(FNNwithoutEncoder, self).__init__(Architecture=Architecture, OldNetfile=OldNetfile)
        self.name = name
        self.setup()

    def setup(self):
        """
        Set up the FNN by:
        - Checking required keys in the Architecture.
        - Determining the activation function and layer sizes.
        - Building a sequence of linear layers, possibly ending with identity if only one layer is specified.
        - Adding optional Fourier features at the input if specified.
        """
        self.checkArchitecture(activation=False, initializer=False, layer_sizes=False,
                               features=True, useBias=True)
        self.activation = get('activation', self.Architecture["activation"])
        layer_sizes = self.Architecture["layer_sizes"]

        # Create linear layers
        self.linears = torch.nn.ModuleList()
        if len(layer_sizes) > 1:
            # Default to using bias on all layers except possibly the last one
            self.bias = [True] * (len(layer_sizes) - 1)
            if 'useBias' in self.Architecture and self.Architecture['useBias'] is False:
                self.bias[-1] = False
        for i in range(len(layer_sizes) - 1):
            self.linears.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=self.bias[i]))

        # If there's only one layer_size, use an identity mapping
        if len(layer_sizes) == 1:
            self.linears.append(identityNN())

        # Initialize feature as identity, but can be replaced with FourierFeature if specified
        self.feature = identityNN()
        if "features" in self.Architecture and self.Architecture["features"] is not None:
            self.feature = FourierFeature(self.Architecture["features"])
            # Fourier features double the dimension for each frequency added
            assert self.feature.features.shape[0]*2 == layer_sizes[0]

    def init_params(self):
        """
        Initialize network parameters using the specified initializer.
        Weights are initialized by the chosen initializer, biases are initialized to zero.
        """
        initializer = get('initializer', self.Architecture["initializer"])
        if initializer is None:
            return
        initializer_zero = get('initializer', "zeros")
        for net in self.linears:
            if net.weight is not None:
                initializer(net.weight)
            if net.bias is not None:
                initializer_zero(net.bias)

    def forward(self, x, shift=0):
        """
        Forward pass:
        1. Apply input transform.
        2. Apply Fourier features if used.
        3. Pass through each linear layer + activation (except the last layer).
        4. Apply output transform.

        Args:
            x (torch.Tensor): Input data.
            shift (int or float, optional): Shift for Fourier features.

        Returns:
            torch.Tensor: Output of the network.
        """
        y = self._input_transform(x)
        # Apply Fourier features or identity
        y = self.feature(y, shift)

        # Pass through all but the last layer with activation
        for linear in self.linears[:-1]:
            y = self.activation(linear(y))

        # Last layer without activation (or identity if single-layer network)
        y = self.linears[-1](y)

        # Apply output transformation
        y = self._output_transform(y, x)
        return y


class FNNwithEncoder(NN):
    """
    Fully-connected Neural Network with an encoder stage, as proposed in the paper 
    "Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks"
    https://doi.org/10.1137/20M1318043

    This class provides a more complex architecture:
    - It uses an encoder--like structure internally, represented by Znet layers that blend between 
      two learned mappings (U and V) at each stage.
    - Incorporates Fourier features if specified.
    - Uses a custom forward pass that combines two separate transformations (U and V nets) and a 
      gating-like mechanism (Znet) to produce the output.
    """

    def __init__(self, name="FNN_ENCODER", Architecture=None, OldNetfile=None):
        self.name = name
        super(FNNwithEncoder, self).__init__(Architecture=Architecture, OldNetfile=OldNetfile)
        self.setup()

    def setup(self):
        """
        Set up the FNNwithEncoder by:
        - Checking required architecture keys.
        - Defining activation function.
        - Building the Znet (a list of linear layers) that will blend between U and V transformations.
        - Building separate U and V networks that act as "encoder" transformations.
        - Adding Fourier features if specified.
        """
        self.checkArchitecture(activation=False, initializer=False, layer_sizes=False,
                               features=True, useBias=True)
        self.activation = get('activation', self.Architecture["activation"])
        
        layer_sizes = self.Architecture["layer_sizes"]
        assert len(layer_sizes) > 2, "FNNwithEncoder expects more than two layers."

        self.Znet = torch.nn.ModuleList()
        bias = [True]*(len(layer_sizes)-1)
        if 'useBias' in self.Architecture and self.Architecture['useBias'] is False:
            bias[-1] = False

        # Build the Znet layers
        for i in range(len(layer_sizes)-1):
            self.Znet.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=bias[i]))

        # U and V networks transform the input before blending via Znet
        self.Unet = torch.nn.Linear(layer_sizes[0], layer_sizes[1])
        self.Vnet = torch.nn.Linear(layer_sizes[0], layer_sizes[1])

        # Add features (Fourier) if specified
        self.addFeature()

    def addFeature(self):
        """
        Add Fourier features at the input stage if specified by the architecture.
        This modifies the input dimension and pre-processes the input before feeding into U and V nets.
        """
        self.feature = identityNN()
        self.feature_inDim = 0
        if "features" in self.Architecture and self.Architecture["features"] is not None:
            self.feature = FourierFeature(self.Architecture["features"])
            shape = self.feature.features.shape
            # Ensure the first part of the input can hold all Fourier features
            assert shape[0]*2 <= self.Architecture["layer_sizes"][0]
            self.feature_inDim = shape[1]

    def init_params(self):
        """
        Initialize parameters for Znet, Unet, and Vnet using the specified initializer.
        Weights are initialized by the chosen method, biases are set to zero.
        """
        initializer = get('initializer', self.Architecture["initializer"])
        if initializer is None:
            return
        initializer_zero = get('initializer', "zeros")
        for net in self.Znet + [self.Unet, self.Vnet]:
            initializer(net.weight)
            if net.bias is not None:
                initializer_zero(net.bias)

    def forward(self, x, shift=0):
        """
        Forward pass for FNNwithEncoder:
        1. Transform input with _input_transform.
        2. Split input into Fourier feature part and remaining part, process features if any.
        3. Compute U = activation(Unet(y)) and V = activation(Vnet(y)).
        4. For each Znet layer (except last), compute Z = activation(Znet(y)) and blend:
           y = Z * U + (1 - Z) * V
        5. Pass through the last Znet layer without blending.
        6. Apply _output_transform before returning.

        Args:
            x (torch.Tensor): Input data.
            shift (int or float, optional): Shift for Fourier features.

        Returns:
            torch.Tensor: Output of the network after encoding, blending, and decoding.
        """
        y0 = self._input_transform(x)

        # Process Fourier features (if any) from the first part of the input
        Nin = self.feature_inDim
        y_feature = self.feature(y0[..., :Nin], shift)
        y = torch.cat((y_feature, y0[..., Nin:]), dim=-1)

        # Compute U and V transformations
        U = self.activation(self.Unet(y))
        V = self.activation(self.Vnet(y))

        # Blend features using Znet layers
        # For all but the last Znet layer, blend U and V using a "gating" approach with Z.
        for linear in self.Znet[:-1]:
            Z = self.activation(linear(y))
            y = Z * U + (1 - Z) * V

        # Final Znet layer without blending
        y = self.Znet[-1](y)

        # Apply output transform
        y = self._output_transform(y, x)
        return y
