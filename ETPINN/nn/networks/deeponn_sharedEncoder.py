# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:14:41 2024

@author: chen096
"""

from .nn import NN, identityNN  
import torch
from .config import get  
from .feature import FourierFeature  

class DEEPONN_sharedEncoder(NN):
    """
    Deep Operator Network with a Shared Encoder.
    
    This variant of DeepONet uses shared encoding layers to process inputs
    into a latent representation, which is then used by both trunk and branch 
    networks for operator learning.
    For more details, refer to the paper 
    "Improved architectures and training algorithms for deep operator networks"
    by Wang, Sifan and Wang, Hanwen and Perdikaris, Paris.
    
    """
    def __init__(self, name="DeepOnet_sharedEncoder", Architecture=None, OldNetfile=None):
        """
        Initializes the DeepONN_sharedEncoder architecture.
        
        Args:
            name (str): Name of the network.
            Architecture (dict): A dictionary defining the architecture parameters.
            OldNetfile (str): (Optional) Path to a pre-trained model file.
        """
        self.name = name  # Assign the name
        super(DEEPONN_sharedEncoder, self).__init__(Architecture=Architecture, OldNetfile=OldNetfile)
        self.setup()  # Setup the network components

    def setup(self):
        """
        Configures the architecture and initializes network components.
        
        - Checks the architecture validity.
        - Initializes trunk and branch shared encoders and layers.
        - Sets up activation functions, Fourier features, and bias.
        """
        # Check and validate the architecture
        self.checkArchitecture(activation=False, initializer=False,
                               layer_sizes_trunk=False,
                               layer_sizes_brunch=False,
                               Nout=False,
                               features_trunk=True,
                               features_brunch=True)

        self.activation = get('activation', self.Architecture["activation"])  # Activation function
        self.outDim = self.Architecture['Nout']  # Output dimension

        # Retrieve trunk and branch layer sizes
        layer_sizes_trunk = self.Architecture["layer_sizes_trunk"]
        layer_sizes_brunch = self.Architecture["layer_sizes_brunch"]

        # Ensure layer sizes meet minimum requirements and match between trunk and branch
        assert len(layer_sizes_trunk) > 2 and len(layer_sizes_brunch) > 2
        assert layer_sizes_brunch[1:] == layer_sizes_trunk[1:]
        assert layer_sizes_brunch[-1] % self.outDim == 0

        # Initialize trunk and branch shared encoder layers
        self.Znet_trunk = torch.nn.ModuleList()
        self.Znet_brunch = torch.nn.ModuleList()
        for i in range(len(layer_sizes_trunk) - 1):
            self.Znet_trunk.append(torch.nn.Linear(layer_sizes_trunk[i], layer_sizes_trunk[i+1]))
            self.Znet_brunch.append(torch.nn.Linear(layer_sizes_brunch[i], layer_sizes_brunch[i+1]))

        # Shared encoders for the trunk and branch inputs
        self.Unet = torch.nn.Linear(layer_sizes_trunk[0], layer_sizes_trunk[1])
        self.Vnet = torch.nn.Linear(layer_sizes_brunch[0], layer_sizes_brunch[1])

        # Feature encoders for trunk and branch inputs
        self.feature_trunk = identityNN()  # Default identity mapping
        self.feature_brunch = identityNN()

        # Add Fourier features if specified in the architecture
        if "features_trunk" in self.Architecture and self.Architecture["features_trunk"] is not None:
            self.feature_trunk = FourierFeature(self.Architecture["features_trunk"])
            assert self.feature_trunk.features.shape[0] * 2 == layer_sizes_trunk[0]
        if "features_brunch" in self.Architecture and self.Architecture["features_brunch"] is not None:
            self.feature_brunch = FourierFeature(self.Architecture["features_brunch"])
            assert self.feature_brunch.features.shape[0] * 2 == layer_sizes_brunch[0]

        # Trainable bias parameter
        self.bias = torch.nn.Parameter(torch.zeros((1, self.outDim), dtype=torch.float32))

        # Initialize network parameters
        self.init_params()

    def init_params(self):
        """
        Initializes the parameters of the network layers using specified initializers.
        """
        initializer = get('initializer', self.Architecture["initializer"])
        if initializer is None:
            return
        initializer_zero = get('initializer', "zeros")
        # Apply initializers to weights and biases
        for net in self.Znet_trunk + self.Znet_brunch + [self.Unet, self.Vnet]:
            initializer(net.weight)
            if net.bias is not None:
                initializer_zero(net.bias)

    def forward(self, x, shift=0):
        """
        Forward pass for DEEPONN_sharedEncoder.
        
        Args:
            x (torch.Tensor): Input tensor containing both trunk and branch inputs.
            shift (float): Phase shift applied to Fourier features (optional).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, outDim).
        """
        # Transform the input if needed
        y = self._input_transform(x)
        y_trunk, y_brunch = y

        # Apply Fourier feature transformation to trunk and branch inputs
        y_brunch = self.feature_brunch(y_brunch, shift)
        y_trunk = self.feature_trunk(y_trunk, shift)

        # Compute shared encodings using U and V networks
        U = self.activation(self.Unet(y_trunk))
        V = self.activation(self.Vnet(y_brunch))

        # Iterate through trunk and branch layers, combining using U and V
        for linear_trunk, linear_brunch in zip(self.Znet_trunk[:-1], self.Znet_brunch[:-1]):
            Z_trunk = self.activation(linear_trunk(y_trunk))
            Z_brunch = self.activation(linear_brunch(y_brunch))
            y_trunk = Z_trunk * U + (1 - Z_trunk) * V
            y_brunch = Z_brunch * U + (1 - Z_brunch) * V

        # Process final layers for trunk and branch
        y_trunk = self.Znet_trunk[-1](y_trunk)
        y_brunch = self.Znet_brunch[-1](y_brunch)

        # Reshape outputs and compute the final output
        y_trunk = y_trunk.reshape(y_trunk.shape[0], -1, self.outDim)
        y_brunch = y_brunch.reshape(y_brunch.shape[0], -1, self.outDim)
        y = torch.einsum("kib,kib->kb", y_trunk, y_brunch)

        # Apply output transformation
        y = self._output_transform(y, x)
        return y
