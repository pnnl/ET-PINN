import torch

class FourierFeature(torch.nn.Module):
    """Fourier Feature Network Module.
    
    This module generates Fourier features for a given input by applying
    a linear transformation followed by trigonometric encoding. This is often
    used in machine learning models to enhance the expressive power for 
    high-frequency components. For more details, refer to the paper 
    "Fourier features let networks learn high frequency functions in
    low dimensional domains"
    
    """
    def __init__(self, features):
        """
        Initializes the FourierFeature module.
        
        Args:
            features (torch.Tensor): A 2D tensor defining the Fourier feature matrix.
                                      The matrix is used to linearly transform the input.
        """
        super().__init__()  # Initialize the base nn.Module class
        self.features = features.to(torch.tensor(0.).device)  # Ensure the feature matrix is on the same device as the tensor
        assert len(features.shape) == 2, "The feature matrix must be a 2D tensor."  # Check that `features` is a 2D tensor

    def forward(self, x, shift=0):
        """
        Forward pass to compute Fourier features.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).
            shift (float): Optional shift to apply to the features for phase adjustment.
        
        Returns:
            torch.Tensor: Output tensor with encoded Fourier features.
        """
        # Perform linear transformation of the input using the Fourier feature matrix
        phi = torch.matmul(x, self.features.T)
        
        phi = torch.cat((phi, phi - 0.5), dim=1)
        
        # Apply scaling and shifting (phase adjustment) to the transformed features
        phi = (phi + shift) * torch.pi
        
        # Apply the cosine function to encode the features with trigonometric transformation
        x = torch.cos(phi)
        
        return x
