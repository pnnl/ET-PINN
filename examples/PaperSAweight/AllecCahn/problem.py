# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import ETPINN
grad = ETPINN.grad
import torch

# This code sets up the Allen–Cahn problem
# The Allen–Cahn equation is given by:
#     u_t = c1^2 * u_xx - c2 * (u^3 - u)
#
# Here, c1 and c2 are parameters that affect the equation's behavior.
c1, c2 = 1E-2, 5

Nin = 2   # Number of input dimensions: (x,t)
Nout = 1  # Number of output dimensions: (u)
path = "numsols"  # Path to stored solution data
L = 1     # Spatial domain half-length, x ∈ [-L, L]

class problem():
    def __init__(self, tlen=1):
        """
        Initialize the Allen–Cahn problem.

        Args:
            tlen (float): The final time up to which the PDE is solved.
        """
        lb, ub = [-L,0], [L,tlen]
        self.tlen = tlen

        # Define geometry of the problem domain using ETPINN:
        self.spatialDomain = ETPINN.geometry.Interval(lb[0], ub[0])
        self.timeDomain = ETPINN.geometry.TimeDomain(lb[1], ub[1])
        self.physicalDomain = ETPINN.geometry.GeometryXTime(self.spatialDomain, self.timeDomain)

        # Default resolution for solutions
        self.Nx_exact, self.Nt_exact = 512, 201
        self.lb, self.ub = lb, ub

        # Number of Fourier modes for input feature mapping
        self.nMode = 10
        self.xModes = torch.arange(1, self.nMode+1)[None,:]

        # Input and output transformation parameters
        self.input_params = {"lb": [lb], "ub": [ub], "xModes": self.xModes}
        self.output_params = {"std": 1, "mean": 0}

    def input_transform(self, xt, lb, ub, xModes):
        """
        Transform input (x,t) by normalizing it and adding Fourier features.

        Args:
            xt (torch.Tensor): Input points shape (N,2) with columns [x, t].
            lb, ub (list): Lower and upper bounds for normalization.
            xModes (torch.Tensor): The Fourier mode wave numbers.

        Returns:
            xt (torch.Tensor): The transformed input with sine/cosine Fourier terms and t.
        """
        # Normalize input domain from [lb, ub] to [-1,1]
        xt = (xt-(ub+lb)/2)/(ub-lb)*2
        # Add Fourier features for the x-coordinate
        xFourier = torch.pi * xt[:,0:1]*xModes
        # Concatenate sin and cos transforms of x plus the original (normalized) time t dimension.
        xt = torch.cat((torch.sin(xFourier), torch.cos(xFourier), xt[:,1:2]), dim=1)
        return xt

    def output_transform(self, y, xt, std, mean):
        """
        Denormalize output from the network.

        Args:
            y (torch.Tensor): The normalized output of the NN.
            xt (torch.Tensor): Input points (not used directly here).
            std, mean (float): Statistics for de-normalization.

        Returns:
            torch.Tensor: Denormalized output.
        """
        return y*std + mean

    def initFun(self, xt):
        """
        Initial condition function: u(x,0) = x^2 * cos(pi*x).

        Args:
            xt (np.array): Input coordinates with shape (N,2).
                           Only the first column (x) is used since this is at t=0.

        Returns:
            np.array: The initial condition values at the given x.
        """
        x = xt[:,0:1]
        return x**2 * np.cos(np.pi*x)

    def solve(self, Nx=None, Nt=None, noise=0, strip_x=1, strip_t=1):
        """
        Load a precomputed numerical solution of the Allen–Cahn equation from a .mat file.

        The file should contain 'x', 'tt' (time), and 'uu' (solution) arrays.

        Args:
            Nx (int): Number of spatial points. Defaults to Nx_exact if None.
            Nt (int): Number of time points. Defaults to Nt_exact if None.
            noise (float): Amount of noise to add to the solution for testing robustness.
            strip_x (int): Downsampling factor in the spatial direction.
            strip_t (int): Downsampling factor in the time direction.

        Returns:
            xt (np.array): Flattened space-time coordinates.
            u (np.array): Corresponding solution values u(x,t).
            (x,t,u2D) (tuple): x grid, t grid, and 2D solution array u(t,x).
        """
        if Nx is None or Nt is None:
            Nx, Nt = self.Nx_exact, self.Nt_exact

        filename = os.path.join(path, f"AC_{Nx}_{Nt}.mat")
        data = loadmat(filename)

        # Extract and downsample data
        x = data['x'].squeeze()[::strip_x]
        t = data['tt'].squeeze()[::strip_t]
        u2D = data['uu'].T[::strip_t, ::strip_x]  # Transpose to get shape (time, space)

        print('resdata', data['uu'].shape)

        # Add optional noise
        u2D += noise*np.random.randn(*u2D.shape)

        # Truncate time dimension to final time tlen
        nChoose = np.sum(t <= self.tlen)
        t = t[:nChoose]
        u2D = u2D[:nChoose,:]

        # Create flattened coordinate array
        xt = np.stack(np.meshgrid(x,t), axis=0).reshape((2,-1)).T
        u = u2D.reshape(1,-1).T

        return xt, u, (x,t,u2D)

    def pde(self, x, y, aux, output):
        """
        PDE residual for the Allen–Cahn equation:
            u_t = c1^2 * u_xx - c2*(u^3 - u)

        Args:
            x (torch.Tensor): Input points (x,t).
            y (torch.Tensor): Taget values, not used here directly, since PDE doesn't need labeled data.
            aux: Auxiliary data (not used here).
            output (torch.Tensor): NN output representing u(x,t).

        Returns:
            (res, dict): Residual tensor and a dictionary with 'res' mean square for metrics.
        """
        uv = ETPINN.grad.getOutput(output)
        u  = uv[:,0:1]

        # Compute derivatives
        u_t  = grad.partialDerivative(x, output, component=0, order_x_partial=(0,1))  # du/dt
        u_xx = grad.partialDerivative(x, output, component=0, order_x_partial=(2,0))  # d²u/dx²

        res = u_t - c1**2 * u_xx + c2*(u**3 - u)
        return res, {"res": torch.mean(torch.square(res))}

    def dataFit(self, x, y, aux, output, name='data'):
        """
        Data fitting loss: encourages output to match provided data y.

        Args:
            x (torch.Tensor): Input points.
            y (torch.Tensor): Target data (e.g., from a known solution).
            aux: Not used.
            output (torch.Tensor): NN predictions.

        Returns:
            diff, {name: mean squared error (MSE)}.
        """
        yp = ETPINN.grad.getOutput(output)
        diff = yp - y
        return diff, {name: torch.mean(torch.square(diff))}

    def IC(self, x, y, aux, output):
        """
        Initial condition loss: ensures the solution at t=0 matches the known initial condition.

        Args:
            x,y,aux,output: standard ETPINN function arguments.

        Returns:
            Resdiual and MSE under the label 'IC'.
        """
        yp = grad.getOutput(output)
        diff = yp - y
        return diff, {'IC': torch.mean(torch.square(diff))}

    def BC(self, x, y, aux, output):
        """
        Boundary condition loss: ensures solution meets boundary conditions u=0 at x boundaries.
        Args:
            x,y,aux,output: standard PINN function arguments.

        Returns:
            Residual and MSE under 'BC'.
        """
        yp = grad.getOutput(output)
        diff = yp
        return diff, {"BC": torch.mean(torch.square(diff))}

    def testError(self, x, y, aux, output):
        """
        Compute a relative test error for evaluation:
        error = ||yp - y|| / ||y||

        Args:
            x,y,aux,output: standard PINN function arguments.

        Returns:
            {'test': relative error}
        """
        yp = ETPINN.grad.getOutput(output)
        err  = torch.norm(yp-y)/torch.norm(y)
        return {'test': err}

