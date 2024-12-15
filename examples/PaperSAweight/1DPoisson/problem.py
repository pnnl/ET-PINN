# -*- coding: utf-8 -*-

import os
import ETPINN
grad = ETPINN.grad
import matplotlib.pyplot as plt
import numpy as np
import torch

# This code sets up a one-dimensional Poisson equation problem
# u_xx = f(x)
# where u = u(x), f(x) is a forcing term or source function

Nin = 1   # Input dimension: x
Nout = 1  # Output dimension: u(x)
path = "numsols"
L = 1     # Domain length 

class problem():
    def __init__(self, tlen=1, ksi=10):
        """
        Initialize the 1D problem domain and parameters. The exact solution is u(x)=sin(2*ksi*pi*x^2).

        Args:
            tlen (float): Not used directly here (set for consistency with other problems).
            ksi (float): A parameter that defines the frequency of the chosen exact solution.                         
        """
        lb, ub = [0], [1]
        self.ksi = ksi
        # Create a 1D interval domain using ETPINN's geometry tools
        self.physicalDomain = ETPINN.geometry.Interval(lb[0], ub[0])
        
        # Parameters for input and output normalization
        self.input_params = {"lb":[lb], "ub":[ub]}
        self.output_params = {"std":1, "mean":0}
        self.lb, self.ub = lb, ub

    def input_transform(self, xt, lb, ub):
        """
        Normalize the input x from [lb, ub] to [-1,1].

        Args:
            xt (torch.Tensor): Input coordinates (N,1).
            lb, ub (list): Lower and upper bounds for the domain.
        
        Returns:
            xt (torch.Tensor): Normalized input.
        """
        xt = (xt-(ub+lb)/2)/(ub-lb)*2
        return xt

    def output_transform(self, y, xt, std, mean):
        """
        Denormalize the output from the NN.

        Args:
            y (torch.Tensor): Normalized NN output.
            xt (torch.Tensor): Input coordinates (not used here).
            std, mean (float): Standard deviation and mean for normalization.
        
        Returns:
            torch.Tensor: Denormalized output.
        """
        return y*std+mean

    def solve(self, Nx=101, noise=0):
        """
        Produce a reference "exact" solution from the chosen exact form:
        u(x) = sin(2*ksi*pi*x^2)

        Args:
            Nx (int): Number of spatial points.
            noise (float): Amount of noise to add to the solution if needed.
        
        Returns:
            x (np.array): Spatial coordinates.
            u (np.array): Exact solution values at those coordinates.
        """
        x = np.linspace(self.lb[0], self.ub[0], Nx)[:,None]
        u = np.sin(2*self.ksi*np.pi*x*x)
        return x, u

    def pde(self, x, y, aux, output):
        """
        Define the PDE residual for the chosen problem.

        Given the exact solution u(x)=sin(2*ksi*pi*x^2), 
        Residual form:
            res = u_xx + 4*k^2*x^2*sin(k*x^2) - 2*k*cos(k*x^2)
        where k = 2*ksi*pi.
        """
        u = grad.getOutput(output)
        u_xx = grad.partialDerivative(x, output, component=0, order_x_partial=(2,))
        k = 2*self.ksi*torch.pi

        # Derived PDE residual for the chosen exact solution form
        res = u_xx + 4*k**2*x*x*torch.sin(k*x*x) - 2*k * torch.cos(k*x*x)
        return res, {"res": torch.mean(torch.square(res))}

    def BC(self, x, y, aux, output):
        """
        Boundary condition u=0 
        """
        yp = grad.getOutput(output)
        diff = yp
        return diff, {"BC": torch.mean(torch.square(diff))}

    def testError(self, x, y, aux, output):
        """
        Compute a relative test error between predicted and exact solutions.

        Args:
            x (torch.Tensor): Input points.
            y (torch.Tensor): Target values.
            aux: Auxiliary info (not used).
            output (torch.Tensor): NN predictions.

        Returns:
            {'test': relative error}
        """
        yp = grad.getOutput(output)
        err = torch.norm(yp - y)/torch.norm(y)
        return {'test': err}
