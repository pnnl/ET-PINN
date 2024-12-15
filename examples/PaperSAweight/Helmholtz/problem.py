# -*- coding: utf-8 -*-

import os
import ETPINN
grad = ETPINN.grad
import matplotlib.pyplot as plt
import numpy as np
import torch

# This code sets up a physics-informed problem for the 2D Helmholtz equation reads:
#
#    Δu + k² u = f
#
# where u = u(x,y), Δ denotes the Laplacian (u_xx + u_yy), and k is a wavenumber parameter.
# The term f(x,y) is a forcing term or source function.

Nin = 2   # Input dimension: (x,y)
Nout = 1  # Output dimension: (u)
path = "numsols"

# Parameters defining the exact solution and forcing:
a1 = 1 
a2 = 4
k  = 1

def fsource(x,y):
    """
    Compute the forcing term of the PDE based on the exact solution:
    u_exact(x,y) = sin(a1*pi*x)*sin(a2*pi*y)
    """
    f = -(a1*np.pi)**2 - (a2*np.pi)**2 + k**2
    f *= np.sin(a1*np.pi*x)*np.sin(a2*np.pi*y)
    return f

class problem():
    def __init__(self, nMode=5):
        """
        Initialize the Helmholtz equation.

        Args:
            nMode (int): The number of Fourier modes in each direction.
        """        
        # Domain:
        lb, ub = [-1,-1], [1,1]
        self.physicalDomain = ETPINN.geometry.Rectangle(np.array(lb), np.array(ub))
        
        # For Fourier feature transformation:
        xyModes = np.stack(np.meshgrid(np.arange(1,nMode+1),
                                       np.arange(1,nMode+1)),
                           axis=-1).reshape(-1,2)
        
        self.input_params = {"lb":[lb], "ub":[ub], "xyModes":xyModes}
        self.output_params = {"std":1, "mean":0}
        self.lb, self.ub = lb, ub

    def input_transform_Fourier(self, xt, lb, ub, xyModes):
        """
        Apply Fourier feature encoding on input (x,y):
        1. Normalize (x,y) to [-1,1].
        2. Compute sin and cos terms with multiple modes.

        Args:
            xt (torch.Tensor): Input coordinates (N,2)
            lb, ub (list): Bounds of the domain
            xyModes (np.array): Array of mode pairs for Fourier features

        Returns:
            torch.Tensor: Transformed input features with various sin/cos combinations.
        """
        xy = (xt-(ub+lb)/2)/(ub-lb)*2
        x = torch.pi * xy[:,0:1]*xyModes[:,0:1].T
        y = torch.pi * xy[:,1:2]*xyModes[:,1:2].T

        # Construct features combining sin and cos of these modes
        # Produces a rich set of frequency-based features
        xy = torch.cat((torch.sin(x)*torch.cos(y),  
                        torch.cos(x)*torch.cos(y),
                        torch.cos(x)*torch.sin(y)),
                       dim=1)
        
        return xy

    def input_transform(self, xt, lb, ub, xyModes):
        """
        Simple normalization of input (x,y) to [-1,1].
        If not using Fourier features, this just returns normalized coordinates.

        Args:
            xt: (N,2) input array
            lb, ub: lower and upper bounds
            xyModes: modes (not used in this simple transform)

        Returns:
            xy: normalized input
        """
        xy = (xt-(ub+lb)/2)/(ub-lb)*2
        return xy

    def output_transform(self, y, xt, std, mean):
        """
        Denormalize the network's output if it was normalized.

        Args:
            y: normalized output
            xt: input (not used)
            std, mean: normalization parameters

        Returns:
            y: denormalized output
        """
        return y*std+mean

    def Exact(self, xy):
        """
        Compute exact solution:
        u_exact(x,y) = sin(a1*pi*x)*sin(a2*pi*y)

        Args:
            xy: shape (...,2) array of coordinates

        Returns:
            u: exact solution array
        """
        x,y = xy[...,0:1], xy[...,1:2]
        u = np.sin(a1*np.pi*x)*np.sin(a2*np.pi*y)
        return u

    def fsource(self, xy):
        """
        Forcing function f(x,y) corresponding to the chosen exact solution.
        Derived from Δu + k² u = f.

        Args:
            xy: coordinates array (...,2)

        Returns:
            f: forcing function values at xy
        """
        x,y = xy[...,0:1], xy[...,1:2]
        u = -(a1*np.pi)**2-(a2*np.pi)**2 + k**2
        u *= np.sin(a1*np.pi*x)*np.sin(a2*np.pi*y)
        return u

    def solve(self, Nx=1001, Ny=1001, noise=0):
        """
        Generate a reference numerical solution (here it's actually exact) on a grid.

        Args:
            Nx, Ny (int): Resolution in x and y directions.
            noise (float): Optionally add noise.

        Returns:
            xy_flat (np.array): Flattened coordinates of shape (N,2)
            u_flat (np.array): Flattened solution u(x,y)
            (x_grid,y_grid,u_grid): Meshgrid arrays for visualization
        """
        x = np.linspace(self.lb[0], self.ub[0], Nx)[:,None]
        y = np.linspace(self.lb[1], self.ub[1], Ny)[:,None]
        x,y = np.meshgrid(x,y)
        xy  = np.stack((x,y), axis=-1)
        u   = self.Exact(xy)
        
        # Add noise if requested
        u += noise*np.random.randn(*u.shape)

        return xy.reshape(-1,2), u.reshape(-1,1), (x,y,u[...,0])

    def pde(self, x, y, aux, output):
        """
        PDE residual for Helmholtz equation:
        
        Δu + k² u = f
        Residual: res = u_xx + u_yy + k² u - f
        
        Args:
            x: input coordinates (x,y)
            y: target values (here y is used to provide f: forcing term)
            aux: auxiliary data (not used here)
            output: NN output representing u(x,y)

        Returns:
            res: PDE residual
            {"res": mean squared residual}: a metric for PINN training
        """
        f = y[:,0:1]
        u = grad.getOutput(output)
        u_xx = grad.partialDerivative(x, output, component=0, order_x_partial=(2,0))
        u_yy = grad.partialDerivative(x, output, component=0, order_x_partial=(0,2))
        res = u_xx + u_yy + k**2*u - f
        return res, {"res":torch.mean(torch.square(res))}

    def BC(self, x, y, aux, output):
        """
        Boundary condition: for this chosen exact solution,
        u is zero on the domain boundaries. So we enforce u=0 at the boundaries.
        """
        yp = grad.getOutput(output)
        diff = yp
        return diff, {"BC":torch.mean(torch.square(diff))}

    def testError(self, x, y, aux, output):
        """
        Compute relative L2 error between NN prediction and exact solution for testing:
        
        error = ||yp - y|| / ||y||

        Args:
            x,y,aux,output: standard arguments.
        
        Returns:
            {'test': relative error}
        """
        yp = grad.getOutput(output)
        err = torch.norm(yp-y)/torch.norm(y)
        return {'test':err}


