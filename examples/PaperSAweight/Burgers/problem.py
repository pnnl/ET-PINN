# -*- coding: utf-8 -*-

import os
import ETPINN
grad = ETPINN.grad
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat

# Problem setup:
# This code sets up a physics-informed problem for the 1D Burgers equation.
# The Burgers equation is a fundamental partial differential equation (PDE)
# often used as a simplified model for fluid dynamics and nonlinear wave propagation.
# It takes the form:
#
#     u_t + u * u_x = ν u_xx
#
# where u is the velocity field, t is time, x is the spatial coordinate,
# and ν (vis) is the viscosity.
#
# In this code, we aim to solve the Burgers equation for a given domain and time horizon
# using a physics-informed neural network (PINN) approach. The code leverages 
# the ETPINN framework and sets up the problem domain, PDE, initial and boundary conditions,
# as well as reading exact solutions from a .mat file for comparison.

Nin = 2  # Number of input dimensions (x and t)
Nout = 1 # Number of output dimensions (u)
path = "numsols"  # Path to directory containing numerical solutions
L = 1   # Spatial half-length of the domain, domain is [-L, L] in x

class problem():
    def __init__(self, tlen=1, vis=0.01):
        """
        Initialize the Burgers equation problem.
        
        Args:
            tlen (float): The final time of the simulation. 
            vis (float): Viscosity parameter (ν) in the Burgers equation.
        """
        lb, ub = [-L,0], [L,tlen]  # Lower and upper bounds in space-time (x in [-L, L], t in [0, tlen])
        self.tlen = tlen
        self.vis = vis
        
        # Define geometry and time domain using ETPINN's geometry utilities.
        self.spatialDomain = ETPINN.geometry.Interval(lb[0], ub[0])
        self.timeDomain    = ETPINN.geometry.TimeDomain(lb[1], ub[1])
        self.physicalDomain = ETPINN.geometry.GeometryXTime(self.spatialDomain, self.timeDomain)

        # Default resolution for exact solution sampling:
        self.Nx_exact, self.Nt_exact = 2001, 10001

        # Input-output normalization parameters
        self.input_params = {"lb":[lb], "ub":[ub]}
        self.output_params = {"std":1, "mean":0}
        self.lb, self.ub = lb, ub

    def input_transform(self, xt, lb, ub):
        """
        Normalize input (x,t) to a canonical domain, typically [-1,1].
        
        Args:
            xt (np.array): Array of input points of shape (N,2) where N is number of points.
            lb (list): Lower bound [x_lb, t_lb].
            ub (list): Upper bound [x_ub, t_ub].
        
        Returns:
            xt (np.array): Transformed input.
        """
        # Shift and scale the input to map domain [lb, ub] to [-1,1]
        xt = (xt-(ub+lb)/2)/(ub-lb)*2
        return xt

    def output_transform(self, y, xt, std, mean):
        """
        Transform the output from normalized to physical scale.
        
        Args:
            y (np.array): Normalized output.
            xt (np.array): Input points (not used here).
            std (float): Standard deviation used for normalization.
            mean (float): Mean used for normalization.
        
        Returns:
            np.array: De-normalized output.
        """
        return y*std+mean

    def initFun(self, xt):
        """
        Initial condition function for the Burgers equation.
        
        Typically, at t=0, we set u(x,0).
        
        Here: u(x,0) = -sin(pi*x/L)
        
        Args:
            xt (np.array): Input points, first column is x.
        
        Returns:
            np.array: Initial values for u at given x.
        """
        x = xt[:,0:1]
        return -np.sin(np.pi*x/L)

    def solve(self, Nx=None, Nt=None, noise=0, strip_x=1, strip_t=1):
        """
        Load a numerical solution of the Burgers equation from a .mat file.
        
        The file 'Burges_v{vis}_{Nx}_{Nt}.mat' contains precomputed solutions.
        
        Args:
            Nx (int): Number of spatial points. Defaults to self.Nx_exact if None.
            Nt (int): Number of time points. Defaults to self.Nt_exact if None.
            noise (float): Optional noise added to the solution.
            strip_x (int): Downsampling factor in x-direction.
            strip_t (int): Downsampling factor in t-direction.
        
        Returns:
            xt (np.array): The coordinates of the solution points.
            u (np.array): The corresponding solution values u(x,t).
            (x,t,u2D): Spatial (x) and temporal (t) grids and solution u2D(t,x).
        """
        if Nx is None or Nt is None:
            Nx, Nt = self.Nx_exact, self.Nt_exact

        filename = os.path.join(path, f"Burges_v{str(self.vis)}_{Nx}_{Nt}.mat")
        data = loadmat(filename)

        x = data['x'].squeeze()[::strip_x]
        t = data['t'].squeeze()[::strip_t]
        u2D = data['res'][::strip_t, ::strip_x]

        print(u2D.shape)

        # Add noise if requested
        u2D += noise*np.random.randn(*u2D.shape)

        # Adjust time dimension to the specified tlen
        nChoose = np.sum(t <= self.tlen)
        t = t[:nChoose]
        u2D = u2D[:nChoose,:]

        # Create a mesh of (x,t)
        xt = np.stack(np.meshgrid(x,t), axis=0).reshape((2,-1)).T
        u = u2D.reshape(1,-1).T

        return xt, u, (x,t,u2D)

    def pde(self, x, y, aux, output):
        """
        PDE residual for the Burgers equation:
        
        Given the output of the NN (u), compute:
        
        u_t + u * u_x - (vis/pi) * u_xx = 0
        
        Args:
            x: input coordinates (x,t)
            y: not used here, since PDE doesn't directly use labeled data
            aux: auxiliary info (not used)
            output: NN output representing u(x,t)
        
        Returns:
            res (torch.Tensor): The PDE residual.
            {"res": <scalar>: The mean squared residual, a metric for PINN training.}
        """
        uv = grad.getOutput(output)
        u  = uv[:,0:1]

        # Partial derivatives of u w.r.t x and t:
        u_t = grad.partialDerivative(x, output, component=0, order_x_partial=(0,1))   # du/dt
        u_x = grad.partialDerivative(x, output, component=0, order_x_partial=(1,0))   # du/dx
        u_xx= grad.partialDerivative(x, output, component=0, order_x_partial=(2,0))   # d²u/dx²

        # Burgers PDE residual:
        res = u_t + u*u_x - self.vis/torch.pi * u_xx
        return res, {"res":torch.mean(torch.square(res))}

    def dataFit(self, x, y, aux, output, name='data'):
        """
        Data fitting loss. If we have target data (y), we match the NN output to it.
        
        Args:
            x: input points
            y: target values
            aux: auxiliary info (not used)
            output: NN output u(x,t)
            name (str): label for this loss component
        
        Returns:
            diff: difference between NN prediction and target
            {name: mean squared error of the difference}
        """
        yp = grad.getOutput(output)
        diff = yp - y
        return diff, {name:torch.mean(torch.square(diff))}


    def IC(self, x, y, aux, output):
        """
        Initial condition loss: ensures the NN matches the initial condition u(x,0).
        """
        return self.dataFit(x, y, aux, output, name='IC')

    def BC(self, x, y, aux, output):
        """
        Boundary condition loss: ensures that u matches boundary conditions u=0 at x boundaries.
        """
        yp = grad.getOutput(output)
        diff = yp
        return diff, {"BC":torch.mean(torch.square(diff))}

    def testError(self, x, y, aux, output):
        """
        Compute a relative test error between NN prediction and true solution y.
        
        Args:
            x: input points
            y: exact solution values
            aux: not used
            output: NN output
        
        Returns:
            dict: {'test': relative error}
        """
        yp = grad.getOutput(output)
        err = torch.norm(yp-y)/torch.norm(y)
        return {'test':err}

