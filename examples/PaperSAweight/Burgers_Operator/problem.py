# -*- coding: utf-8 -*-
#


import os
import ETPINN
grad = ETPINN.grad
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.io import savemat, loadmat
import torch

# This code sets up a problem for the 1D Burgers equation with operators,

# The Burgers equation is written as:
#
#     u_t + u*u_x = ν u_xx
#
# where ν (here denoted as 'vis') is the viscosity.
#
# The input X can be decomposed into (xt, u0), where xt are space-time coordinates
# and u0 represent initial condition 


Nin = 2   # Input dimension: (x,t) typically, plus operator input handled separately
Nout = 1  # Output dimension: u(x,t)
path = "numsols"
    
class problem():
    def __init__(self, vis=0.01):
        """
        Initialize the Burgers problem with given viscosity.

        Args:
            vis (float): Viscosity (ν) parameter in the Burgers equation.
        """
        lb, ub = [0,0],[1,1]
        self.vis = vis
        # Use ETPINN geometry objects to define the problem domain in space and time
        self.spatialDomain = ETPINN.geometry.Interval(lb[0], ub[0])
        self.timeDomain    = ETPINN.geometry.TimeDomain(lb[1], ub[1])
        self.physicalDomain = ETPINN.geometry.GeometryXTime(self.spatialDomain, self.timeDomain)

        # Input/output normalization parameters
        self.input_params = {
            "lb":[lb],
            "ub":[ub],
            "u0_mean":0,
            "u0_std":1,
        }
        self.output_params = {"std":1, "mean":0}
        self.lb, self.ub = lb, ub

    def input_transform(self, X, lb, ub, u0_mean, u0_std):
        """
        Transform input coordinates (xt) and  operator inputs (u0).

        Args:
            X (tuple): Input tuple (xt, u0) where:
                       xt is a torch.Tensor of shape (N,2) with columns [x,t],
                       u0 is the operator input, if present.
            lb, ub (list): Lower and upper bounds for normalization.
            u0_mean, u0_std (float): Mean and std for normalizing u0.

        Returns:
            (xt,u0) after normalization. Here we only apply normalization to xt.
        """
        xt, u0 = X
        xt = (xt-(ub+lb)/2)/(ub-lb)*2
        # u0 could be normalized if needed, but currently is returned as is.
        return xt, u0

    def output_transform(self, y, X, std, mean):
        """
        De-normalize the output from the network if it was normalized.

        Args:
            y (torch.Tensor): NN output (normalized).
            X (tuple): Input tuple (xt, u0) where:
                       xt is a torch.Tensor of shape (N,2) with columns [x,t],
                       u0 is the operator input, if present.
            std, mean (float): Statistics for denormalization.

        Returns:
            torch.Tensor: Denormalized output.
        """
        return y*std+mean

    def pde(self, x, y, aux, output):
        """
        PDE residual for the Burgers equation:        
        PDE: u_t + u*u_x - ν u_xx = 0
        """
        uv = grad.getOutput(output)
        u  = uv[:,0:1]
        u_t = grad.partialDerivative(x, output, component=0, indx=0, order_x_partial=(0,1))  # du/dt
        u_x = grad.partialDerivative(x, output, component=0, indx=0, order_x_partial=(1,0))  # du/dx
        u_xx= grad.partialDerivative(x, output, component=0, indx=0, order_x_partial=(2,0)) # d²u/dx²

        res = u_t + u*u_x - self.vis*u_xx
        return res, {"res": torch.mean(torch.square(res))}

    def dataFit(self, x, y, aux, output, name='data'):
        """
        Data fitting loss: Compare NN output to given data y.
        """
        yp = grad.getOutput(output)
        diff = yp - y
        return diff, {name: torch.mean(torch.square(diff))}

    def IC(self, x, y, aux, output):
        """
        Initial condition loss: ensures u(x,0) matches the given initial data y.
        """
        yp = grad.getOutput(output)
        diff = yp - y
        return diff, {"IC":torch.mean(torch.square(diff))}

    def BC0(self, x, y, aux, output):
        """
        A boundary condition variant BC0:
        enforcing u = u_x = 0 at the boundary.
        """
        du = grad.getOutput(output)
        du_x = grad.partialDerivative(x, output, component=0, indx=0, order_x_partial=(1,0))
        
        diff = torch.cat((du, du_x), dim=1)
        return diff, {"BC":torch.mean(torch.square(du)),
                      "BC_x":torch.mean(torch.square(du_x))}

    def BC(self, x, y, aux, output):
        """
        Boundary condition that enforces u=0.
        """
        du = grad.getOutput(output)
        return du, {"BC":torch.mean(torch.square(du))}

    def BC_x(self, x, y, aux, output):
        """
        Boundary condition that enforces u_x=0.
        """
        du_x=grad.partialDerivative(x, output, component=0, indx=0, order_x_partial=(1,0))
        return du_x, {"BC_x":torch.mean(torch.square(du_x))}

    def testError(self, x, y, aux, output):
        """
        Compute relative test error: ||yp - y|| / ||y||
        
        Args:
            x,y,aux,output: Standard arguments.
        
        Returns:
            {'test': relative error}
        """
        yp = grad.getOutput(output)
        err  = torch.norm(yp-y)/torch.norm(y)
        return {'test':err}
