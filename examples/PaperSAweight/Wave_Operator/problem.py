# -*- coding: utf-8 -*-
#


import ETPINN
grad = ETPINN.grad
import torch

# This code sets up a physics-informed problem for the Wave equation with operator
#
# The wave equation in one dimension is typically given by:
#     u_tt = c² u_xx
#
# The input X can be decomposed into (xt, u0), where xt are space-time coordinates
# and u0 represent initial condition 

Nin = 2   # Input dimension: (x,t)
Nout = 1  # Output dimension: u(x,t)
L = 1     # Spatial domain length

class problem():
    def __init__(self, tlen=1, vis=0.01):
        """
        Initialize the wave equation problem.

        Args:
            tlen (float): Final time for the simulation.
            vis (float): Parameter playing the role of c², the wave speed squared in the PDE.
        """
        lb, ub = [0,0], [L,tlen]  # (x,t) domain from x=0 to x=L and t=0 to t=tlen
        self.tlen = tlen
        self.vis = vis

        # Geometry definition using ETPINN:
        self.spatialDomain = ETPINN.geometry.Interval(lb[0], ub[0])
        self.timeDomain    = ETPINN.geometry.TimeDomain(lb[1], ub[1])
        self.physicalDomain = ETPINN.geometry.GeometryXTime(self.spatialDomain, self.timeDomain)


        # Input/Output normalization parameters
        self.input_params = {"lb":[lb], "ub":[ub],
                             "u0_mean":0,
                             "u0_std":0}
        self.output_params = {"std":1, "mean":0}
        self.lb, self.ub = lb, ub

    def input_transform(self, X, lb, ub, u0_mean, u0_std):
        """
        Normalize the input (x,t) to a [-1,1] domain.

        Args:
            X (tuple): Input (xt, u0) where xt is (N,2) array for coordinates, u0 is initial condition data.
            lb, ub (list): Domain bounds.
            u0_mean, u0_std (float): Statistics for normalizing u0 if required.

        Returns:
            (xt, u0) after normalization of xt.
        """
        xt, u0 = X
        xt = (xt-(ub+lb)/2)/(ub-lb)*2
        return xt, u0

    def output_transform(self, y, xt, std, mean):
        """
        Denormalize the output.

        Args:
            y (torch.Tensor): Normalized NN output.
            xt (torch.Tensor): Inputs.
            std, mean (float): Stats for denormalization.

        Returns:
            torch.Tensor: Denormalized output.
        """
        return y*std+mean

    def pde(self, x, y, aux, output):
        """
        PDE residual for the wave equation:
        PDE: u_tt - vis*u_xx = 0
        where vis = c².
        """
        u_tt = grad.partialDerivative(x, output, component=0, indx=0, order_x_partial=(0,2))  # d²u/dt²
        u_xx = grad.partialDerivative(x, output, component=0, indx=0, order_x_partial=(2,0))  # d²u/dx²
        res = u_tt - self.vis*u_xx
        return res, {"res": torch.mean(torch.square(res))}

    def dataFit(self, x, y, aux, output, name='data'):
        """
        Data-fitting loss: if we have reference data y for some points,
        ensure NN matches it.

        Args:
            x,y,aux,output: Standard arguments.
            name (str): Label for this data loss component.

        Returns:
            diff, {name: MSE(diff)}.
        """
        yp = grad.getOutput(output)
        diff = yp - y
        return diff, {name: torch.mean(torch.square(diff))}

    def IC0(self, x, y, aux, output):
        """
        Initial condition that includes both u-u0=0 and u_t=0:
        """
        du = grad.getOutput(output)-y
        u_t = grad.partialDerivative(x, output, component=0, indx=0, order_x_partial=(0,1))
        diff = torch.cat((du, u_t), dim=1)
        return diff, {"IC": torch.mean(torch.square(du)),
                      "IC_t": torch.mean(torch.square(u_t))}

    def IC(self, x, y, aux, output):
        """
        Simpler initial condition enforcing only u(x,0) = u0.
        """
        du = grad.getOutput(output)-y
        return du, {"IC": torch.mean(torch.square(du))}

    def IC_t(self, x, y, aux, output):
        """
        Initial condition on the velocity u_t(x,0) = 0.
        """
        u_t = grad.partialDerivative(x, output, component=0, indx=0, order_x_partial=(0,1))
        return u_t, {"IC_t": torch.mean(torch.square(u_t))}

    def BC(self, x, y, aux, output):
        """
        Boundary condition u=0 
        """
        u = grad.getOutput(output)
        diff = u
        return diff, {"BC": torch.mean(torch.square(u))}

    def testError(self, x, y, aux, output):
        """
        Compute relative test error = ||yp - y|| / ||y||.

        Args:
            x,y,aux,output: standard arguments.

        Returns:
            {'test': relative error}
        """
        yp = grad.getOutput(output)
        err = torch.norm(yp-y)/torch.norm(y)
        return {'test': err}


