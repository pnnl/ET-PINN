# -*- coding: utf-8 -*-
# This script sets up and trains a physics-informed DeepONet (PIDeepONet) for a given operator learning problem based on ETPINN framework.
#
# The script:
# - Parses command-line arguments to configure certain parameters (like number of residual points, viscosity, etc.).
# - Sets up a problem class (imported from 'problem.py') which defines PDE, domain, and boundary/initial conditions.
# - Configures training steps, optimizer parameters, and adaptive weighting methods.
# - Builds training bulletins (pde, BC, IC) and test bulletins for evaluating performance.
# - Trains the model using Adam and optional LBFGS optimizers.
# - Saves and optionally loads previously trained networks.
#
# Command-line arguments:
# sys.argv[1:] are interpreted as [repeatID, is2, is3].
# Default: args = [0, 0, 1] if no arguments provided.

import sys, os
# Parse command-line arguments:
args = [int(p) for p in sys.argv[1:]]
if len(args) == 0:
    args = [0, 0, 1]
is1, is2, is3 = args[0:3]
repeatID = is1

import ETPINN
# Configure device, precision, and seed for reproducibility
ETPINN.config.oneKey_configure(use_cuda=True, precision=32, seed=2**(repeatID%10+10+1))

import numpy as np
sys.path.insert(0,'../tools/NNs')
from problem import problem, Nin, Nout
from numpy.random import permutation as perm
from ETPINN.utils.functions import treeToNumpy, treeToTensor
import scipy
import torch

# Set parameters for the run:
case_params = {
    "repeatID": repeatID,
    "vis": (0.01, 0.001, 0.0001)[is2],
    "SAweight": (None, 'BRDR')[is3],  
}

# Define constants for training data sizes and batch sizes
Nic, Nbc, Nres, N_train = 101, 100, 2500, 1000
batchSize_ic, batchSize_bc, batchSize_res = 10000, 10000, 10000

# Define training steps and optimizers
PP = 1
StepTotal = 200000 * PP
AdamRatio = 1.0
StepAdam = int(StepTotal * AdamRatio)
StepLbfgs = int(StepTotal * (1 - AdamRatio))
savenet_everyAdam = -1  
savenet_everyLBFGS = int(StepLbfgs * 0.1)
lr = 0.001
lr_decay = lambda step: 0.99**( step / (StepAdam // 400) // PP )
display_every = 100

# Define SAweight options
SAweightOption = {}
SAweightOption[None] = {
    'name': 'default',
    'init': lambda x, shape: np.ones(shape)
}
SAweightOption['BRDR'] = {
    'name': 'BRDR',
    'init': lambda x, shape: np.ones(shape),
    'betaC': 0.9999,
    'betaW': 0.999,
}


class case(ETPINN.utils.cases_generator):
    """
    The case class represents a single training scenario with given parameters.
    It inherits from cases_generator to allow iteration over multiple parameter combinations if desired.

    On initialization, it sets up parameters like repeatID, vis, nResi, SAweight.
    The init method sets up the problem, network, and files for saving results.
    buildTest builds test datasets, buildHF builds PDE/BC/IC bulletins, and buildModel assembles everything.
    """     
    def __init__(self, *, repeatID=0, vis=0.01, SAweight=True):
        # Initialize the case with given parameters
        super(case, self).__init__(repeatID=repeatID, vis=vis, SAweight=SAweight)

    def init(self):
        # Initialize PDE problem
        self.problem = problem(vis=self.vis)

        # Define a DEEPONN architecture (Deep Operator Network) with shared encoder
        # "trunk" network: handles spatial-temporal coordinates (x,t)
        # "brunch" network: handles the input function (u0)
        Architecture = {
            'activation': 'tanh',
            'initializer': "default",
            "layer_sizes_trunk": [2] + [100]*7,   # trunk input dimension: 2 (x,t)
            "layer_sizes_brunch": [101] + [100]*7, # brunch input dimension: 101 (discretized u0)
            "Nout": 1,  # output dimension: solution u(x,t; u0)
        }

        self.net = ETPINN.nn.DEEPONN_sharedEncoder(Architecture=Architecture)

        # Apply input/output transforms defined in problem.py
        self.net.apply_input_transform(self.problem.input_transform, self.problem.input_params)
        self.net.apply_output_transform(self.problem.output_transform, self.problem.output_params)

        caseStr = self.caseStr
        subdir = "AdaptiveWeight"
        # Filesystem object to manage directories and file naming
        self.fileSystem = ETPINN.fileSystem('.', subdir, '_' + caseStr)
        return self

    def forwardPeriodic(self, x):
        # forwardPeriodic is defined to handle periodic boundary conditions.
        x0 = x
        x1 = [torch.cat((x[0][:,0:1] + 1, x[0][:,1:2]), dim=1), x[1]]
        y0 = self.net.forward(x0)
        y1 = self.net.forward(x1)
        return y0 - y1

    @ETPINN.utils.decorator.timing
    def buildTest(self, Nchoose=10000):
        # Build a test dataset:
        data = scipy.io.loadmat(f'burger_nu_{self.vis}_test.mat')
        u0 = data['input']   # shape: [N_test, Nic]
        sol = data['output'] # solution data [N_test, ..., 1], flattened below
        P = 101
        # Create grid of t and x
        t = np.linspace(0, 1, P)
        x = np.linspace(0, 1, P)
        T, X = np.meshgrid(t, x)
        xt = np.hstack([T.flatten()[:, None], X.flatten()[:, None]])  # (P^2, 2)

        # Replicate input conditions (u0) across all (x,t) points
        utest = np.tile(u0[:, None, :], (1, P*P, 1)).reshape(-1, Nic)
        ytest = np.tile(xt[None, :, :], (len(u0), 1, 1)).reshape(-1, 2)
        stest = sol.reshape(-1, 1)

        # Randomly choose a subset of test points if desired
        ind = perm(utest.shape[0])[:int(Nchoose)]
        datatest = ETPINN.data.datasetDomain(X=(ytest[ind], utest[ind]), Y=stest[ind])

        # Create a test bulletin for  testing
        bulletinsTest = ETPINN.testBulletin(
            'test', datatest, self.net.forward, self.problem.testError,
            fileSystem=self.fileSystem
        )
        return bulletinsTest

    def buildHF(self, forwardFun=None):
        # Build training datasets for PDE (residual), IC, BC with periodic conditions.
        if forwardFun is None:
            forwardFun = self.net.forward

        # Load training initial conditions
        u0 = scipy.io.loadmat('burger_train.mat')['input'][:N_train]

        # Initial condition dataset (IC)
        xtic = np.zeros((Nic, 2))
        xtic[:, 0] = np.linspace(0, 1, Nic) # x coordinates from 0 to 1
        # Create IC dataset by tiling u0 and xtic
        uic = np.tile(u0[:, None, :], (1, Nic, 1)).reshape(-1, Nic)
        yic = np.tile(xtic[None, :, :], (N_train, 1, 1)).reshape(-1, 2)
        sic = u0.reshape(-1, 1)
        dataIC = ETPINN.data.datasetIC(X=(yic, uic), Y=sic)

        # Boundary conditions (BC): 
        xtbc = np.random.rand(N_train, Nbc, 2)
        xtbc[:, :, 0] = 0  # fix x=0
        ybc = xtbc.reshape(-1, 2)
        ubc = np.tile(u0[:, None, :], (1, Nbc, 1)).reshape(-1, Nic)
        dataBC = ETPINN.data.datasetBC(X=(ybc, ubc))
        dataBC_x = ETPINN.data.datasetBC(X=(ybc, ubc))

        # PDE residual data (Eq): 
        yres = np.random.rand(N_train*Nres, 2)
        ures = np.tile(u0[:, None, :], (1, Nres, 1)).reshape(-1, Nic)
        dataEq = ETPINN.data.datasetBC(X=(yres, ures))

        # Get SAweight option based on SAweight setting
        option = SAweightOption[self.SAweight]
        option_pde = option.copy()

        # Create training bulletins: PDE, BC, BC_x (derivative BC), and IC
        pde = ETPINN.trainBulletin(
            'pde', dataEq, 1.0, forwardFun, self.problem.pde, fileSystem=self.fileSystem,
            SAweightOption=option_pde, Xrequires_grad=[True, False], batchSize=batchSize_res
        )
        bc = ETPINN.trainBulletin(
            'BC', dataBC, 1.0, self.forwardPeriodic, self.problem.BC, fileSystem=self.fileSystem,
            SAweightOption=option, Xrequires_grad=False, batchSize=batchSize_bc
        )
        bc_x = ETPINN.trainBulletin(
            'BC_x', dataBC_x, 1.0, self.forwardPeriodic, self.problem.BC_x, fileSystem=self.fileSystem,
            SAweightOption=option, Xrequires_grad=[True, False], batchSize=batchSize_bc
        )
        ic = ETPINN.trainBulletin(
            'IC', dataIC, 1.0, forwardFun, self.problem.IC, fileSystem=self.fileSystem,
            SAweightOption=option, Xrequires_grad=False, batchSize=batchSize_ic
        )

        bulletins = [ic, bc, bc_x, pde]
        return bulletins

    def buildModel(self):
        # Combine training bulletins and testing bulletins into the model
        bulletins = self.buildHF()
        bulletinsTest = self.buildTest()
        self.model = ETPINN.trainModel(
            self.problem, self.net, bulletins,
            fileSystem=self.fileSystem,
            testBulletins=bulletinsTest
        )
        return self

    @ETPINN.utils.decorator.timing
    def train(self):
        # Train the model: first Adam, then LBFGS if applicable
        nstr = len(self.caseStr)
        print("~"*nstr + "\n" + self.caseStr + "\n" + "~"*nstr)

        optimizerAdam = {'name': 'Adam', 'lr': lr, "lr_decay": lr_decay}
        optimizerLBFGS = {'name': 'L-BFGS'}
        commonParams = {
            "display_every": display_every,
            "savenet_every": savenet_everyAdam,
            "doRestore": False
        }

        # Train with Adam optimizer
        self.model.train(optimizerAdam, steps=StepAdam, **commonParams)
        # If AdamRatio=1.0, StepLbfgs=0, so no L-BFGS step will run.
        self.model.train(optimizerLBFGS, steps=StepLbfgs, **commonParams)

    def loadnet(self, netfile=None, status=None):
        # Load a pre-trained network if it exists
        try:
            name = self.net.name + '_best' if status == 'best' else self.net.name
            netfile = self.fileSystem.netfile(name)
            if os.path.isfile(netfile):
                self.net.loadnet(netfile)
                valid = True
                # Could optionally validate predictions here
                if valid:
                    return True
                else:
                    print(f'invalid net file <<{self.caseStr}>>')
                    return False
        except:
            print(f'unable to load net file <<{self.caseStr}>>')
            return False

    def predict(self, x, forwardFun=None):
        # Predict using the trained network
        if forwardFun is None:
            forwardFun = self.net.forward
        y = treeToNumpy(forwardFun(treeToTensor(xi) for xi in x))
        return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.close("all")
    allCases = case(**case_params)

    # Loop over cases (if case generator creates multiple cases)
    for casei in allCases:
        hasNet = casei.loadnet()
        # If a trained model already exists and we don't want to retrain, skip training:
        if False and hasNet:
            print(f"case <<{casei.caseStr}>> has already been built")
            continue

        # Otherwise, initialize and train the model
        casei.init()  # Re-initialize if needed
        casei.buildModel().train()
