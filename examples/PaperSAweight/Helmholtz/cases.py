# -*- coding: utf-8 -*-
#
# This script sets up and trains a physics-informed neural network (PINN) for a given PDE problem based on ETPINN framework.
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
# sys.argv[1:] are interpreted as [repeatID, nResi, is3].
# Default: args = [0, 100, 3] if no arguments provided.


import sys, os

# Parse command-line arguments
args = [int(p) for p in sys.argv[1:]]
if len(args) == 0:
    args = [0, 100, 3]  # Default values if no arguments provided

is1, is2, is3 = args
repeatID = is1

import ETPINN
# Configure device (CUDA if available), precision (32-bit float), and set a seed for reproducibility
ETPINN.config.oneKey_configure(use_cuda=True, precision=32, seed=2**(repeatID%10+10+1))

import numpy as np
from problem import problem, Nin, Nout
from numpy.random import permutation as perm
from ETPINN.utils.functions import treeToNumpy, treeToTensor
import torch

# Configure parameters based on command-line arguments
case_params = {
    "repeatID" : is1,
    "nResi"    : is2,
    "SAweight" : (None, 'minMax', 'RBA', 'BRDR', 'BRDR100')[is3],
}

nMode = 5  # Number of Fourier modes (problem-specific parameter)

# Fixed training parameters
StepTotal = 100000           # Total training steps
AdamRatio = 1.0              # Ratio of total steps for Adam optimizer
Nbatch_HF = 1                # Batch number for PDE residual dataset
StepAdam  = int(StepTotal*AdamRatio)
StepLbfgs = 0  # If LBFGS steps are zero, only Adam will run
savenet_everyAdam = -1  # Frequency of saving checkpoints during Adam training
savenet_everyLBFGS = -1 # Frequency of saving checkpoints during LBFGS training
lr        = 0.001       # Learning rate for Adam
lr_decay  = lambda step: 0.99**(step/(StepTotal//400))
display_every = 100      # Print training status every 100 steps

# Define different SAweight (self-adaptive weighting) options
SAweightOption = {}
SAweightOption[None] = {'name':'default', 'init': lambda x,shape: np.ones(shape)}
SAweightOption['BRDR'] = {
    'name':'BRDR',
    'init': lambda x,shape: np.ones(shape),
    'betaC':0.999,
    'betaW':0.999,
}
SAweightOption['BRDR100'] = SAweightOption['BRDR']
SAweightOption['minMax'] = {
    "name":"minMax",
    'useSAweight': True,
    'mask': lambda x: x**2,
    'mask_derivative': lambda x: 2*x,
    'mask_inverse': lambda x: x**0.5,
    'init': lambda x,shape:np.random.rand(*shape),
    'optimizerOption':{'name':'Adam','lr':5E-3,"lr_decay":lambda x: 1.0},
}
SAweightOption['RBA'] = {
    'name':'RBA',
    'init': lambda x,shape: np.zeros(shape),
    'beta':0.9999,
    'eta':0.001,
    'lambda0':0,
}


class case(ETPINN.utils.cases_generator):
    """
    The case class represents a single training scenario with given parameters.
    It inherits from cases_generator to allow iteration over multiple parameter combinations if desired.

    On initialization, it sets up parameters like repeatID, vis, nResi, SAweight.
    The init method sets up the problem, network, and files for saving results.
    buildTest builds test datasets, buildHF builds PDE/BC/IC bulletins, and buildModel assembles everything.
    """
    def __init__(self, *, repeatID=0, nResi=10000, SAweight=None):
        super(case, self).__init__(repeatID=repeatID, nResi=nResi, SAweight=SAweight)

    def init(self):
        # Initialize PDE problem from problem.py
        self.problem = problem(nMode=nMode)
        Architecture = {
            'useEncoder': True,
            'activation': 'tanh',
            'initializer': "default",
            "layer_sizes": [2+0*nMode*nMode*3] + [128]*6 + [Nout],
        }
        # Build a Network
        self.net = ETPINN.nn.FNN(Architecture=Architecture)

        # Apply input and output transformations for normalization
        self.net.apply_input_transform(self.problem.input_transform, self.problem.input_params)
        self.net.apply_output_transform(self.problem.output_transform, self.problem.output_params)

        caseStr = self.caseStr
        subdir = "AdaptiveWeight"
        # Filesystem object to manage directories and file naming
        self.fileSystem = ETPINN.fileSystem('.', subdir, '_'+caseStr)
        return self

    @ETPINN.utils.decorator.timing
    def buildTest(self, Nchoose=10000):
        # Generate a test dataset by solving the problem at a fine resolution
        XLF, YLF, _ = self.problem.solve(Nx=1001, Ny=1001)
        ind = perm(XLF.shape[0])[:Nchoose]
        datatest = ETPINN.data.datasetDomain(X=XLF, Y=YLF)[ind]

        # Create a test bulletin for evaluating model performance 
        bulletinsTest = ETPINN.testBulletin('test', datatest, self.net.forward, self.problem.testError,
                                            fileSystem=self.fileSystem)
        return bulletinsTest

    def buildHF(self, forwardFun=None):
        # Build PDE and BC datasets
        if forwardFun is None: 
            forwardFun = self.net.forward

        # Create PDE residual points 
        x = np.linspace(-1,1,101)
        y = np.linspace(-1,1,101)
        xy = np.stack(np.meshgrid(x,y), axis=-1).reshape(-1,2)
        xyResi = xy[np.random.permutation(xy.shape[0])[:self.nResi],:]
        dataHF = ETPINN.data.datasetDomain(X=xyResi, Y=self.problem.fsource)

        # Create BC dataset 
        NBC=200
        vec = np.linspace(-1,1, NBC//4)
        xyBC  = np.zeros((NBC,2))
        xyBC[0*NBC//4:1*NBC//4,1]=-1
        xyBC[1*NBC//4:2*NBC//4,1]=1
        xyBC[2*NBC//4:3*NBC//4,0]=-1
        xyBC[3*NBC//4:4*NBC//4,0]=1
        xyBC[0*NBC//4:1*NBC//4,0]=vec
        xyBC[1*NBC//4:2*NBC//4,0]=vec
        xyBC[2*NBC//4:3*NBC//4,1]=vec
        xyBC[3*NBC//4:4*NBC//4,1]=vec
        dataBC = ETPINN.data.datasetBC(X=xyBC)

        # Retrieve SAweight option
        option = SAweightOption[self.SAweight]

        # PDE bulletin
        pde = ETPINN.trainBulletin('pde', dataHF, 1.0, forwardFun, self.problem.pde,
                                   fileSystem=self.fileSystem, SAweightOption=option, Xrequires_grad=True)

        # BC bulletin
        option_bc = option.copy() if self.SAweight != 'RBA' else {'name':'default', 'init':100}
        if self.SAweight == 'BRDR100':
            option_bc['lambda0'] = 100
        bc = ETPINN.trainBulletin('BC', dataBC, 1.0, forwardFun, self.problem.BC,fileSystem=self.fileSystem, SAweightOption=option_bc)
        
        bulletins = [bc, pde]
        return bulletins

    def buildModel(self):
        # Build the entire model by combining train PDE and BC bulletins, along with test bulletins
        bulletins = self.buildHF()
        bulletinsTest = self.buildTest()
        self.model = ETPINN.trainModel(self.problem, self.net, bulletins,
                                       fileSystem=self.fileSystem,
                                       testBulletins=bulletinsTest)
        return self

    @ETPINN.utils.decorator.timing
    def train(self):
        # Print case information and start training
        nstr = len(self.caseStr)
        print("~"*nstr+ "\n" +  self.caseStr+ "\n" +"~"*nstr)
        optimizerAdam = {'name':'Adam', 'lr':lr, "lr_decay":lr_decay}
        optimizerLBFGS = {'name':'L-BFGS'}
        commonParamsAdam = {
            "display_every": display_every,
            "savenet_every": savenet_everyAdam,
            "doRestore": False
        }
        commonParamsBFGS = {
            "display_every": display_every,
            "doRestore": False
        }

        # Train with Adam optimizer
        self.model.train(optimizerAdam, steps=StepAdam, **commonParamsAdam)
        # If LBFGS steps > 0, train with LBFGS after Adam (here it's zero by default)
        self.model.train(optimizerLBFGS, steps=StepLbfgs, **commonParamsBFGS)

    def loadnet(self, netfile=None, status=None):
        # Attempt to load a previously saved network
        try:
            name = self.net.name + '_best' if status=='best' else self.net.name
            netfile = self.fileSystem.netfile(name)
            if os.path.isfile(netfile):
                self.net.loadnet(netfile)
                # Validate the loaded net by checking if predictions are finite
                test = self.buildTest()
                valid = np.all(np.isfinite(self.predict(treeToNumpy(test.testSet.X))))
                if valid:
                    return True
                else:
                    print(f'invalid net file <<{self.caseStr}>>')
                    return False
            else:
                return False
        except Exception as e:
            print(e,f'unable to load net file <<{self.caseStr}>>')
            return False

    def predict(self, x, forwardFun=None):
        # Predict the solution at given points x using the trained network
        if forwardFun is None:
            forwardFun = self.net.forward
        y = treeToNumpy(forwardFun(treeToTensor(x)))
        return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.close("all")

    # Create a 'case' instance with the chosen parameters and iterate over them
    allCases = case(**case_params)
    for casei in allCases:
        # Attempt to load and check a previously trained network
        hasNet = casei.loadnet()
        if False and hasNet:
            print(f"case <<{casei.caseStr}>> has already been built")
            continue
        casei.init()  # Re-initialize 
        casei.buildModel().train()  # Build the model and start training
