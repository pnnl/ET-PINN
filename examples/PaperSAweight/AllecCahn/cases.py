# -*- coding: utf-8 -*-

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
# Default: args = [0, 100, 2] if no arguments provided.

import sys, os

# Parse command-line arguments
args = [int(p) for p in sys.argv[1:]]
if len(args)==0:
    args = [0, 100, 2]
is1, is2, is3 = args[0:3]
repeatID = is1

import ETPINN
# Configure device, precision, and seed for reproducibility
ETPINN.config.oneKey_configure(use_cuda=True, precision=32, seed=2**(repeatID%10+10+1))

import numpy as np
from problem import problem, Nin, Nout
from numpy.random import permutation as perm
from ETPINN.utils.functions import treeToNumpy, treeToTensor
import torch

# Define case parameters based on command-line inputs
case_params = {
    "repeatID" : is1,
    "nResi"    : is2,
    "SAweight" : (None, 'minMax', 'RBA', 'BRDR', 'BRDR100')[is3]
}

# Fixed training parameters
StepTotal = 300000           # Total training steps
AdamRatio = 1.0              # Ratio of total steps for Adam optimizer
Nbatch_HF = 1                # Batch number for PDE residual dataset
StepAdam = int(StepTotal*AdamRatio)
StepLbfgs = int(StepTotal*(1-AdamRatio)) 
savenet_everyAdam =-1 # int(StepAdam*0.01) # Frequency of saving checkpoints during Adam training
savenet_everyLBFGS = -1      # Frequency of saving checkpoints during LBFGS training
lr = 0.001                   # Learning rate for Adam
lr_decay = lambda step: 0.99**( step/(StepAdam//400) )
display_every = 100          # Print training status every 100 steps

# Define different SAweight (self-adaptive weighting) options
SAweightOption = {}
SAweightOption[None] = {'name':'default', 'init': lambda x,shape: np.ones(shape)}

SAweightOption['BRDR'] = {
    'name':'BRDR',
    'init': lambda x,shape: np.ones(shape),
    'betaC':0.999,
    'betaW':0.999,
}
SAweightOption['BRDR100'] = SAweightOption['BRDR']  # Reuse same config

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
    'beta':0.999,
    'eta':0.01,
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
    def __init__(self, *, repeatID=0, nResi=10000, SAweight=True):
        super(case, self).__init__(repeatID=repeatID, nResi=nResi, SAweight=SAweight)

    def init(self):
        # Initialize the PDE problem defined in problem.py
        self.problem = problem()
        Nin = 2*self.problem.nMode+1  # Update input dimension based on the number of Fourier modes

        # Define the neural network architecture
        Architecture = {
            'useEncoder':True,
            'activation':'tanh',
            'initializer':"default",
            "layer_sizes":[Nin] + [128]*6 + [Nout],
        }
        self.net = ETPINN.nn.FNN(Architecture=Architecture)

        # Apply input/output transforms for normalization, as defined in the problem
        self.net.apply_input_transform(self.problem.input_transform, self.problem.input_params)
        self.net.apply_output_transform(self.problem.output_transform, self.problem.output_params)

        # Create a filesystem object to save results
        caseStr = self.caseStr
        subdir = "AdaptiveWeight"
        self.fileSystem = ETPINN.fileSystem('.', subdir, '_'+caseStr)
        return self

    @ETPINN.utils.decorator.timing
    def buildTest(self, Nchoose=50000):
        # Build test dataset by sampling from the problem's solution
        XLF, YLF, _ = self.problem.solve()
        ind = perm(XLF.shape[0])[:int(Nchoose)]
        datatest = ETPINN.data.datasetDomain(X=XLF, Y=YLF)[ind]

        # Create a test bulletin to evaluate the trained model
        bulletinsTest = ETPINN.testBulletin('test', datatest, self.net.forward, self.problem.testError,
                                            fileSystem=self.fileSystem)
        return bulletinsTest

    def buildHF(self, forwardFun=None):
        # Build PDE  dataset 
        if forwardFun is None:
            forwardFun = self.net.forward
        dataHF = ETPINN.data.datasetDomain(X='LHS',
                                           physicalDomain=self.problem.physicalDomain,
                                           length=self.nResi)

        # Initial condition dataset
        dataIC = ETPINN.data.datasetIC(X='uniform',
                                       Y=self.problem.initFun,
                                       physicalDomain=self.problem.physicalDomain,
                                       length=512)

        # Boundary condition dataset
        dataBC = ETPINN.data.datasetBC(X='pseudo',
                                       filterX=lambda x: x[np.isclose(x[:,0], self.problem.lb[0])
                                                           | np.isclose(x[:,0], self.problem.ub[0]), :],
                                       physicalDomain=self.problem.physicalDomain,
                                       length=200*max(int(self.nResi/9999),1))

        # Get SAweight options for PDE, BC, IC bulletins
        option = SAweightOption[self.SAweight]

        # PDE bulletin
        pde = ETPINN.trainBulletin('pde', dataHF, 1.0, forwardFun, self.problem.pde, fileSystem=self.fileSystem,
                                   SAweightOption=option,
                                   Xrequires_grad=True)

        # BC bulletin
        bc = ETPINN.trainBulletin('BC', dataBC, 1.0, forwardFun, self.problem.BC, fileSystem=self.fileSystem,
                                  SAweightOption= option)

        option_ic = option.copy() if self.SAweight != 'RBA' else {'name':'default', 'init': 100 }
        if self.SAweight == 'BRDR100':
            option_ic['lambda0'] = 100
        # IC bulletin
        ic = ETPINN.trainBulletin('IC', dataIC, 1.0, forwardFun, self.problem.IC, fileSystem=self.fileSystem,
                                  SAweightOption= option_ic)

        # Return the bulletins used for training: IC and PDE (and optionally BC)
        # Note: BC is defined but not appended here, since the boundary conditions are inherently satisfied with Fourier feature embedding.
        bulletins = [ic, pde]
        return bulletins

    def buildModel(self):
        # Assemble the model using the train bulletins and a test bulletin
        bulletins = self.buildHF()
        bulletinsTest = self.buildTest()
        self.model = ETPINN.trainModel(self.problem, self.net, bulletins,
                                       fileSystem=self.fileSystem,
                                       testBulletins=bulletinsTest)
        return self

    @ETPINN.utils.decorator.timing
    def train(self):
        # Print case info and start training
        nstr = len(self.caseStr)
        print("~"*nstr+ "\n" +  self.caseStr+ "\n" +"~"*nstr)

        optimizerAdam  = {'name':'Adam','lr':lr,"lr_decay":lr_decay}
        optimizerLBFGS = {'name':'L-BFGS'}
        commonParams = {
            "display_every":display_every,
            "savenet_every":savenet_everyAdam,
            "doRestore":False
        }

        # Train with Adam optimizer first
        self.model.train(optimizerAdam, steps=StepAdam, **commonParams)
        # If LBFGS steps are available, train with LBFGS next
        self.model.train(optimizerLBFGS, steps=StepLbfgs, **commonParams)

    def loadnet(self, netfile=None, status=None):
        # Attempt to load an existing trained model (checkpoint)
        try:
            name = self.net.name + '_best' if status=='best' else self.net.name
            netfile = self.fileSystem.netfile(name)
            if os.path.isfile(netfile):
                self.net.loadnet(netfile)
                valid=True
                # If additional checks required, implement them here
                return True
            else:
                return False
        except:
            print(f'unable to load net file <<{self.caseStr}>>')
            return False

    def predict(self, x, forwardFun=None):
        # Use the trained network to predict outputs for given inputs x
        if forwardFun is None: forwardFun = self.net.forward
        y = treeToNumpy(forwardFun(treeToTensor(x)))
        return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.close("all")
    allCases = case(**case_params)
    for casei in allCases:
        hasNet = casei.loadnet()
        if False and hasNet:
            print(f"case <<{casei.caseStr}>> has already been built")
            continue
        casei.init() # re-init if needed
        casei.buildModel().train()
