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
# sys.argv[1:] are interpreted as [repeatID, nResi, is3, is4].
# Default: args = [0, 100, 1,0] if no arguments provided.

import sys, os
import ETPINN
import numpy as np
from problem import problem, Nin, Nout
from ETPINN.utils.functions import treeToNumpy, treeToTensor

# Parse command-line arguments. 
args = [int(p) for p in sys.argv[1:]]
if len(args) == 0:
    # Default args if none provided
    args = [0, 100, 1, 0]
is1, is2, is3, is4 = args[0:4]

repeatID = is1
# Configure device, precision, and seed for reproducibility
ETPINN.config.oneKey_configure(
    use_cuda=True,       # Use GPU if available
    precision=32,        # Precision setting for computations
    seed=2**(repeatID % 10 + 10 + 1) # Set a reproducible random seed
)

# Configure parameters based on command-line arguments
case_params = {
    "repeatID": is1,
    "nResi": is2,
    "ksi": (2, 4, 8)[is3],
    "useSAweight": (True, False)[is4],
}

# Fixed training parameters
StepTotal = 100000   # Total number of training steps
AdamRatio = 1.0      # Ratio of steps to use Adam optimizer before switching to L-BFGS
Nbatch_HF = 1        # Batch number for PDE residual trainingtraining
StepAdam  = int(StepTotal * AdamRatio)     # Steps for Adam optimization
StepLbfgs = int(StepTotal * (1 - AdamRatio)) # Steps for L-BFGS optimization
savenet_everyAdam = -1  # Frequency of saving network during Adam training (-1 means no save)
savenet_everyLBFGS = -1 # Frequency of saving network during L-BFGS (-1 means no save)
lr = 0.001               # Learning rate
lr_decay = lambda step: 1 # Learning rate decay function (no decay here)
display_every = 100       # Display training info every 100 steps

# Define self-adaptive weighting option
SAweightOption = {
    'name': 'BRDR',
    'init': lambda x, shape: np.ones(shape),
    'betaC': 0.999,
    'betaW': 0.999,
    'doUpdateLamda': True,
}


class case(ETPINN.utils.cases_generator):
    """
    The case class represents a single training scenario with given parameters.
    It inherits from cases_generator to allow iteration over multiple parameter combinations if desired.

    On initialization, it sets up parameters like repeatID, vis, nResi, SAweight.
    The init method sets up the problem, network, and files for saving results.
    buildTest builds test datasets, buildHF builds PDE/BC/IC bulletins, and buildModel assembles everything.
    """    
    def __init__(self, *, repeatID=0, ksi=0.01, 
                 nResi=10000, useSAweight=True):
        # Initialize the case with specified parameters.
        super(case, self).__init__(repeatID=repeatID,
                                   ksi=ksi, nResi=nResi, useSAweight=useSAweight)

    def init(self):
        # Initialize the PDE problem
        self.problem = problem(ksi=self.ksi)
        
        # Define network architecture:
        Architecture = {
            'useEncoder': False,
            'activation': 'tanh',
            'initializer': "Glorot normal",
            "layer_sizes": [Nin] + [50]*6 + [Nout], 
            "features": None,
        }

        # Create a network using ETPINN
        self.net = ETPINN.nn.FNN(Architecture=Architecture)
        
        # Apply input and output transforms specified by the problem
        self.net.apply_input_transform(self.problem.input_transform, self.problem.input_params)
        self.net.apply_output_transform(self.problem.output_transform, self.problem.output_params)
        
        # Filesystem object to manage directories and file naming
        caseStr = self.caseStr
        subdir = "AdaptiveWeight"
        self.fileSystem = ETPINN.fileSystem('.', subdir, '_' + caseStr)        
        return self

    @ETPINN.utils.decorator.timing
    def buildTest(self):
        # Build a test dataset for performance evaluation
        XLF, YLF = self.problem.solve(Nx=10000)
        datatest = ETPINN.data.datasetDomain(X=XLF, Y=YLF)
        
        # Prepare a test bulletin to evaluate the model
        bulletinsTest = ETPINN.testBulletin(
            'test', datatest, self.net.forward, self.problem.testError,
            fileSystem=self.fileSystem
        )
        return bulletinsTest

    def buildHF(self, forwardFun=None):
        # Build bulletins for PDE and BC training
        if forwardFun is None:
            forwardFun = self.net.forward
        
        # PDE residual training data
        dataHF = ETPINN.data.datasetDomain(
            X='uniform', 
            physicalDomain=self.problem.physicalDomain, 
            length=self.nResi
        )
        
        # Boundary condition training data
        dataBC = ETPINN.data.datasetBC(
            X=np.array([[self.problem.lb[0]],
                        [self.problem.ub[0]]], dtype=np.float64),
            Y=np.array([[0.],
                        [0.]], dtype=np.float64)
        )
        
        # Create training bulletins for PDE and BC.
        pde = ETPINN.trainBulletin(
            'pde', dataHF, 1.0, forwardFun, self.problem.pde,
            fileSystem=self.fileSystem,
            SAweightOption=SAweightOption if self.useSAweight else None, Xrequires_grad=True
        )
        
        bc = ETPINN.trainBulletin(
            'BC', dataBC, 1.0, forwardFun, self.problem.BC,
            fileSystem=self.fileSystem,
            SAweightOption=SAweightOption if self.useSAweight else None
        )
        
        bulletins = [bc, pde]
        return bulletins

    def buildModel(self):
        # Build model with train training bulletins and test bulletin
        bulletins = self.buildHF()
        bulletinsTest = self.buildTest()
        
        # Construct the training model which combines problem, network, and bulletins
        self.model = ETPINN.trainModel(
            self.problem, self.net, bulletins,
            fileSystem=self.fileSystem,
            testBulletins=bulletinsTest
        )
        return self

    @ETPINN.utils.decorator.timing
    def train(self):
        # Training procedure: first Adam, then L-BFGS if StepLbfgs > 0
        nstr = len(self.caseStr)
        print("~" * nstr + "\n" + self.caseStr + "\n" + "~" * nstr)

        # Define optimizers
        optimizerAdam = {'name': 'Adam', 'lr': lr, "lr_decay": lr_decay}
        optimizerLBFGS = {'name': 'L-BFGS'}
        
        # Common training parameters for Adam and LBFGS
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
        
        # If there are LBFGS steps, train with LBFGS optimizer
        # (If AdamRatio=1.0, StepLbfgs=0 and no LBFGS steps will run)
        if StepLbfgs > 0:
            self.model.train(optimizerLBFGS, steps=StepLbfgs, **commonParamsBFGS)

    def loadnet(self, netfile=None, status=None, epoch=None):
        # Load a pretrained network from a file, if exists
        try:
            if netfile is None:
                name = self.net.name + '_best' if status == 'best' else self.net.name
                netfile = self.fileSystem.netfile(name)
            if os.path.isfile(netfile):
                self.net.loadnet(netfile)
                # Validate the network by making a prediction on test data
                test = self.buildTest()
                valid = np.all(np.isfinite(self.predict(treeToNumpy(test.testSet.X))))
                if valid:
                    return True
                else:
                    print(f'invalid net file <<{self.caseStr}>>')
                    return False
        except:
            print(f'unable to load net file <<{self.caseStr}>>')
            return False

    def predict(self, x, forwardFun=None):
        # Make a prediction using the trained network
        if forwardFun is None:
            forwardFun = self.net.forward
        y = treeToNumpy(forwardFun(treeToTensor(x)))
        return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.close("all")

    # Create a case instance with parameters defined above
    allCases = case(**case_params)

    # Iterate over all generated cases (if the generator produces multiple)
    for casei in allCases:
        # Attempt to load  and check a pre-trained network if available
        hasNet = casei.loadnet()
        # If we have a network already and don't want to retrain, skip
        if False and hasNet:
            print(f"case <<{casei.caseStr}>> has already been built")
            continue
        # Otherwise, initialize, build model, and train
        casei.init()  # Re-initialize (in case loadnet did not succeed)
        casei.buildModel().train()
