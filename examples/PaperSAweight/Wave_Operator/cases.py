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
# Default: args = [0, 1, 1] if no arguments provided.

import sys, os
# Parse command-line arguments. 
args = [int(p) for p in sys.argv[1:]]
if len(args) == 0:
    args = [0, 1, 1]
is1, is2, is3 = args[0:3]
repeatID = is1

import ETPINN
# Configure device, precision, and seed for reproducibility
ETPINN.config.oneKey_configure(use_cuda=True, precision=32, seed=3**(repeatID+1))

import numpy as np
from problem import problem, Nin, Nout
from numpy.random import permutation as perm
from ETPINN.utils.functions import treeToNumpy, treeToTensor

# Configure parameters based on command-line arguments
case_params = {
    "repeatID": is1,
    "vis": (1, 2, 4, 8)[is2],      
    "SAweight": (None, 'BRDR')[is3]
}

# Define constants for training data sizes and batch sizes
Nic, Nbc, Nres, N_train = 101, 100, 2500, 1000
batchSize_ic, batchSize_bc, batchSize_res = 10000, 10000, 10000

# Define training steps and optimizers
StepTotal = 200000
AdamRatio = 1.0
StepAdam = int(StepTotal * AdamRatio)
StepLbfgs = int(StepTotal * (1 - AdamRatio))
savenet_everyAdam = -1  
savenet_everyLBFGS = -1 
lr = 0.001
lr_decay = lambda step: 0.99 ** (step / (StepAdam // 400))  
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
    def __init__(self, *, repeatID=0, vis=2, SAweight=None):
        # Initialize the case with specified parameters.
        super(case, self).__init__(repeatID=repeatID, vis=vis, SAweight=SAweight)

    def init(self):
        # Initialize the PDE problem
        self.problem = problem(vis=self.vis)
        
        # Define architecture for a DEEPONN (Deep Operator Network) model
        # "trunk" network handles coordinates (x,t), "brunch" handles input functions (u0)
        Architecture = {
            'activation': 'tanh',
            'initializer': "default",
            "layer_sizes_trunk": [2] + [100]*7,  # trunk network: input=(x,t) -> deep layers
            "layer_sizes_brunch": [101] + [100]*7, # brunch network: input=(u0) -> deep layers
            "Nout": 1, # single output dimension (e.g., solution u at given (x,t;u0))
        }


        # Create a DEEPONN model
        self.net = ETPINN.nn.DEEPONN_sharedEncoder(Architecture=Architecture)
        
        # Apply input and output transforms (normalization, boundary handling, etc.)
        self.net.apply_input_transform(self.problem.input_transform, self.problem.input_params)
        self.net.apply_output_transform(self.problem.output_transform, self.problem.output_params)

        caseStr = self.caseStr
        subdir = "AdaptiveWeight"
        # Filesystem object to manage directories and file naming
        self.fileSystem = ETPINN.fileSystem('.', subdir, '_' + caseStr)
        return self

    @ETPINN.utils.decorator.timing
    def buildTest(self, Nchoose=10000):
        # Build test dataset for evaluating the trained operator.
        data = np.load(f'wave_test_C={self.vis}.npz')
        u0 = data['u0']
        sol = data['u']
        t = data['t']
        x = data['x']

        # Create a mesh of space-time points
        T, X = np.meshgrid(t, x)
        xt = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])

        # Replicate u0 across all (x,t) points to form input pairs (ytest=(x,t), utest=u0)
        utest = np.tile(u0[:, None, :], (1, X.size, 1)).reshape(-1, Nic)
        ytest = np.tile(xt[None, :, :], (len(u0), 1, 1)).reshape(-1, 2)
        stest = sol.reshape(-1, 1)  # reference solution flatten

        # Randomly select a subset of test points
        ind = perm(utest.shape[0])[:int(Nchoose)]
        datatest = ETPINN.data.datasetDomain(X=(ytest[ind], utest[ind]), Y=stest[ind])
        
        # Create a test bulletin for automated testing after training steps
        bulletinsTest = ETPINN.testBulletin('test', datatest, self.net.forward, self.problem.testError,
                                            fileSystem=self.fileSystem)
        return bulletinsTest

    def buildHF(self, forwardFun=None):
        # Build the high-fidelity training datasets (IC, BC, PDE residual) for this PDE-operator problem.
        if forwardFun is None:
            forwardFun = self.net.forward

        data = np.load('wave_train.npz')
        u0 = data['u0'][:N_train]

        # Initial condition (IC): (x,0) domain, match solution at t=0
        xtic = np.zeros((Nic, 2))
        xtic[:, 0] = data['x']  # spatial coordinates
        # Tile and reshape to create (N_train * Nic) samples
        uic = np.tile(u0[:, None, :], (1, Nic, 1)).reshape(-1, Nic)
        yic = np.tile(xtic[None, :, :], (N_train, 1, 1)).reshape(-1, 2)
        sic = u0.reshape(-1, 1)  # solution = initial condition
        dataIC = ETPINN.data.datasetIC(X=(yic, uic), Y=sic)
        dataIC_t = ETPINN.data.datasetIC(X=(yic, uic))

        # Boundary conditions (BC): Random points on t in [0,1] and x=0 or x=1
        xtbc = np.random.rand(N_train, Nbc, 2)
        # Half at x=0, half at x=1 boundary
        xtbc[:, :Nbc//2, 0] = 0
        xtbc[:, Nbc//2:, 0] = 1
        ybc = xtbc.reshape(-1, 2)
        ubc = np.tile(u0[:, None, :], (1, Nbc, 1)).reshape(-1, Nic)
        dataBC = ETPINN.data.datasetBC(X=(ybc, ubc))

        # PDE residual (Eq points): interior points in domain
        yres = np.random.rand(N_train*Nres, 2)
        ures = np.tile(u0[:, None, :], (1, Nres, 1)).reshape(-1, Nic)
        dataEq = ETPINN.data.datasetBC(X=(yres, ures))  # Using BC dataset type for convenience here

        # Get SAweight option based on chosen SAweight scheme (None or 'BRDR')
        option = SAweightOption[self.SAweight]

        # Create training bulletins 
        pde = ETPINN.trainBulletin(
            'pde', dataEq, 1.0, forwardFun, self.problem.pde, fileSystem=self.fileSystem,
            SAweightOption=option, Xrequires_grad=[True, False], batchSize=batchSize_res
        )
        bc = ETPINN.trainBulletin(
            'BC', dataBC, 1.0, forwardFun, self.problem.BC, fileSystem=self.fileSystem,
            SAweightOption=option, Xrequires_grad=False, batchSize=batchSize_bc
        )
        ic = ETPINN.trainBulletin(
            'IC', dataIC, 1.0, forwardFun, self.problem.IC, fileSystem=self.fileSystem,
            SAweightOption=option, Xrequires_grad=[True, False], batchSize=batchSize_ic
        )
        ic_t = ETPINN.trainBulletin(
            'IC_t', dataIC_t, 1.0, forwardFun, self.problem.IC_t, fileSystem=self.fileSystem,
            SAweightOption=option, Xrequires_grad=[True, False], batchSize=batchSize_ic
        )

        bulletins = [ic, ic_t, bc, pde]
        return bulletins

    def buildModel(self):
        # Build the training model by combining the train bulletins and the test bulletin
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
        # Train the model, first using Adam optimizer, then L-BFGS if StepLbfgs > 0
        nstr = len(self.caseStr)
        print("~" * nstr + "\n" + self.caseStr + "\n" + "~" * nstr)

        optimizerAdam = {'name': 'Adam', 'lr': lr, "lr_decay": lr_decay}
        optimizerLBFGS = {'name': 'L-BFGS'}
        commonParams = {
            "display_every": display_every,
            "savenet_every": savenet_everyAdam,
            "doRestore": False
        }

        # Train with Adam
        self.model.train(optimizerAdam, steps=StepAdam, **commonParams)
        # If AdamRatio=1.0, no LBFGS steps will be performed because StepLbfgs=0
        self.model.train(optimizerLBFGS, steps=StepLbfgs, **commonParams)

    def loadnet(self, netfile=None, status=None):
        # Load a pretrained network if exists
        try:
            name = self.net.name + '_best' if status == 'best' else self.net.name
            netfile = self.fileSystem.netfile(name)
            if os.path.isfile(netfile):
                self.net.loadnet(netfile)
                valid = True
                # If desired, we could verify validity by test predictions here.
                # Currently, just assume valid if it loads successfully.
                if valid:
                    return True
                else:
                    print(f'invalid net file <<{self.caseStr}>>')
                    return False
        except:
            print(f'unable to load net file <<{self.caseStr}>>')
            return False

    def predict(self, x, forwardFun=None):
        # Generate predictions from the trained network
        if forwardFun is None:
            forwardFun = self.net.forward
        y = treeToNumpy(forwardFun(treeToTensor(xi) for xi in x))
        return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.close("all")
    allCases = case(**case_params)

    # Iterate over all cases (if there's more than one)
    for casei in allCases:
        hasNet = casei.loadnet()
        # If a pretrained network is found and we don't want to retrain, we could skip:
        if False and hasNet:
            print(f"case <<{casei.caseStr}>> has already been built")
            continue
        casei.init()  # re-initialize the case
        casei.buildModel().train()  # Build and then train the model
