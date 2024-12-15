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
# sys.argv[1:] are interpreted as [repeatID, nResi, is3, is4].
# Default: args = [0, 100, 1,3] if no arguments provided.


import sys, os
# Parse command-line arguments
args = [int(p) for p in sys.argv[1:]]
if len(args)==0:
    # default parameters
    args = [0, 100, 1,3]
is1, is2, is3, is4 = args[0:4]
repeatID = is1

import ETPINN
# Configure device, precision, and seed for reproducibility
ETPINN.config.oneKey_configure(use_cuda=True, precision=32, seed=2**(repeatID%10+10+1))

import numpy as np
from problem import problem, Nin, Nout
from ETPINN.utils.functions import treeToNumpy, treeToTensor
from numpy.random import permutation as perm

# Configure parameters based on command-line arguments
case_params = {
    "repeatID" : is1,
    "nResi"    : is2,
    "vis"      : (0.01, 0.00314, 0.001, 0.000314)[is3],
    "SAweight" : (None, 'minMax', 'RBA', 'BRDR', 'BRDR100')[is4]
}

# Fixed training parameters
StepTotal = 40000       # Total training steps
AdamRatio = 1           # Ratio of total steps for Adam optimizer
Nbatch_HF = 1           # Batch number for PDE residual dataset
StepAdam  = int(StepTotal*AdamRatio)
StepLbfgs = 0           
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
    'doUpdateLamda':True,
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
    def __init__(self,*,repeatID=0, vis=0.01, nResi=10000, SAweight=True):
        super(case, self).__init__(repeatID=repeatID,
                                   vis=vis,
                                   nResi=nResi,
                                   SAweight=SAweight)
    def init(self):
        # Initialize the PDE problem
        self.problem = problem(vis=self.vis)
        Architecture = {
            'useEncoder':True,
            'activation':'tanh',
            'initializer':"default",
            "layer_sizes":[Nin] + [128]*6 + [Nout],
        }
        
        # Create neural network
        self.net = ETPINN.nn.FNN(Architecture=Architecture)
        
        # Apply input/output transforms from the problem definition
        self.net.apply_input_transform(self.problem.input_transform, self.problem.input_params)
        self.net.apply_output_transform(self.problem.output_transform, self.problem.output_params)
        
        # Create a filesystem object for saving results and checkpoints
        caseStr = self.caseStr
        subdir = "AdaptiveWeight"
        self.fileSystem = ETPINN.fileSystem('.',subdir,'_'+caseStr)
        return self

    @ETPINN.utils.decorator.timing
    def buildTest(self, Nchoose=10000):
        # Build test dataset by solving problem and choosing a subset
        XLF,YLF,_ = self.problem.solve()
        ind = perm(XLF.shape[0])[:int(Nchoose)]
        datatest = ETPINN.data.datasetDomain(X=XLF,Y=YLF)[ind]
        
        # Create a test bulletin for evaluating the model
        bulletinsTest = ETPINN.testBulletin('test', datatest, self.net.forward, self.problem.testError,
                                            fileSystem=self.fileSystem)
        return bulletinsTest

    def buildHF(self, forwardFun=None):
        # Build PDE  training dataset
        if forwardFun is None: forwardFun = self.net.forward
        dataHF = ETPINN.data.datasetDomain(X='LHS',
                                           physicalDomain=self.problem.physicalDomain,
                                           length=self.nResi)

        # Initial condition dataset
        dataIC = ETPINN.data.datasetIC(X='uniform',
                                       Y=self.problem.initFun,
                                       physicalDomain=self.problem.physicalDomain,
                                       length=100*max(int(self.nResi/9999),1))
        
        # Boundary condition dataset
        dataBC = ETPINN.data.datasetBC(X='pseudo',
                                       filterX = lambda x: x[np.isclose(x[:,0], self.problem.lb[0])
                                                             | np.isclose(x[:,0], self.problem.ub[0]), :],
                                       physicalDomain=self.problem.physicalDomain,
                                       length=200*max(int(self.nResi/9999),1))
        
        # SAweight config for PDE bulletin
        option = SAweightOption[self.SAweight]
        option_pde = option.copy()
        if self.SAweight == 'BRDR100':
            option_pde['lambda0'] = 5
        
        # PDE bulletin 
        pde = ETPINN.trainBulletin('pde', dataHF, 1.0, forwardFun, self.problem.pde,
                                   fileSystem=self.fileSystem,
                                   SAweightOption= option_pde,
                                   Xrequires_grad=True)
        
        # BC and IC bulletins with respective SAweight options
        option_bc = option.copy() if self.SAweight != 'RBA' else {'name':'default', 'init':1}
        bc = ETPINN.trainBulletin('BC', dataBC, 1.0, forwardFun, self.problem.BC, fileSystem=self.fileSystem,
                                  SAweightOption= option_bc)
        
        option_ic = option.copy() if self.SAweight != 'RBA' else {'name':'default', 'init':1}
        ic = ETPINN.trainBulletin('IC', dataIC, 1.0, forwardFun, self.problem.IC, fileSystem=self.fileSystem,
                                  SAweightOption= option_ic)
        
        bulletins = [ic, bc, pde]
        return bulletins

    def buildModel(self):
        # Build training and test bulletins, then wrap them into a trainModel object
        bulletins = self.buildHF()
        bulletinsTest = self.buildTest()
        self.model = ETPINN.trainModel(self.problem, self.net, bulletins,
                                       fileSystem=self.fileSystem,
                                       testBulletins=bulletinsTest)
        return self

    @ETPINN.utils.decorator.timing
    def train(self):
        # Print case string and start training
        nstr = len(self.caseStr)
        print("~"*nstr+ "\n" +  self.caseStr+ "\n" +"~"*nstr)        
        optimizerAdam  = {'name':'Adam','lr':lr,"lr_decay":lr_decay}
        optimizerLBFGS = {'name':'L-BFGS'}
        commonParams   = { "display_every":display_every, 
                           "savenet_every":savenet_everyAdam,
                           "doRestore":False}
        
        # First, train with Adam optimizer
        self.model.train(optimizerAdam, steps=StepAdam, **commonParams)
        # Optionally train with LBFGS optimizer if StepLbfgs > 0
        self.model.train(optimizerLBFGS, steps=StepLbfgs, **commonParams)

    def loadnet(self, netfile=None, status=None):
        # Load a saved network if it exists
        try:
            name = self.net.name + '_best' if status=='best' else self.net.name
            netfile = self.fileSystem.netfile(name)
            if os.path.isfile(netfile):
                self.net.loadnet(netfile)
                valid=True
                # If needed, check validity by test dataset prediction
                # For now, assume valid=True
                if valid:
                    return True
                else:
                    print(f'invalid net file <<{self.caseStr}>>')
                    return False
        except:
            print(f'unable to load net file <<{self.caseStr}>>')
            return False

    def predict(self,x,forwardFun=None):
        # Predict using the trained network
        if forwardFun is None: forwardFun = self.net.forward
        y = treeToNumpy(forwardFun(treeToTensor(x))) 
        return y

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.close("all")
    allCases = case(**case_params)
    for casei in allCases:
        # Attempt to load and check previously trained network
        hasNet = casei.loadnet()
        if False and hasNet:
            print(f"case <<{casei.caseStr}>> has already been built")
            continue
        casei.init() # re-init if needed
        casei.buildModel().train()
