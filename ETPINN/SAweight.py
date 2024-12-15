# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 21:43:38 2022

@author: chen096
This module implements various three self-adaptive weighting strategies: minmax, RBA, BRDR.
These strategies adjust weights applied to training points during loss computation,
with the goal of improved training accuracy.
"""

import numpy as np
import torch
from . import optimizers
from . import utils
from .utils.functions import treeToNumpy, treeToTensor
from .trainPlot import SAplot

# Default options for different methods if none are provided.
defaultOption = {'init': lambda x,shape: np.ones(shape)}

defaultMinMaxOption = {
    'init': lambda x,shape: np.random.rand(*shape),
    'mask': lambda x: x**2,
    'mask_derivative': lambda x: 2*x,
    'mask_inverse': lambda x: x**0.5,                       
    'optimizerOption':{'name':'Adam','lr':5E-3,"lr_decay":lambda step: 1},
}

defaultRBAOption = {
    'init': lambda x,shape: np.ones(shape),
    'beta':0.999,
    'eta':0.01,
    'lambda0':0,
}

defaultBRDROption = {
    'init': lambda x,shape: np.ones(shape),
    'betaC':0.999,
    'betaW':0.999,
    'lamda0':1,
}

class SAMaster():
    """
    A master class to manage multiple SAweight instances (bulletins).
    This can coordinate operations across multiple training bulletins if needed.
    """
    def __init__(self):
        self.bulletins={}  # Maps bulletin to an index.
        self.trainModel=None

    def register(self, bulletin):
        """
        Register a bulletin into the master, assign it an index, and associate the master
        with the trainModel from the bulletin.
        """
        if bulletin not in self.bulletins:
            self.bulletins[bulletin] = len(self.bulletins)
            self.trainModel = bulletin.trainModel

_SAMaster = SAMaster()

class SAweight():
    """
    Base class for self-adaptive adaptive weighting strategies.
    It assigns weights to samples in the training set and adjusts
    how loss is computed to focus the training process.

    Default behavior:
    - Initializes weights as ones (or using a specified init function).
    - Provides methods to integrate these weights into the loss calculation.
    - Allows updating weights after each iteration or batch if needed.
    """
    def __init__(self, trainBulletin, option, Master=_SAMaster):
        """
        Args:
            trainBulletin: trainBulletin class defined in model.py
            option: option dict
        """             
        self.trainBulletin = trainBulletin
        self.trainModel    = self.trainBulletin.trainModel
        self.option        = option or defaultOption
        self.doUpdate      = True  # Whether to actively update weights

        # A function to generate the save filename for SAweights
        self.saveFile      = lambda step: self.trainBulletin.fileSystem.dataSave(
            self.trainBulletin.name+'_SAweight' +
            ('_step='+str(step) if step is not None else "")
        )

        # Print information about dataset size
        print(f"{self.trainBulletin.name}:, shape={len(self.trainBulletin.trainSet.X)}")

        # Evaluate one sample to determine shape of loss vector
        feedOnePoint = treeToTensor(self.trainBulletin.trainSet[0:1].unpack())
        self.trainBulletin.applyGrad(feedOnePoint[0])
        lossVec, _ = self.trainBulletin.output_lossVec(*feedOnePoint)

        # Initialize SAweights using provided init function or default
        assert callable(self.option['init']) or isinstance(self.option['init'], (int, float)) 
        initFun = (self.option['init'] if callable(self.option['init'])
                   else lambda x,shape: np.ones(shape)*self.option['init'])

        shape = (len(self.trainBulletin.trainSet), *lossVec.shape[1:])
        self.SAweights = initFun(treeToNumpy(self.trainBulletin.trainSet.X), shape)

        # Compute number of batches
        batchSize = self.trainBulletin.batchSize
        self.nbatch = 1 if batchSize==0 else self.SAweights.shape[0]/batchSize
        print(f"{self.trainBulletin.name}: nbatch={self.nbatch}, shape={self.SAweights.shape}")

        self.size = self.SAweights.size
        self.Master = Master
        self.Master.register(self)

    def preUpdate(self,*kargs):
        """
        Called before the weight update step
        Default does nothing.
        """
        return

    def postUpdate(self,*kargs):
        """
        Called after the weight update step
        Default does nothing.
        """
        return

    def update(self):
        """
        Called to actually update the weights.
        Default does nothing .
        """
        return

    def dataSetUpdate(self):
        """
        Called if the dataset itself is updated (e.g., refined).
        Can re-map weights to new samples. Default does nothing.
        """
        return

    def assembly(self, R2):
        """
        Integrate weights into the final loss computation.

        Args:
            R2: The squared residual or error vector for the current batch.

        Returns:
            loss: Weighted mean loss.
        """
        bulletin = self.trainBulletin
        ind = slice(None)
        if self.nbatch > 1:
            ind = np.array(bulletin.batchIndex)
        lossWeight = treeToTensor(self.SAweights[ind])
        loss = torch.mean(R2 * lossWeight)      
        return loss

    def save(self, step=None):
        """
        Save SAweights to disk for checkpointing.
        """
        torch.save(self.SAweights, self.saveFile(step))

    def restore(self, step=None):
        """
        Restore SAweights from a saved file.
        """
        self.SAweights = torch.load(self.saveFile(step), map_location=lambda storage, loc: storage)

    def plot(self):
        """
        Optional: plot or visualize the SAweights.
        Default does nothing.
        """
        pass

    def finalise(self):
        """
        Called at the end of training to finalize.
        Default does nothing.
        """
        pass

    def printStatus(self):
        """
        Print summary statistics of the current SAweights.
        """
        weights = self.SAweights 
        W = (weights.mean(), weights.min(), weights.std())
        print((f"weight for {self.trainBulletin.name}: [" + ", ".join([" %11.4e"]*3) + "]")%W )


class minMax(SAweight):
    """
    An implementation of  min-max adaptive weighting method, 
    proposed in the paper "Self-adaptive physics-informed neural networks using a soft attention mechanism"
    by McClenny, Levi, and Ulisses Braga-Neto.
    """
    def __init__(self, trainBulletin, option=defaultMinMaxOption):
        """
        Args:
            trainBulletin: trainBulletin class defined in model.py
            option: option dict
        """                 
        super(minMax, self).__init__(trainBulletin, option)
        self.initOptimizer()

    def initOptimizer(self):
        """
        Initialize Adam optimizer for updating SAweights.
        """
        self.exp_avg     = np.zeros_like(self.SAweights)
        self.exp_avg_sq  = np.zeros_like(self.SAweights)
        self.grad        = None
        self.step        = 0

        assert self.option['optimizerOption']['name']=='Adam'
        dummy_params = treeToTensor(np.ones(1))  # a dummy tensor for PyTorch optimizer
        self.optimizer, self.lr_scheduler = optimizers.get([dummy_params], self.option['optimizerOption']) 
        self.lr_run = self.optimizer.param_groups[0]['lr']     

    def assembly(self, R2):
        """
        Compute loss with masked weights.
        The mask is applied to SAweights before multiplying by R2.
        """
        lossWeight = treeToTensor(self.option['mask'](self.SAweights[self.ind]))
        loss = torch.mean(R2 * lossWeight)
        return loss

    def preUpdate(self, R2):
        """
        Compute gradient for SAweights based on the mask derivative and R2.
        Gradient is stored in self.grad.
        """
        if self.doUpdate:
            bulletin = self.trainBulletin
            self.ind = None
            if self.nbatch > 1:
                self.ind = np.array(bulletin.batchIndex)
            batchSize = self.SAweights[self.ind,:].size
            self.grad = -1/batchSize * treeToNumpy(R2) * self.option['mask_derivative'](self.SAweights[self.ind,:])
        return

    def update(self):
        """
        Use Adam to adjust SAweights based on computed gradients.
        """
        if not self.doUpdate: 
            return
        if self.optimizer:
            self.step += 1
            ind = self.ind
            beta1, beta2 = self.optimizer.param_groups[0]['betas']

            # Adam moments update
            self.exp_avg[ind]    = self.exp_avg[ind]*beta1    + (1-beta1)*self.grad  
            self.exp_avg_sq[ind] = self.exp_avg_sq[ind]*beta2 + (1-beta2)*self.grad*self.grad

            # Bias correction
            exp_avg_tilde    = self.exp_avg[ind]/(1-beta1**self.step)
            exp_avg_sq_tilde = self.exp_avg_sq[ind]/(1-beta2**self.step)

            # Adam step
            stepsize = exp_avg_tilde / (np.sqrt(exp_avg_sq_tilde) + 1e-8)
            self.SAweights[ind] += -self.lr_run * stepsize

        if self.lr_scheduler:
            # Update learning rate if scheduler is provided
            self.lr_run = self.optimizer.param_groups[0]['initial_lr'] * self.option['optimizerOption']['lr_decay'](self.step)


class RBA(SAweight):
    """
    implementation of Residual-based attention (RBA) weighting strategy
    proposed in the paper "Residual-based attention in physics-informed neural networks" 
    by Anagnostopoulos, Sokratis J and Toscano, Juan Diego and Stergiopulos, Nikolaos and Karniadakis, George Em
    """
    def __init__(self, trainBulletin, option=defaultRBAOption):
        """
        Args:
            trainBulletin: trainBulletin class defined in model.py
            option: option dict
        """                 
        super(RBA, self).__init__(trainBulletin, option)
        self.beta = self.option['beta']
        self.eta = self.option['eta']
        self.lamda0 = self.option['lambda0']
        self.R = self.SAweights*0
        self.Rmax = 1.

    def assembly(self, R2):
        # Weights are effectively squared
        lossWeight = treeToTensor(np.square(self.SAweights[self.ind,:]))
        loss = torch.mean(R2 * lossWeight)
        return loss

    def preUpdate(self, R2):
        """
        Compute residual magnitude R and its maximum,
        which are used to scale the updates to SAweights.
        """
        if self.doUpdate:
            bulletin = self.trainBulletin
            self.ind = slice(None)
            if self.nbatch > 1:
                self.ind = np.array(bulletin.batchIndex)
            self.R = np.sqrt(treeToNumpy(R2))
            self.Rmax = self.R.max()
        return

    def update(self):
        """
        Update SAweights: exponentially decay old weights and add a fraction of R/Rmax.
        """
        if not self.doUpdate: 
            return
        self.SAweights[self.ind] = self.beta * self.SAweights[self.ind] + self.eta * self.R / self.Rmax


class BRDRMaster(SAMaster):
    """
    A specialized master for the BRDR weighting strategy:
    - Maintains a global lambda (lamda) parameter and updates it.
    - Tracks mean inverse residual ratios (irdrMean) to normalize updates.
    """
    def __init__(self):
        super(BRDRMaster, self).__init__()
        self.lamda = 1
        self.lr = 0.001
        self.step = 1
        self.irdrMean = 1

    def updateK(self):
        """
        Update the global mean of inverse residual-driven ratio (irdrMean)
        across all bulletins.
        """
        irdrSum = 0
        batchSize = 0
        for bulletin, idx in self.bulletins.items():
            irdrSum += bulletin.irdrSum
            batchSize += bulletin.batchSize
        self.irdrMean = irdrSum / batchSize
        return

    def updateLamda(self):
        """
        Update lambda based on the current total_loss and NTK.
        This tries to adaptively scale the loss.
        """
        self.lr = min([para['lr'] for para in self.trainModel.opt.param_groups])
        self.step += 1
        if not self.doUpdateLamda:
            return
        with torch.no_grad():
            total_loss = self.trainModel.total_loss
            NTK = 0
            # Compute gradient norm to estimate Hessian or NTK complexity
            for value in self.trainModel.params.values():
                for paras in value:
                    for th in paras['params']:
                        if th.grad is not None:
                            NTK += th.grad.square().sum()

            lamda = (1 - self.lr)*self.lamda + self.lr*(2*total_loss/self.lr/NTK.item()*self.lamda)
            cor = lamda / self.lamda
            self.lamda = lamda
            # Scale gradients by factor 'cor'
            for value in self.trainModel.params.values():
                for paras in value:
                    for th in paras['params']:
                        if th.grad is not None:
                            th.grad *= cor
        return

_BRDRMaster = BRDRMaster()

def restart():
    """
    Helper function to reset global masters for BRDR and default SA.
    """
    global _BRDRMaster
    global _SAMaster
    _BRDRMaster = BRDRMaster()           
    _SAMaster = SAMaster()
    return

class BRDR(SAweight):
    """
    Implentation of BRDR (Balanced residual decay rate) weighting strategy
    proposed in the paper "Self-adaptive weights based on balanced residual decay rate for
    physics-informed neural networks and deep operator networks"
    by Chen, Wenqian and Howard, Amanda A and Stinis, Panos.
    
    """
    def __init__(self, trainBulletin, option=defaultBRDROption, Master=None):
        """
        Args:
            trainBulletin: trainBulletin class defined in model.py
            option: option dict
        """                 
        Master = Master or _BRDRMaster
        super(BRDR, self).__init__(trainBulletin, option, Master=Master)
        self.ind = None
        self.irdrSum = 0
        self.exp_avg_sq = self.SAweights*0
        self.last_seen = np.zeros_like(self.SAweights, dtype='int32')
        self.betaC = self.option['betaC']
        self.betaW = self.option['betaW']
        self.intervalMax = int(10*self.nbatch)
        self.betaC_effect = self.betaC**(np.arange(self.intervalMax+1))        
        self.betaW_effect = self.betaW**(np.arange(self.intervalMax+1))
        self.lamda0 = self.option.get('lamda0', 1.0)
        self.weight_pDistribution = np.ones_like(self.SAweights)
        
        # extra options to help test UpdateWeight or UpdateLamda sepearately
        self.doUpdateWeight = self.option.get('doUpdateWeight', True)
        self.doUpdateLamda  = self.option.get('doUpdateLamda', True)
        self.Master.doUpdateLamda  = self.doUpdateLamda
        

    def save(self, step=None):
        """
        Save state including SAweights and lamda for BRDR.
        """
        torch.save({"SAweights":self.SAweights,
                    "lamda":self.Master.lamda,
                    "irdr":self.irdr,},
                   self.saveFile(step))

    def restore(self,step=None):
        """
        Restore BRDR state including SAweights and lamda.
        """
        data= torch.load(self.saveFile(step), map_location=lambda storage, loc: storage)
        self.SAweights = data['SAweights']
        self.Master.lamda = data['lamda']
        self.irdr = data['irdr']

    def preUpdate(self, R2):
        """
        Compute inverse residual-decay ratio (irdr), update exponential averages of R2.
        """
        eps = 1e-14
        if self.doUpdate:
            bulletin = self.trainBulletin
            ind = self.ind = slice(None)
            if self.nbatch > 1:
                ind = self.ind = np.array(bulletin.batchIndex)
            exp = self.exp_avg_sq[ind]
            R2 = treeToNumpy(R2)
            self.batchSize = R2.size
            if self.nbatch > 1:
                self.dd_step = np.minimum(self.Master.step - self.last_seen[ind], self.intervalMax)
                beta_effect = self.betaC_effect[self.dd_step] 
            else:
                beta_effect = self.betaC

            exp += (1 - beta_effect)*(R2*R2 - exp)
            self.exp_avg_sq[ind] = exp
            cor = 1/(1 - self.betaC**self.Master.step)
            self.irdr = R2 / (np.sqrt(exp*cor) + eps)
            self.irdrSum = self.irdr.sum()
        return

    def postUpdate(self):
        """
        After weight updates, if this is the first bulletin, update global lamda.
        """
        if not self.doUpdate: 
            return
        if self.Master.bulletins[self] == 0:
            self.Master.updateLamda()
        return     

    def assembly(self, R2):
        """
        Compute final loss incorporating lamda and SAweights.
        """
        lamda, lossWeight = treeToTensor((self.Master.lamda, self.SAweights[self.ind],))
        loss = self.batchSize/self.size * lamda*self.lamda0*torch.mean(R2*lossWeight)
        return loss

    def update(self, do=False):
        """
        Update SAweights based on the ratio of irdr to global irdrMean.
        Applies exponential smoothing.
        """
        if not self.doUpdate:
            return
        if not self.doUpdateWeight:
            return
        if self.Master.bulletins[self] == 0:
            self.Master.updateK()
        ind = self.ind
        if self.nbatch > 1:
            beta_effect = self.betaW_effect[self.dd_step]
        else:
            beta_effect = self.betaW

        self.SAweights[ind] += (1 - beta_effect)*(self.irdr / self.Master.irdrMean - self.SAweights[ind])
        self.last_seen[ind] = self.Master.step
        return

    def printStatus(self):
        """
        Print lamda and current SAweight stats.
        Only the first registered bulletin prints lamda.
        """
        if self.Master.bulletins[self] == 0:
            print(f'lamda: {self.Master.lamda:11.4e}')
        super().printStatus()

    def inputNormalise(self,x):
        """
        Normalize input coordinates based on problem bounds.
        """
        lb = np.array(self.trainModel.problem.lb).reshape(1,-1)
        ub = np.array(self.trainModel.problem.ub).reshape(1,-1)
        nx = lb.shape[1]
        xnorm = (x[:,:nx] - (lb+ub)/2)/(ub-lb)*2
        return xnorm

    def dataSetUpdate(self):
        """
        If the dataset is refined, adaptively re-compute weights for new samples
        via inverse distance weighting from old samples.
        """
        assert self.trainBulletin.SArefine.name == "propabalityDensity"
        inputDistance = self.trainBulletin.SArefine.inputDistance
        dataSet_old   = self.trainBulletin.SArefine.dataSet_old
        dataSet       = self.trainBulletin.trainSet
        nNew = self.trainBulletin.SArefine.nChoose
        index= self.trainBulletin.SArefine.index_dataSet
        dis = inputDistance(dataSet[-nNew:].X, dataSet_old.X)
        distances = np.maximum(dis, 1E-14)
        idw_weights = 1.0 / distances**2
        idw_weights /= idw_weights.sum(axis=1)[:,None]
        SAweights = self.SAweights.copy()
        exp_avg_sq = self.exp_avg_sq.copy()
        self.SAweights[-nNew:]  = np.dot(idw_weights, SAweights)
        self.SAweights[:nNew]   = SAweights[index[:nNew]]
        self.exp_avg_sq[-nNew:] = np.dot(idw_weights, exp_avg_sq)
        self.exp_avg_sq[:nNew]  = exp_avg_sq[index[:nNew]]
        return


# A dictionary mapping method names to classes
METHOD_DICT = {
    None: SAweight,
    "default": SAweight,
    "minMax": minMax,
    'BRDR': BRDR,
    'RBA': RBA,
}

def get(option):
    """
    Retrieve the SAweight class corresponding to the given option.
    The 'option' dict must contain a 'name' key that specifies which strategy to use.
    """
    if option is None:
        return METHOD_DICT[None]
    assert "name" in option
    methodName = option['name']
    assert methodName in METHOD_DICT
    return METHOD_DICT[methodName]
