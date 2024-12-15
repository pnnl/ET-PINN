import torch
import numpy as np
from .utils.functions import treeToNumpy, treeToTensor
from . import utils

defaultOption = {
    "name": "default",
    "maxProbabilityRatio": 1,
    "betaR": 1,
}


class default():
    """
    A base adptive refinement strategy.
    """
    def __init__(self, trainBulletin, option=defaultOption):
        """
        Args:
            trainBulletin: trainBulletin class defined in model.py
            option: option dict
        """        
        self.trainBulletin = trainBulletin
        self.trainModel    = self.trainBulletin.trainModel
        self.option        = option or defaultOption
        self.dataSet       = self.trainBulletin.trainSet
        self.dataSet_old   = None
        self.doUpdate      = True
        self.name          = self.option['name']

    def updateCriterion(self):
        """
        Determine whether dataset refinement should occur.
        """
        #TBD
        return False

    def refineDataset(self):
        """
        Refine the dataset by adding/removing samples.
        """
        #TBD
        return

    def flatten(self, f, op='sum'):
        """
        Flatten a multi-dimensional array along one axis using a specified operation.
        """
        #TBD
        return 

    def sample(self, n, toTensor=True):
        """
        resample n points for self.dataset.
        """
        #TBD
        return dataSet

    def probability(self, loss):
        """
        Compute a probability distribution over samples
        """
        #TBD
        return 

    def printStatus(self):
        """
        Print status of this refinement strategy.
        """
        #TBD
        pass



METHOD_DICT = {
    None: default,
}

def get(option):
    """
    Retrieve the refinement strategy class based on the provided option.

    Args:
        option (dict): Should have a key 'name' identifying the refinement strategy.

    Returns:
        class: The refinement strategy class corresponding to the 'name'.
    """
    if option is None:
        return METHOD_DICT[None]
    assert "name" in option
    methodName = option['name']
    assert methodName in METHOD_DICT
    return METHOD_DICT[methodName]
