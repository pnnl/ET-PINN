import os
import numpy as np

from . import gradients as grad
from . import optimizers
from . import utils
from . import nn
import torch
import time
from . import SAweight, SArefine
from .utils.functions import treeToNumpy, treeToTensor


class bulletin():
    """
    The base bulletin class holds shared attributes and methods for 'trainBulletin' and 'testBulletin'.
    A 'bulletin' manages data input (possibly multiple input tensors), the requirement for gradients, 
    and slicing of input tensors if needed.
    """
    def __init__(self, Xrequires_grad, dataSet):
        # Xrequires_grad can be a boolean or a list of booleans indicating if gradients are required 
        # with respect to the input tensors.
        self.Xrequires_grad = Xrequires_grad
        self.is_multiX = False

        # Check if dataSet.X is multiple tensors (tuple/list) or a single tensor.
        if isinstance(dataSet.X, (list, tuple)):
            self.is_multiX = True
            # Ensure the length of Xrequires_grad matches the number of input tensors.
            if isinstance(Xrequires_grad, (list, tuple)):
                assert len(Xrequires_grad) == len(dataSet.X)
                self.Xrequires_grad = Xrequires_grad
            else:
                self.Xrequires_grad = [Xrequires_grad] * len(dataSet.X)
        else:
            # Single input tensor case: just keep the boolean as is.
            pass

    def init(self):
        """
        To be called after initialization to prepare PyTorch computations.
        For example, sets 'requires_grad' for inputs or compiles necessary functions.
        """
        self.compilePytorch()

    def reset(self, **args):
        """
        Reset certain attributes of the bulletin based on provided keyword arguments.
        Raises an exception if an unknown key is provided.
        """
        for key in args:
            if key in self.__dict__:
                self.__dict__[key] = args[key]
            else:
                raise Exception("Unknown key %s" % key)

    def applyGrad(self, X):
        """
        Set the `requires_grad_` property of inputs based on self.Xrequires_grad.
        If multiple inputs, apply to each; otherwise, apply to the single input tensor.
        """
        if self.is_multiX:
            for x, requires_grad in zip(X, self.Xrequires_grad):
                x.requires_grad_(requires_grad)
        else:
            X.requires_grad_(self.Xrequires_grad)
        return

    def sliceInput(self, X):
        """
        Slices the input tensors if gradients are required dimension-wise. 
        This can help in some advanced partial derivative computations.

        Returns:
            Xtuple: A tuple of either the original tensors (if no grad required) or the sliced tensors (one slice per dimension).
            Xrecover: The recovered tensor from the tuple by concatenation if gradient slices were created.
        """
        if self.is_multiX:
            # For multi-inputs, if grad is required, slice each input tensor along the last dimension.
            Xtuple = tuple(
                xi if not requires_grad
                else tuple(xi[..., j:j+1] for j in range(xi.shape[-1]))
                for xi, requires_grad in zip(X, self.Xrequires_grad)
            )
            Xrecover = tuple(
                xi if not requires_grad else torch.cat(xi, dim=-1)
                for xi, requires_grad in zip(Xtuple, self.Xrequires_grad)
            )
        else:
            # Single input case
            Xtuple = (
                X if not self.Xrequires_grad
                else tuple(X[..., j:j+1] for j in range(X.shape[-1]))
            )
            Xrecover = (
                Xtuple if not self.Xrequires_grad
                else torch.cat(Xtuple, dim=-1)
            )
        return Xtuple, Xrecover


class testBulletin(bulletin):
    """
    testBulletin is used for testing purposes.
    It stores a test dataset, a forward function (netForward), and a loss function (lossFun).
    It can compute output metrics without affecting training.
    """
    def __init__(self, name, testSet, netForward, lossFun, fileSystem=None, Xrequires_grad=False):
        """
        Args:
            name: The name of the test bulletin
            testSet: Dataset for test, an instance of class dataset
            netForward: Forward propagation function of network
            lossFun: Loss function to compute residuals, defined by user
            fileSystem: File name system to handle output files.
            Xrequires_grad: whether to apply requires_grad_=True for components of dataset.X
        """
        super(testBulletin, self).__init__(Xrequires_grad, testSet)
        self.name = name
        self.netForward = netForward
        self.lossFun = lossFun

        # Optionally save the test dataset for reference
        if fileSystem is not None:
            testSet.save(fileSystem.dataSave(self.name))

        # Convert testSet to torch tensors for forward computations
        self.testSet = testSet.transferToTensor()
        self.feedTestData = self.testSet.full_batch()
        self.cur_metric_test = None

    def compilePytorch(self):
        """
        Prepares the test bulletin for PyTorch computations, setting gradients if needed,
        and defines a function to compute losses and metrics for the test data.
        """
        self.applyGrad(self.feedTestData[0])

        def output_losses(batchInput, batchTarget, batchAux):
            # Forward pass
            batchOutput = self.netForward(batchInput)
            # Compute metrics using the loss function (in eval mode)
            metrics = self.lossFun(batchInput, batchTarget, batchAux, batchOutput)
            return metrics

        self.output_losses = output_losses


class trainBulletin(bulletin):
    """
    trainBulletin is used for training purposes.
    It holds a training set, validation set, and logic to handle batching, 
    adaptive sampling (via SAweight and SArefine), and computing training/validation losses.
    """
    def __init__(self, name, dataSet, trainRatio, netForward, lossFun,
                 fileSystem=None, validSet=None, batchSize=0,
                 SAweightOption=None, SArefineOption=None,
                 Xrequires_grad=False):
        """
        Args:
            name: The name of the train bulletin
            dataSet: Dataset for training/validating purposes, an instance of the class dataset
            netForward: Forward propagation function of network
            lossFun: Loss function to compute residuals, defined by user
            fileSystem: File name system to handle output files.
            validSet: A optinal dataset for validation
            SAweightOption: options for self-adaptive weighting
            SArefineOption: options for self-adaptive refining/sampling
            Xrequires_grad: whether to apply requires_grad_=True for components of dataset.X
        """
        
        super(trainBulletin, self).__init__(Xrequires_grad, dataSet)
        self.name = name
        self.netForward = netForward
        self.lossFun = lossFun
        self.fileSystem = fileSystem

        # Split data into train and validation sets if not provided
        if validSet is None:
            trainSet, validSet = dataSet.split(trainRatio)
        else:
            trainSet = dataSet

        self.doValid = True
        if len(validSet) == 0:
            self.doValid = False

        # Save the train/valid sets for reference
        if fileSystem is not None:
            trainSet.save(fileSystem.dataSave(self.name + '_trainSet'))
            validSet.save(fileSystem.dataSave(self.name + '_validSet'))

        self.trainSet = trainSet
        self.validSet = validSet.transferToTensor()

        self.batchSize = batchSize
        self.trainModel = None

        # SAweight and SArefine handle adaptive sampling and weighting
        self.SAweightOption = SAweightOption
        self.SAweight = None
        self.SArefineOption = SArefineOption
        self.SArefine = None

        # Full batch data for validation
        self.feedValidData = self.validSet.full_batch()
        self.feedTrainData = None

        # Current metrics and losses
        self.cur_loss_train = None
        self.cur_metric_train = None
        self.cur_lossVec_train = None
        self.cur_loss_valid = None
        self.cur_metric_valid = None

    def datasetUpdate(self, forceUpdate=False):
        """
        Update the dataset adaptively if the refinement criterion is met.
        This involves SArefine (refinement) and SAweight (adjusting weights).
        Returns True if the dataset was updated, False otherwise.
        """
        data_is_updated = False
        if forceUpdate or self.SArefine.updateCriterion():
            self.SArefine.refineDataset()
            self.SAweight.dataSetUpdate()
            data_is_updated = True
        return data_is_updated

    def nextBatch(self):
        """
        Fetch the next batch of training data and convert it to torch tensors.
        Also sets 'requires_grad' if needed.
        """
        self.batchIndex, self.feedTrainData = self.trainSet.next_batch()
        self.feedTrainData = treeToTensor(self.feedTrainData)
        self.applyGrad(self.feedTrainData[0])

    def compilePytorch(self):
        """
        Prepares PyTorch computations for training, sets up prediction and loss functions.
        Also initializes SAweight and SArefine if not provided.
        """
        # If validation and training sets require gradients (rare), set them here.
        if isinstance(self.trainSet.X, (list, tuple)):
            for i in range(len(self.trainSet.X)):
                self.validSet.X[i].requires_grad_()
        else:
            if self.doValid:
                self.validSet.X.requires_grad_()

        # predict function: run netForward and convert outputs to numpy
        self.predict = lambda x: treeToNumpy(self.netForward(treeToTensor(x)))

        def output_lossVec(batchInput, batchTarget, batchAux):
            # Compute predictions
            batchOutput = self.netForward(batchInput)
            # Compute loss vector and metrics
            lossVec, metrics = self.lossFun(batchInput, batchTarget, batchAux, batchOutput)
            grad.partialDerivative.clearCache()
            return lossVec, metrics

        self.output_lossVec = output_lossVec

        def output_validate_losses(batchInput, batchTarget, batchAux):
            # Compute validation losses and metrics
            lossVec, metrics = self.output_lossVec(batchInput, batchTarget, batchAux)
            loss = torch.mean(torch.square(lossVec))
            return loss, metrics

        self.output_validate_losses = output_validate_losses

        # Initialize SAweight and SArefine strategies
        if self.SAweight is None:
            self.SAweight = SAweight.get(self.SAweightOption)(self, self.SAweightOption)
        self.SArefine = SArefine.get(self.SArefineOption)(self, self.SArefineOption)


class trainModel():
    """
    trainModel handles the entire training process:
    - Holds one or more networks (nets).
    - Manages multiple trainBulletins (and optionally testBulletins).
    - Sets up optimizers, schedules training steps, saves and restores model states, and handles validation & testing.
    """
    def __init__(self, problem, nets, trainBulletins, testBulletins=None, fileSystem=None):
        """
        Args:
            problem: The problem class to be solved
            nets: The networks to be used
            trainBulletins: a list of bulletins for trianing 
            testBulletins: a list of bulletins for testing 
            fileSystem: File name system to handle output files.
        """
        _toList = lambda x: x if isinstance(x, (list, tuple)) else [x]
        self.problem = problem
        self.nets = _toList(nets)
        self.trainBulletins = _toList(trainBulletins)
        # Ensure unique names for each trainBulletin
        if len({t.name for t in self.trainBulletins}) < len(trainBulletins):
            raise Exception("Different train bulletins must have unique names")

        self.doTest = True if testBulletins is not None else False
        if testBulletins is not None:
            self.testBulletins = _toList(testBulletins)

        self.fileSystem = fileSystem
        self.opt = None
        self.lr_scheduler = None

        self.trainState = trainState(self)
        self.step_start = 0
        self.epoch_start = 0

    def init(self):
        """
        Initializes trainModel:
        - Gathers parameters from nets and external variables.
        - Initializes bulletins and compiles PyTorch functions.
        - Sets up the optimizer.
        """
        self.params = {'net': [], 'external_trainable_variables': []}
        self.has_external_trainable_variables = False

        # Collect network parameters
        for net in self.nets:
            self.params['net'].append({'params': nn.parameters.get_net_parameters(net)})

        # Check if there are extra trainable parameters defined outside the nets
        if nn.parameters.has_extra_params():
            self.has_external_trainable_variables = True
            self.params['external_trainable_variables'].append({"params": nn.parameters.get_extra_params()})

        # Initialize train bulletins
        for bulletin in self.trainBulletins:
            bulletin.trainModel = self
            bulletin.init()

        # Initialize test bulletins if any
        if self.doTest:
            for bulletin in self.testBulletins:
                bulletin.init()

        self.compilePytorch()

        # Initialize optimizer based on optimizerOption (set later)
        if not optimizers.is_BFGS(self.optimizerOption['name']):
            params = []
            for value in self.params.values():
                params += value
            self.opt, self.lr_scheduler = optimizers.get(params, self.optimizerOption)
        else:
            # BFGS optimizer requires a different parameter handling
            params = []
            for value in self.params.values():
                for p in value:
                    params += p["params"]
            self.opt, self.lr_scheduler = optimizers.get(params, self.optimizerOption)

    def restartOptimizer(self, onlyBFGS=True):
        """
        Restart the optimizer if needed (e.g., after dataset updates for BFGS).
        If onlyBFGS is True, only restart if the optimizer is BFGS.
        """
        is_BFGS = optimizers.is_BFGS(self.optimizerOption['name'])
        if (not is_BFGS) and onlyBFGS:
            return

        self.params = {'net': [], 'external_trainable_variables': []}
        self.has_external_trainable_variables = False

        # Re-collect parameters
        for net in self.nets:
            self.params['net'].append({'params': nn.parameters.get_net_parameters(net)})
        if nn.parameters.has_extra_params():
            self.has_external_trainable_variables = True
            self.params['external_trainable_variables'].append({"params": nn.parameters.get_extra_params()})

        # Re-initialize optimizer
        params = []
        for value in self.params.values():
            for p in value:
                params += p["params"]
        self.opt, self.lr_scheduler = optimizers.get(params, self.optimizerOption)
        print("Optimizer is restarted")

    def compilePytorch(self):
        """
        Compiles the main training closure used by the optimizer.
        The closure:
        - Zeroes gradients
        - Computes lossVec for each trainBulletin
        - Updates SAweight
        - Assembles total loss and backpropagates
        """
        def closure():
            self.opt.zero_grad()
            for bulletin in self.trainBulletins:
                bulletin.lossVec, bulletin.cur_metric_train = bulletin.output_lossVec(*bulletin.feedTrainData)
                bulletin.lossVec = torch.square(bulletin.lossVec)
                bulletin.SAweight.preUpdate(bulletin.lossVec)

            for bulletin in self.trainBulletins:
                bulletin.SAweight.update()

            total_loss = 0
            for bulletin in self.trainBulletins:
                bulletin.cur_loss_train = bulletin.SAweight.assembly(bulletin.lossVec)
                total_loss += bulletin.cur_loss_train

            total_loss.backward()
            self.total_loss = total_loss

            for bulletin in self.trainBulletins:
                bulletin.SAweight.postUpdate()

            return total_loss

        self.closure = closure

    def trainStep(self):
        """
        Performs a training step by calling the optimizer with the closure.
        If a learning rate scheduler is present, it steps after the optimizer step.
        """
        self.opt.step(self.closure)
        if self.lr_scheduler:
            self.lr_scheduler.step()

    @utils.decorator.timing
    def train(self, optimizerOption, *,
              steps=3000, display_every=10,
              savenet_every=-1,
              valid_every=-1,
              batchSize=None,
              doRestore=False):
        """
        The main training loop:
        - Setup the optimizer and training parameters.
        - Optionally restore from a previous checkpoint.
        - For each step:
            - Fetch next batch
            - Run trainStep
            - Perform validation, testing, saving, logging at intervals
            - Possibly update dataset (adaptive sampling) and restart optimizer if needed
        Args:
            optimizerOption: Options for optimizer
            steps: The total number of training steos
            display_every: every certain number of steps, peform saving, testing, logging
            savenet_every: every certain number of steps, peform saving
            valid_every:   every certain number of steps, peform validation
            batchSize:     batch sizes for all trianing components
            doRestore:     restore from checkpoint if set as true.
        """        
        self.optimizerOption = optimizerOption
        self.use_BFGS = optimizers.is_BFGS(self.optimizerOption['name'])

        def tolist(var, default):
            if var is None:
                var = [default]*len(self.trainBulletins)
            var = var if isinstance(var, (list, tuple)) else [var]*len(self.trainBulletins)
            return var

        # Reset bulletins with possibly updated batchSize
        for i, bulletin in enumerate(self.trainBulletins):
            bulletin.reset(fileSystem=self.fileSystem)
            if tolist(batchSize, None)[i] is not None:
                bulletin.reset(batchSize=tolist(batchSize, None)[i])

        self.stopTraining = False
        self.init()

        if doRestore:
            self.restore()

        # Prepare training (batch sampler, etc.)
        self.batchNum = self.trainState.prepareTrain()
        epochs = steps // self.batchNum
        valid_every = valid_every if valid_every >= 1 else display_every

        for i in range(0, steps):
            self.batchID = i % self.batchNum
            self.trainState.NextBatch()
            self.trainStep()

            if i % valid_every == 0 or i + 1 == steps:
                self.validate()
                updated = self.trainState.update_best()
                if updated:
                    self.save(status='best')

            anyData_is_updated = False
            for bulletin in self.trainBulletins:
                data_is_updated = bulletin.datasetUpdate()
                anyData_is_updated = anyData_is_updated or data_is_updated

            if anyData_is_updated:
                self.restartOptimizer()

            if i % display_every == 0 or i + 1 == steps:
                self.test()
                self.save()
                self.trainState.record(doFlush=(i + 1 == steps))
                self.trainState.printStatus()

            if savenet_every > 0 and i % savenet_every == 0:
                self.save(step=i+self.step_start)

            if self.stopTraining:
                print("Warning: the training process encountered NaNs or large values.")
                print("Reloading the best network and restarting the optimizer.")
                self.restore(onlyNet=True, step=None, status='best')
                self.restartOptimizer()
                # Could force a dataset update or other recovery actions
                # break

        self.step_start += steps
        self.trainState.finalize()
        return

    def test(self):
        """
        Run test evaluations if testBulletins are present.
        Just computes metrics without backprop.
        """
        if not self.doTest:
            return
        for bulletin in self.testBulletins:
            bulletin.cur_metric_test = bulletin.output_losses(*bulletin.feedTestData)

    def validate(self):
        """
        Compute validation losses and metrics for train bulletins.
        If no validation set, use the training set as a fallback.
        """
        for bulletin in self.trainBulletins:
            if bulletin.doValid:
                bulletin.cur_loss_valid, bulletin.cur_metric_valid = bulletin.output_validate_losses(*bulletin.feedValidData)
            else:
                bulletin.cur_loss_valid, bulletin.cur_metric_valid = bulletin.output_validate_losses(*bulletin.feedTrainData)

    def save(self, step=None, status=None):
        """
        Save networks, SAweights, and external parameters.
        If status='best', append '_best' to filenames.
        step can also be included in the filename for checkpointing.
        """
        for net in self.nets:
            name = net.name + '_best' if status == 'best' else net.name
            netfile = self.fileSystem.netfile(name if step is None else name+"_step="+str(step))
            net.savenet(netfile)

        for bulletin in self.trainBulletins:
            bulletin.SAweight.save(step=step)

        params = {'external_trainable_variables': self.params['external_trainable_variables']}
        name = net.name + '_best' if status == 'best' else net.name
        paramsSave_file = self.fileSystem.paramsSave(name if step is None else name+"_step="+str(step))
        torch.save(params, paramsSave_file)
        return

    def restore(self, onlyNet=True, step=None, status=None):
        """
        Restore networks and parameters from disk.
        If onlyNet=True, only restore network weights.
        If status='best', load the '_best' checkpoint.
        """
        for net in self.nets:
            name = net.name + '_best' if status == 'best' else net.name
            netfile = self.fileSystem.netfile(name if step is None else name+"_step="+str(step))
            if os.path.isfile(netfile):
                net.loadnet(netfile)
            else:
                print("warning: no file for net %s" % net.name)

        if onlyNet:
            return

        paramsLoad_file = self.fileSystem.paramsLoad()
        if not os.path.isfile(paramsLoad_file):
            print("warning: no parameters file loaded")
            return
        params = torch.load(paramsLoad_file, map_location=lambda storage, loc: storage)
        self.opt.load_state_dict(params['opt_state'])
        # TODO: load external parameter if needed


class trainState:
    """
    trainState manages the state of training:
    - Keeps track of steps and epochs.
    - Records and saves training/validation/test metrics.
    - Updates the "best" model checkpoint if validation improves.
    - Handles logging and plotting of training history.
    """
    def __init__(self, model):
        self.epoch = -1
        self.step = -1
        self.model = model
        self.feedDataTrain = None
        self.feedDataTest = None

        self.loss_metric_train = {}
        self.loss_metric_test = {}
        self.history_train = []
        self.history_valid = []
        self.history_test = []
        self.history_header = []

        # Best results tracking
        self.best_step = 0
        self.best_loss_metric_train = {}
        self.best_loss_metric_valid = {}

        self.reset_best()

        # Files for saving history
        self.file_history_train = self.model.fileSystem.history("train")
        self.file_history_valid = self.model.fileSystem.history("valid")
        self.file_history_test = self.model.fileSystem.history("test")
        self.fig_history_train = self.model.fileSystem.figHistory("train")
        self.fig_history_valid = self.model.fileSystem.figHistory("valid")
        self.fig_history_test = self.model.fileSystem.figHistory("test")

        self.timeformat = "%Y-%m-%d_%H-%M-%S"

    def prepareTrain(self):
        """
        Prepare for training:
        - Reset best metrics.
        - If using BFGS, warn about batch sizes and SAweight updates.
        - Set batch samplers and return batchNum.
        """
        self.reset_best()
        if self.model.use_BFGS:
            print("warning: batchsize set to 0 for BFGS optimizer")
            print("warning: SAweight updating is blocked for BFGS optimizer")
            for bulletin in self.model.trainBulletins:
                bulletin.SAweight.doUpdate = False

        batchNums = set()
        for bulletin in self.model.trainBulletins:
            batchSize = bulletin.batchSize if not self.model.use_BFGS else 0
            batchNum = bulletin.trainSet.setBatchSampler(batchSize)
            batchNums.add(batchNum)

        # Handle cases where different bulletins have different batch sizes
        if len(batchNums) == 1:
            self.batchNum = batchNums.pop()
        elif len(batchNums) == 2 and min(batchNums) == 1:
            # If one dataset doesn't really need batching (batchNum=1)
            # use the other one which has more batches
            self.batchNum = max(batchNums)
        else:
            self.batchNum = 1

        self.time_strat = time.time()
        return self.batchNum

    def NextBatch(self):
        # Advance one step in training (one batch)
        self.step += 1
        if self.model.batchID == 0:
            self.epoch += 1
        for bulletin in self.model.trainBulletins:
            bulletin.nextBatch()

    def update_best(self):
        """
        Check if validation results improved, update 'best' metrics if so.
        Here, "improvement" is defined as having all losses less than previous best.
        """
        required_update = True
        loss_valid = []
        for bulletin in self.model.trainBulletins:
            loss_valid.append(treeToNumpy(bulletin.cur_loss_valid))

        best_loss_max = max(self.best_loss_metric_valid['loss'])
        for loss in loss_valid:
            if (not np.isfinite(loss)) or loss > best_loss_max:
                required_update = False
                break

        if required_update:
            self.best_step = self.step
            self.best_loss_metric_valid['loss'] = loss_valid
            format_line = "%-8d" * 1 + "   " + ",%11.4e  " * len(loss_valid)
            print(("update best: " + format_line) % (self.step, *loss_valid))
        return required_update

    def reset_best(self):
        # Initialize best metrics to very large values
        self.best_loss_metric_valid['loss'] = [1E10] * len(self.model.trainBulletins)
        self.best_loss_metric_train['loss'] = [1E10] * len(self.model.trainBulletins)

    def getStatusTest(self):
        if not self.model.doTest:
            return None
        name = []
        cur_var = []
        for bulletin in self.model.testBulletins:
            tmp = list(bulletin.cur_metric_test.values())
            name += list(bulletin.cur_metric_test.keys())
            cur_var += tmp
        name = ["step"] + name
        cur_var = [self.step] + treeToNumpy(cur_var)
        return name, cur_var

    def getStatusTrain(self, training=True):
        """
        Get current status (losses and metrics) from the train bulletins.
        training=True returns training losses and metrics,
        training=False returns validation losses and metrics.
        """
        name = []
        cur_var = []
        total_loss = 0
        for bulletin in self.model.trainBulletins:
            if training:
                # First element is loss, rest are metrics
                tmp = [bulletin.cur_loss_train] + list(bulletin.cur_metric_train.values())
                name += list(bulletin.cur_metric_train.keys())
            else:
                tmp = [bulletin.cur_loss_valid] + list(bulletin.cur_metric_valid.values())
                name += list(bulletin.cur_metric_valid.keys())

            total_loss += tmp[0]
            cur_var += tmp[1:]

        name = ["step", "total_loss"] + name
        cur_var = [self.step, total_loss] + cur_var
        cur_var[1:] = treeToNumpy(cur_var[1:])

        # If any NaN or extremely large loss, flag to stop training
        if np.isnan(cur_var).any() or cur_var[1] > 1E12:
            self.model.stopTraining = True

        return name, cur_var

    def formatStatus(self, cur_var):
        # Format status line with time and data
        nint = 1
        nfloat = len(cur_var) - 1
        format_line = "%-8d" * nint + "   " + ",%11.4e  " * nfloat
        dt = time.time() - self.time_strat
        return self.timestamp(dt) + ", " + format_line % tuple(cur_var)

    def formatHeader(self, header):
        # Format header line
        nint = 1
        nfloat = len(header) - 1
        format_line = "%-8s" * nint + "   " + ",%11s  " * nfloat
        return "   time    , " + format_line % tuple(header)

    counter = -1

    def printStatus(self, training=True):
        """
        Print current training/validation status to console periodically.
        Prints headers every 20 lines.
        """
        self.counter += 1
        name, cur_var = self.getStatusTrain(training)
        if self.step == 0 or self.counter % 20 == 0:
            head = self.formatHeader(name)
            print('-' * len(head))
            print(head)
        print(self.formatStatus(cur_var))
        # Also print SAweight and SArefine statuses
        for bulletin in self.model.trainBulletins:
            bulletin.SAweight.printStatus()
            bulletin.SArefine.printStatus()
        return

    def timestamp(self, t, tformat=None):
        # Format a time value in a human-readable string
        if tformat is not None:
            time_array = time.localtime(t)
            timestr = time.strftime(tformat, time_array)
        else:
            d, r = divmod(t, 86400)
            h, r = divmod(r, 3600)
            m, s = divmod(r, 60)
            timestr = '{0}d:{1:02}-{2:02}-{3:02}'.format(int(d), int(h), int(m), int(s))
        return timestr

    def record(self, doFlush=True):
        """
        Records current training/validation/test statuses to files.
        On doFlush=True, writes and clears the in-memory buffers.
        """
        end = 3
        if not self.model.doTest:
            end = 2
        save_vars = [self.history_train, self.history_valid, self.history_test][:end]
        name_vars = [self.getStatusTrain(True), self.getStatusTrain(False), self.getStatusTest()][:end]
        files = [self.file_history_train, self.file_history_valid, self.file_history_test][:end]

        for save_var, (name, cur_var), file in zip(save_vars, name_vars, files):
            save_var.append(cur_var)
            if len(save_var) >= 1 or doFlush:
                self.flush(file, save_var, name)
                save_var.clear()
        return

    def flush(self, file, data, header):
        """
        Writes a batch of data lines into the specified file.
        If new is True (when step=0), writes the header line first.
        """
        if data == []:
            return
        new = (self.step == 0)
        writeMode = "w+" if new else "a+"
        with open(file, writeMode) as f:
            if new:
                f.write(self.formatHeader(header) + "\n")
            for line in data:
                f.write(self.formatStatus(line) + "\n")
        data.clear()
        return

    fig_train = None
    fig_test = None
    fig_valid = None

    def finalize(self):
        """
        Called at the end of training:
        - Generate history plots.
        - Save and plot SAweights.
        """
        from .trainPlot import historyPlot
        self.fig_train = historyPlot("train    history", self.file_history_train, num=self.fig_train, save=self.fig_history_train)
        self.fig_valid = historyPlot("validate history", self.file_history_valid, num=self.fig_valid, save=self.fig_history_valid)
        if self.model.doTest:
            self.fig_test = historyPlot("test     history", self.file_history_test, num=self.fig_test, save=self.fig_history_test)

        for bulletin in self.model.trainBulletins:
            bulletin.SAweight.save()
            bulletin.SAweight.plot()
        pass
