import numpy as np
from collections.abc import Iterable
import pickle
import copy
from .sampler import BatchSampler
from ..utils.functions import treeToNumpy, treeToTensor

# The dataSet class and its subclasses serve as containers and interfaces for
# managing and manipulating sets of data (X, Y, Aux). They allow loading, saving,
# appending, resampling, splitting, and batch-wise iteration.
class dataSet:
    """
    A generic dataset class that stores features (X), targets (Y), and auxiliary data (Aux).
    It supports loading/saving from files, indexing, concatenation, and other basic operations.
    """

    def __init__(self, X=None, Y=None, file=None):
        """
        Initialize the dataset.

        Args:
            X: Feature data. Could be a numpy array, a list/tuple of arrays, or None.
            Y: Target data. Could be a numpy array, None, or a list/tuple where the second element is Aux.
               If Y is a two-element list/tuple, the first element is taken as Y and the second as Aux.
            file: Optional file path. If provided, data is loaded from the file.
        """
        # If a file is given, load data from it
        if file is not None:
            X, Y = self.load(file)

        self.X = X

        # Handle Y and Aux initialization
        if isinstance(Y, (list, tuple)) and len(Y) == 2:
            self.Y, self.Aux = Y
        elif not isinstance(Y, (list, tuple)):
            self.Y = Y
            self.Aux = None
        else:
            raise Exception("Y should be None, an array, or a two-component list/tuple.")

        self.formatCheck()

    def clear(self):
        """
        Clears all data in the dataset, resetting the arrays to empty.
        Useful for freeing memory or starting fresh with the same structure.
        """
        if isinstance(self.X, (list, tuple)):
            self.X = tuple([x[:0] for x in self.X])
        else:
            self.X = self.X[:0]

        self.Y = self.Y[:0] if self.Y is not None else None
        self.Aux = self.Aux[:0] if self.Aux is not None else None
        self.formatCheck()

    def copy(self):
        """
        Create a deep copy of the dataset, duplicating all underlying arrays.

        Returns:
            dataSet: A new dataSet object with copied data.
        """
        newData = copy.copy(self)
        newData.X, newData.Y, newData.Aux = copy.deepcopy(self.X), copy.deepcopy(self.Y), copy.deepcopy(self.Aux)
        return newData

    def append(self, dataSet):
        """
        Append another dataset to the current one along the sample dimension.
        Both datasets must be compatible in structure and shape.

        Args:
            dataSet (dataSet): Another dataset with compatible shapes to append.
        """
        # Concatenate X
        if isinstance(self.X, (list, tuple)):
            self.X = tuple([np.concatenate((self.X[i], dataSet.X[i]), axis=0) for i in range(len(self.X))])
        else:
            self.X = np.concatenate((self.X, dataSet.X), axis=0)

        # Concatenate Y
        if self.Y is not None:
            assert dataSet.Y is not None
            self.Y = np.concatenate((self.Y, dataSet.Y), axis=0)
        else:
            assert dataSet.Y is None

        # Concatenate Aux
        if self.Aux is not None:
            assert dataSet.Aux is not None
            self.Aux = np.concatenate((self.Aux, dataSet.Aux), axis=0)
        else:
            assert dataSet.Aux is None

        self.formatCheck()

    def resample(self, num, random=None, FX=None):
        """
        Resample the dataset. By default, this implementation returns False,
        indicating that no resampling logic is defined here.

        Args:
            num (int): The number of samples to resample.
            random: Randomization method.
            FX: Optional function for generating new samples.

        Returns:
            bool: False by default, as resampling is not implemented at this level.
        """
        return False

    def formatCheck(self):
        """
        Checks the format of X, Y, and Aux arrays to ensure they have consistent lengths.
        Raises assertions if any mismatch is found.
        """
        assert self.X is not None, "X cannot be None."
        if self.Y is not None:
            assert len(self) == len(self.Y), "Y length must match X length."
        if self.Aux is not None:
            assert len(self) == len(self.Aux), "Aux length must match X length."

    def unpack(self):
        """
        Returns all underlying arrays (X, Y, Aux) as a tuple.

        Returns:
            tuple: (X, Y, Aux)
        """
        return self.X, self.Y, self.Aux

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Assumes that the first dimension of X (or X[0] if a list/tuple) gives the sample count.
        """
        if isinstance(self.X, (list, tuple)):
            return int(self.X[0].shape[0])
        else:
            return int(self.X.shape[0])

    def __getitem__(self, index):
        """
        Index the dataset to retrieve a subset of samples.

        Args:
            index: Integer, slice, or array-like indices.

        Returns:
            dataSet: A new dataset containing the requested subset of samples.
        """
        if isinstance(self.X, (list, tuple)):
            X_sub = tuple([x[index, ...] for x in self.X])
        else:
            X_sub = self.X[index, ...]
        Y_sub = self.Y[index, ...] if self.Y is not None else None
        Aux_sub = self.Aux[index, ...] if self.Aux is not None else None
        return dataSet(X_sub, [Y_sub, Aux_sub])

    def setBatchSampler(self, batchSize):
        """
        Initialize a batch sampler for the dataset, enabling iteration over batches.

        Args:
            batchSize (int): Desired batch size.

        Returns:
            int: Number of batches (Nbatch) in an epoch.
        """
        self.length = len(self)
        batchSize = min(self.length, batchSize)
        self.batchSize = batchSize if batchSize > 0 else self.length
        assert 0 < self.batchSize <= self.length
        self.train_sampler = BatchSampler(self.length, batchSize=self.batchSize, shuffle=True)
        Nbatch = self.length // self.batchSize
        return Nbatch

    def save(self, file):
        """
        Save the dataset to a file using pickle.

        Args:
            file (str): File path to save the dataset.
        """
        with open(file, 'wb') as f:
            pickle.dump({"X": self.X,
                         "Y": self.Y,
                         "Aux": self.Aux}, f)

    def load(self, file):
        """
        Load a dataset from a pickle file.

        Args:
            file (str): File path from which to load the dataset.

        Returns:
            tuple: (X, [Y, Aux]) loaded from file.
        """
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data["X"], [data["Y"], data["Aux"]]

    def split(self, ratio=1, return_index=False):
        """
        Split the dataset into two subsets according to a given ratio.

        Args:
            ratio (float): Fraction of data in the first split. (0 < ratio <= 1)
            return_index (bool): If True, also return the indices used for splitting.

        Returns:
            (dataSet, dataSet) or (dataSet, dataSet, indices):
            First set is the chosen fraction of data, second is the remainder.
            If return_index is True, also returns the permutation indices.
        """
        n = len(self)
        n1 = int(n * ratio)
        if n1 == n:
            # If ratio is 1, the second split is empty
            return (self, self[[]]) if not return_index else (self, self[[]], np.arange(n))

        ind = np.random.permutation(n)
        subset1 = self[ind[:n1]]
        subset2 = self[ind[n1:]]
        if return_index:
            return subset1, subset2, ind
        else:
            return subset1, subset2

    def next_batch(self):
        """
        Retrieve the next batch of samples (indices and data) using the configured batch sampler.

        Returns:
            (indices, (X, Y, Aux)): A tuple of the batch indices and the corresponding data.
        """
        if self.batchSize == self.length:  # full batch case
            return list(range(self.length)), (self.X, self.Y, self.Aux)
        indices = self.train_sampler.get_next()
        return indices, self[indices].unpack()

    def full_batch(self):
        """
        Retrieve the full dataset as a single batch.

        Returns:
            (X, Y, Aux): All data in one batch.
        """
        return self.unpack()

    def transferToTensor(self):
        """
        Convert data arrays in X, Y, Aux to tensors (if a suitable backend is available).

        Returns:
            dataSet: A new dataset with tensors instead of numpy arrays.
        """
        newDataSet = self.copy()
        newDataSet.X = treeToTensor(newDataSet.X)
        newDataSet.Y, newDataSet.Aux = treeToTensor(newDataSet.Y, newDataSet.Aux)
        return newDataSet

    def transferToNumpy(self):
        """
        Convert data arrays in X, Y, Aux to numpy arrays (if they are not already).

        Returns:
            dataSet: A new dataset with numpy arrays.
        """
        newDataSet = self.copy()
        newDataSet.X = treeToNumpy(newDataSet.X)
        newDataSet.Y, newDataSet.Aux = treeToNumpy(newDataSet.Y, newDataSet.Aux)
        return newDataSet


def _getX(X, length, sampleFun=None):
    """
    Resolve the input X into a usable array of features.
    If X is callable, treat it as a sampling function.
    If X is a string, use sampleFun to generate samples.
    If X is a numpy array or a list/tuple, use directly.
    """
    if isinstance(X, (list, tuple)):
        X_ = X
    elif callable(X):
        sampleFun = X
        X_ = sampleFun(length, None)
    elif isinstance(X, np.ndarray):
        X_ = X
    elif isinstance(X, str):
        assert sampleFun is not None, "A sampling function must be provided when X is a string."
        assert length > 0 and isinstance(length, int), "Length must be a positive integer."
        X_ = sampleFun(length, X)
    else:
        raise Exception("X should be a list/tuple, a distribution name <str>, a numpy array, or a callable.")
    return X_


def _getY(Y, X):
    """
    Resolve the input Y into usable target data, Y_.
    Y can be:
    - A two-element tuple/list: (Y_data, Aux_data)
    - A callable: Y(X) returns targets
    - An iterable: Convert to a numpy array
    - None: No targets
    """
    if isinstance(Y, (list, tuple)) and len(Y) == 2:
        Y_ = Y
    elif callable(Y):
        Y_ = Y(X)
    elif isinstance(Y, Iterable) and not isinstance(Y, (list, tuple)):
        Y_ = np.asarray(list(Y), np.defaultReal)
    elif Y is None:
        Y_ = None
    else:
        raise Exception(
            "Y should be iterable data, a two-element list/tuple, a callable, or None."
        )
    return Y_


class _physicalDomain:
    """
    An internal helper class for handling physical domain sampling.
    Depending on the domainType ('boundary', 'domain', 'initial'), it sets the appropriate sampling function.
    """

    def __init__(self, physicalDomain=None, domainType='boundary'):
        self.physicalDomain = physicalDomain
        self.domainType = domainType
        self.init()

    def init(self):
        """
        Initialize the sampling function based on domainType.
        If no physicalDomain is provided, sampleFun is set to None.
        """
        sampleFunDict = {
            'boundary': self.boundarySample,
            'domain': self.domainSample,
            'initial': self.initialSample
        }
        if self.physicalDomain is None:
            self.sampleFun = None
        else:
            self.sampleFun = sampleFunDict[self.domainType.lower()]

    def domainSample(self, length, random):
        """
        Sample points inside the domain.
        If random == "uniform", use uniform sampling. Otherwise, use random sampling.
        """
        if random == "uniform":
            return self.physicalDomain.uniform_points(length, boundary=False)
        else:
            return self.physicalDomain.random_points(length, random)

    def boundarySample(self, length, random):
        """
        Sample points on the boundary of the domain.
        If random == "uniform", use uniform boundary sampling. Otherwise, use random boundary sampling.
        """
        if random == "uniform":
            return self.physicalDomain.uniform_boundary_points(length)
        else:
            return self.physicalDomain.random_boundary_points(length, random)

    def initialSample(self, length, random):
        """
        Sample points for initial conditions (e.g., at t=0).
        If random == "uniform", use uniform initial sampling. Otherwise, use random initial sampling.
        """
        # assert isinstance(self.physicalDomain, GeometryXTime)  # Uncomment if applicable
        if random == "uniform":
            return self.physicalDomain.uniform_initial_points(length)
        else:
            return self.physicalDomain.random_initial_points(length, random)


class datasetDomain(dataSet):
    """
    A dataset specifically for domain sampling.
    Given a physical domain, it samples points inside that domain and applies an optional filter function.
    """

    def __init__(self, X=None, Y=None, physicalDomain=None, length=0, filterX=None):
        if filterX is None:
            filterX = lambda x: x
        self.filterX = filterX
        self.FX = Y

        # Create a _physicalDomain with 'domain' type and extract its sample function
        self.physicalDoamin = _physicalDomain(physicalDomain, 'domain')
        self.sampleFun = self.physicalDoamin.sampleFun

        # Resolve X using _getX. If X is callable, it overrides sampleFun.
        X_ = _getX(X, length, self.sampleFun)
        if callable(X):
            self.sampleFun = X

        # Apply the filter function and get Y using _getY
        X_ = filterX(X_)
        Y_ = _getY(Y, X_)

        super(datasetDomain, self).__init__(X=X_, Y=Y_)

    def resample(self, num, random='pseudo', inplace=False):
        """
        Resample the dataset with a new set of points.
        Only works if FX is callable (i.e., Y can be recomputed).

        Args:
            num (int): Number of samples to generate.
            random: Randomization method.
            inplace (bool): If True, modifies this dataset. Otherwise, returns a new dataset.

        Returns:
            datasetDomain: The resampled dataset (either modified in place or a new one).
        """
        handle = self if inplace else self.copy()
        if (not callable(handle.FX)) and (handle.FX is not None):
            raise Exception("This dataSet does not support resampling")

        # Generate new samples using sampleFun and random method
        X_ = _getX(random, num, handle.sampleFun)
        X_ = handle.filterX(X_)
        Y_ = _getY(handle.FX, X_)

        super(datasetDomain, handle).__init__(X=X_, Y=Y_)
        return handle


class datasetBC(dataSet):
    """
    A dataset specifically for boundary conditions.
    It samples points from the boundary of a domain and can compute normal vectors at boundary points if desired.
    """

    def __init__(self, X=None, Y=None, physicalDomain=None, length=0, filterX=None, useBCnormal=False):
        if filterX is None:
            filterX = lambda x: x
        self.filterX = filterX
        self.FX = Y
        self.useBCnormal = useBCnormal

        # Create a _physicalDomain with 'boundary' type and extract its sample function
        self.physicalDoamin = _physicalDomain(physicalDomain, 'boundary')
        self.sampleFun = self.physicalDoamin.sampleFun

        # Resolve X. If X is callable, it becomes the sampleFun.
        X_ = _getX(X, length, self.sampleFun)
        if callable(X):
            self.sampleFun = X

        # Filter X and compute Y
        X_ = filterX(X_)
        Y_ = _getY(Y, X_)

        # Initialize the parent class
        super(datasetBC, self).__init__(X=X_, Y=Y_)

        # If requested, compute and store boundary normals as Aux data
        if useBCnormal:
            BCnormal = self._physicalDomain.boundary_normal(X_)
            self.Aux = BCnormal if self.Aux is None else np.concatenate((self.Aux, BCnormal), axis=1)

    def resample(self, num, random='pseudo', inplace=False):
        """
        Resample boundary points and recompute Y and boundary normals if needed.
        Only works if FX is callable.

        Args:
            num (int): Number of samples to generate.
            random: Randomization method.
            inplace (bool): If True, modifies this dataset in place. Otherwise, returns a new dataset.

        Returns:
            datasetBC: The resampled dataset (in place or new).
        """
        handle = self if inplace else self.copy()
        X_ = _getX(random, num, handle.sampleFun)
        X_ = handle.filterX(X_)

        if (not callable(handle.FX)) and (handle.FX is not None):
            raise Exception("This dataSet does not support resampling")

        Y_ = _getY(self.FX, X_)

        super(datasetBC, handle).__init__(X=X_, Y=Y_)

        # Recompute boundary normals if required
        if handle.useBCnormal:
            BCnormal = handle._physicalDomain.boundary_normal(X_)
            handle.Aux = BCnormal if self.Aux is None else np.concatenate((self.Aux, BCnormal), axis=1)

        return handle


class datasetIC(dataSet):
    """
    A dataset for initial conditions.
    It samples points at the initial domain configuration (e.g., t=0) and applies optional filtering.
    """
    def __init__(self, X=None, Y=None, physicalDomain=None, length=0, filterX=None):
        if filterX is None:
            filterX = lambda x: x
        self.filterX = filterX
        self.FX = Y

        # Create a _physicalDomain with 'initial' type and extract its sample function
        self.physicalDoamin = _physicalDomain(physicalDomain, 'initial')
        self.sampleFun = self.physicalDoamin.sampleFun

        # Resolve X. If X is callable, it becomes the sampleFun.
        X_ = _getX(X, length, self.sampleFun)
        if callable(X):
            self.sampleFun = X

        # Filter X and compute Y
        X_ = filterX(X_)
        Y_ = _getY(Y, X_)

        super(datasetIC, self).__init__(X=X_, Y=Y_)

    def resample(self, num, random='pseudo', inplace=False):
        """
        Resample initial condition points and recompute Y if possible.
        Only works if FX is callable.

        Args:
            num (int): Number of samples to generate.
            random: Randomization method.
            inplace (bool): If True, modifies this dataset in place. Otherwise, returns a new dataset.

        Returns:
            datasetIC: The resampled dataset (in place or new).
        """
        handle = self if inplace else self.copy()

        X_ = _getX(random, num, handle.sampleFun)
        X_ = handle.filterX(X_)

        if (not callable(handle.FX)) and (handle.FX is not None):
            raise Exception("This dataSet does not support resampling")

        Y_ = _getY(self.FX, X_)
        super(datasetIC, handle).__init__(X=X_, Y=Y_)

        return handle
