from itertools import product
import abc

class cases_generator(abc.ABC):
    """
    This class cases_generator helps to iterate over all possible combinations of given parameters,
    It is mainly designed for grid tests in machine learning
    """
    def __init__(self, **kwargs):
        # Initialize a dictionary to store parameter names and their corresponding values.
        # Each parameter will either map to a tuple of values or be converted into a one-tuple if it's a single value.
        loopParas = {}
        for key, __values in kwargs.items():
            # If the passed value is already a tuple, use it as-is.
            # Otherwise, wrap the single value in a tuple for consistency.
            if isinstance(__values, tuple):
                loopParas[key] = __values
            else:
                loopParas[key] = (__values,)

        # Store the names of the parameters in a list for indexing purposes.
        self.__names = list(loopParas.keys())
        
        # This generates every combination of parameter values as a sequence of tuples.
        self.__values = list(product(*list(loopParas.values())))
        
        # Keep the original parameter dictionary for reference.
        self.__loopParas = loopParas
        
        # Current index points to which set of parameter combination we are on.
        self.__ind = 0
        
        # Set the initial parameters based on the first combination (index 0).
        self.__setParas(0)
        
        # Store the shape of the parameter grid. For example, if we have 
        # parameters with lengths (3, 2, 4), shape will be (3, 2, 4).
        self.shape = tuple(len(value) for value in loopParas.values())
        
        # Total number of parameter combinations.
        self.length = len(self.__values)

    @abc.abstractmethod
    def init(self):
        # Abstract method that must be implemented by subclasses.
        # This can be used to perform any initialization required for each parameter combination.
        pass

    def __iter__(self):
        # Make the class iterable so we can loop over parameter combinations easily.
        return self

    def __len__(self):
        # Return the total number of parameter combinations.
        return self.length

    def getDim(self, name):
        # Given a parameter name, return its dimension (its index in the parameters list).
        assert name in self.__names
        dim = self.__names.index(name)
        return dim

    def getNames(self):
        # Return the list of parameter names.
        return self.__names

    def getIndex(self):
        # Return the current linear index and the corresponding multidimensional indices (subscripts).
        return self.__ind, self.ind2sub(self.__ind)

    iterID = 0  # This is used to track iteration progress when using the iterator protocol.

    def __next__(self):
        # Implement the iterator protocol. On each call, move to the next parameter combination.
        if self.iterID >= self.length:
            # If we've iterated through all combinations, reset iterID and stop iteration.
            self.iterID = 0
            raise StopIteration
        # Set parameters for the current combination and increment iterID.
        self.__ind = self.iterID
        self.__setParas(self.iterID)
        self.iterID += 1
        return self

    def __getitem__(self, index):
        # Allow indexing into the parameter combinations. For example: instance[5] sets the parameters to combination #5.
        assert 0 <= index < self.length
        self.__setParas(index)
        return self

    def ind2sub(self, ind):
        # Convert a linear index into its corresponding multidimensional index.
        # For example, if shape = (3,2) and ind=3, this might return (1,1).
        sub = []
        for name, value in zip(self.__names, self.__values[ind]):
            # For each parameter, find the index of the current value within its tuple of possible values.
            sub.append(self.__loopParas[name].index(value))
        return tuple(sub)

    def sub2ind(self, sub):
        # Convert a multidimensional index (sub) back into a linear index.
        # For example, given (row, col) -> linear index in a flattened array.
        value = []
        for name, ii in zip(self.__names, sub):
            # Retrieve the actual value from the parameter list using the subscript indices.
            value.append(self.__loopParas[name][ii])
        # Find the tuple of values in __values and return its index.
        ind = self.__values.index(tuple(value))
        return ind

    def __setParas(self, ind):
        # Update the object's attributes to the parameter values corresponding to the given index.
        self.__ind = ind
        # For each parameter name and corresponding value at the specified index, set it as an attribute.
        for name, value in zip(self.__names, self.__values[ind]):
            self.__dict__[name] = value
        # Create a string representation of the current parameter combination.
        self.caseStr = self.__caseStr()
        # Call the init method for any subclass-specific initialization.
        self.init()

    def __caseStr(self):
        # Build a string that describes the current set of parameters in the format "name=value".
        s = ""
        for name, value in zip(self.__names, self.__values[self.__ind]):
            s += "_" + name + "=" + str(value)
        return s[1:]  # Remove the leading underscore.
