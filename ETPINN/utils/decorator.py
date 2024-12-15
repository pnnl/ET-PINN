import sys
import timeit
from functools import wraps

def timing(f):
    """Decorator for measuring the execution time of methods.

    Args:
        f (function): The function to be wrapped.

    Returns:
        function: Wrapped function that measures and prints execution time.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = timeit.default_timer()  # Start timer
        result = f(*args, **kwargs)  # Execute the function
        te = timeit.default_timer()  # End timer
        print("%r took %f s\n" % (f.__name__, te - ts))  # Print execution time
        sys.stdout.flush()  # Flush the output buffer to ensure immediate print
        return result
    return wrapper


def treeFunction(fun, **kwargs):
    """
    Applies a function recursively to tree-like data structures.

    Args:
        fun (function): The function to apply to each element of the structure.
        kwargs: Additional arguments to pass to the function.

    Returns:
        function: A recursive function that applies `fun` to the structure.
    """
    @wraps(fun)
    def Func(*args):
        out = []
        for arg in args:
            if isinstance(arg, dict):  # If the element is a dictionary
                tmp = {key: Func(arg[key]) for key in arg}  # Apply function to each value
                out.append(tmp)
            elif isinstance(arg, (list, tuple)):  # If the element is a list or tuple
                tmp = [Func(iarg) for iarg in arg]  # Apply function to each element
                out.append(type(arg)(tmp))  # Preserve the original type (list/tuple)
            else:  # Base case: Apply the function directly
                out.append(fun(arg, **kwargs) if arg is not None else None)
        return out if len(out) > 1 else out[0]  # Return a single element if the output is scalar
    return Func


