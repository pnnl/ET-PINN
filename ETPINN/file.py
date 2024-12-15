# -*- coding: utf-8 -*-
import os

class fileSystem():
    def __init__(self, root='.', subdir='', caseStr='', **kwargs):
        # Initialize a dictionary to store file records: {flag: (file_prefix, scheme)}
        # Each entry maps a descriptive identifier (flag) to a tuple consisting of:
        # - file_prefix: a directory path or file prefix
        # - scheme: file extension or format identifier
        self.record = {}

        # Register some default file paths and flags. The `caseStr` is used
        # to differentiate files for different "cases" or scenarios.
        # 'dataSave'   -> typically used to save data
        # 'paramsLoad' -> used to load net parameters
        # 'paramsSave' -> used to save net parameters
        # 'history'    -> used to store historical logs or results
        # 'figHistory' -> used for figures or images related to the history
        # 'netfile'    -> used for network or model-related files
        self.register("dataSave"  , os.path.join(root, 'data/'       , subdir, 'data' + caseStr)      , "data")
        self.register("paramsLoad", os.path.join(root, 'params/'     , subdir, 'paramsLoad' + caseStr), "params")
        self.register("paramsSave", os.path.join(root, 'params/'     , subdir, 'paramsSave' + caseStr), "params")
        self.register("history"   , os.path.join(root, 'history/'    , subdir, 'history' + caseStr)   , "txt")
        self.register("figHistory", os.path.join(root, 'history/fig/', subdir, 'figHistory' + caseStr), "jpg")
        self.register("netfile"   , os.path.join(root, 'params/net/' , subdir, 'net_' + caseStr)      , "net")

        # Keep track of the default flags so we know which ones are built-in.
        self.defaultFlag = list(self.record.keys())

        # Additional flags can be passed through **kwargs.
        # Each kwarg must be a tuple of (file_prefix, scheme).
        # This allows for flexible extension of the filesystem configuration.
        for flag, info in kwargs.items():
            assert isinstance(info, tuple) and len(info) == 2, "Each additional argument must be a tuple (file_prefix, scheme)."
            file_prefix, scheme = info
            self.register(flag, file_prefix, scheme)

    def register(self, flag, file_prefix, scheme):
        """
        Register a file flag with a given prefix and scheme.
        
        If the flag already exists and is not one of the default flags,
        print a warning indicating that it is being overwritten.
        
        Once registered, the instance will have a method accessible by `self.flag(name="")`
        which returns the full filename: file_prefix + [optional '_'+name] + '.' + scheme.
        """
        # If the flag is already present and is not a default one, warn about overwriting.
        if flag in self.record and flag not in self.defaultFlag:
            print(f"Warning, file flag <{flag}> has been overwritten")
        
        # Ensure that the directory structure exists before storing the file prefix.
        self.prepare(file_prefix)

        # Store the record and create a callable attribute to generate filenames.
        self.record[flag] = (file_prefix, scheme)

        def fun(name=""):
            # This nested function returns a complete filename for the given flag.
            # Optionally, a name can be appended to differentiate between multiple files
            # using the same prefix and scheme.
            return self.__add(flag, name=name)

        # Set the attribute on the instance so one can call, for example:
        # fs.dataSave(name="trial1") -> "/some/path/data_trial1.data"
        setattr(self, flag, fun)

        return self

    def prepare(self, prefix):
        """
        Ensure that the directory path for the given prefix exists.
        If it does not, it will create all necessary parent directories.
        """
        father_path = os.path.dirname(prefix)
        if not os.path.isdir(father_path):
            os.makedirs(father_path)
        return prefix

    def __add(self, flag, name):
        """
        Internal helper method to construct the full file name.
        
        Args:
            flag (str): The identifier for the file type (e.g., 'dataSave').
            name (str): An optional suffix to distinguish different files under the same flag.
        
        Returns:
            str: The full filename, combining the prefix, optional suffix, and scheme.
        """
        add = name if name == "" else "_" + name
        file_prefix, scheme = self.record[flag]
        return file_prefix + add + "." + scheme
