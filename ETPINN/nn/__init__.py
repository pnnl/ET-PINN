import sys
import torch
import importlib
from . import parameters

def _load_backend(mod_name):
    mod = importlib.import_module(".%s" % mod_name, __name__)
    thismod = sys.modules[__name__]
    for api, obj in mod.__dict__.items():
        setattr(thismod, api, obj)
        
_load_backend('networks')

