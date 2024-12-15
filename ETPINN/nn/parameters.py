import torch.nn as nn
from collections import OrderedDict
import torch

_extra_params=OrderedDict()

def clear():
    _extra_params.clear()
    return 

def has_extra_params():
    if len(_extra_params)>0:
        return True
    else:
        return False
        
def get_extra_params():
    params = [p for _,p in _extra_params.items()]
    return params

def register_parameters(name,init):
    _extra_params[name] = nn.Parameter(torch.as_tensor(init, dtype=torch.defaultReal))
    return _extra_params[name]
 
def get_net_parameters(net):
    return list(net.parameters())