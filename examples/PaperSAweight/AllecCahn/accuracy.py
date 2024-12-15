# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 08:30:41 2022

@author: chen096
"""
import numpy as np
from cases import case
import matplotlib.pyplot as plt
case_params={"repeatID" :(11,12,13,14,15),
             "nResi"    :25600,
             "SAweight" : (None, 'minMax', 'RBA', 'BRDR', 'BRDR100')
            }


allCases = case(**case_params)


# Exact
xyE,uE,_=allCases[0].problem.solve()
print('Exact', xyE.shape)


def getError(up,uE):
    return np.linalg.norm(up-uE)/np.linalg.norm(uE)

def CalculateError(allCases):
    shape = list(allCases.shape)
    Error = np.ones(tuple(shape))
    for casei in allCases:
        ind, sub = casei.getIndex()          
        success = casei.loadnet(status='best')
        if not success:
            print(f"Waning: fail to load net for case {casei.caseStr}")
            continue
  
        upred      = casei.predict(xyE)
        Error[sub] = getError(upred, uE)                       
    return Error
    
# import deepxde.backend as bkd
# filterRatio=bkd.to_numpy(allCases[0].net.filterRatio)
# plt.figure(); 
# plt.hist(filterRatio[0],20); 
# plt.show()

if __name__=="__main__"   :
    Error = CalculateError(allCases)
    print("error:")
    print(allCases.shape)
    print(allCases.getNames())
    print(Error)

    print(Error.mean(axis=0))
    print(Error.std(axis=0))


