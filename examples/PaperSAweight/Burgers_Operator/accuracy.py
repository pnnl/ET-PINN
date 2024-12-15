# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 08:30:41 2022

@author: chen096
"""
import numpy as np
import scipy
from cases import case_params,case
import matplotlib.pyplot as plt
from numpy.random import permutation as perm

case_params={"repeatID" :(11, 12, 13, 14, 15)[:],
             "vis"      : (0.01, 0.001, 0.0001)[:],
             "SAweight":(None, 'BRDR', 'BRDR100')[1]
            }

allCases = case(**case_params)

Nic=P=101
_cache={}

# Exact
def getExact(casei):
    key=str(casei.vis)
    if key in _cache:
        return _cache[key]
    data=scipy.io.loadmat(f'burger_nu_{casei.vis}_test.mat')
    u0 = data['input'] 
    sol= data['output']
    t = np.linspace(0, 1, P)
    x = np.linspace(0, 1, P)
    T, X = np.meshgrid(t, x)
    xt = np.hstack([T.flatten()[:,None], X.flatten()[:,None]])
    utest=np.tile(u0[:,None,:], (1,P*P,1)).reshape(-1,Nic)
    ytest=np.tile(xt[None,:,:], (len(u0),1,1)).reshape(-1,2)
    sE=sol.reshape(-1,1)
    _cache[key]=(sE,ytest,utest)
    return _cache[key]
    

def getError(sp,stest):
    sE=stest.reshape(-1,P*P)
    sp=sp.reshape(-1,P*P)
    err = np.linalg.norm(sp-sE, axis=1)/np.linalg.norm(sE,axis=1)
    print( err.mean(), err.std() )
    return err, err.mean()

def CalculateError(allCases):
    shape = list(allCases.shape) + [500]
    Error = np.ones(tuple(shape))
    for casei in allCases:
        ind, sub = casei.getIndex()  
        success = casei.loadnet()#status='best')
        if not success:
            print(f"Waning: fail to load net for case {casei.caseStr}")
            continue
        #Exact
        sE,ytest,utest = getExact(casei)
        sp = np.zeros_like(sE)
        batch=10000
        i=0
        while i<len(sp):
            sp[i:i+batch]      = casei.predict((ytest[i:i+batch] , utest[i:i+batch] ))
            i+=batch
        Error[sub] = getError(sp, sE)[0]                       
    return Error

if __name__=="__main__"   :
    Error = CalculateError(allCases)
    print("error:")
    print(Error)


    print(Error.mean(axis=-1).mean(axis=0))
    print(Error.mean(axis=-1).std(axis=0))

    np.savez_compressed('error.npz', Error=Error)

