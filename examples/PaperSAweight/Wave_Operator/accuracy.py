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

case_params={"repeatID" :(11, 12, 13, 14, 15),
             "vis"      : (2),
             "SAweight":(None, 'BRDR')[1],
            }

allCases = case(**case_params)

Nic=P=101
_cache={}

# Exact
def getExact(casei):
    key=str(casei.vis)
    if key in _cache:
        return _cache[key]
    data=np.load(f'wave_test_C={casei.vis}.npz')
    u0 = data['u0'] 
    sol= data['u']
    t = np.linspace(0, 1, P)
    x = np.linspace(0, 1, P)
    T, X = np.meshgrid(t, x)
    xt = np.hstack([X.flatten()[:,None], T.flatten()[:,None]])
    utest=np.tile(u0[:,None,:], (1,P*P,1)).reshape(-1,Nic)
    ytest=np.tile(xt[None,:,:], (len(u0),1,1)).reshape(-1,2)
    sE=sol.reshape(-1,1)
    _cache[key]=(sE,ytest,utest)
    return _cache[key]
    

def getError(sp,stest):
    sE=stest.reshape(-1,P*P)
    sp=sp.reshape(-1,P*P)
    err = np.linalg.norm(sp-sE, axis=1)/np.linalg.norm(sE,axis=1)
    return err.mean()

def CalculateError(allCases):
    shape = list(allCases.shape)
    Error = np.ones(tuple(shape))
    for casei in allCases:
        ind, sub = casei.getIndex()  
        success = casei.loadnet() #status='best')
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
        Error[sub] = getError(sp, sE)                       
    return Error
    
def Gettime(allCases):
    shape = list(allCases.shape)
    Time = np.ones(tuple(shape))
    Iter = np.ones(tuple(shape))
    import pandas as pd
    import os
    from datetime import datetime
    for casei in allCases:
        ind, sub = casei.getIndex()  
        historyfile = casei.fileSystem.history("train")
        if os.path.isfile(historyfile):
            df  = pd.read_csv(historyfile, header=0, sep="\s*,\s*",engine='python')
            time=df.iloc[:,0].values
            time0=datetime.strptime("00-00-00","%H-%M-%S")
            time1=time0
            for i,t in enumerate(time):
                if i==0: continue
                if t<time[i-1]:
                    time1=datetime.strptime(time[i-1][3:],"%H-%M-%S")
                    break;
            time2=datetime.strptime(time[-1][3:],"%H-%M-%S")
            Time[sub] =(time2-time0).seconds+(time1-time0).seconds                 
            Iter[sub] = int(df.iloc[-1,1])
    return Time,Iter
if __name__=="__main__"   :
    Error = CalculateError(allCases)
    print("error:")
    print(Error)

    print(Error.mean(axis=0))
    print(Error.std(axis=0))

