# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:30:53 2024

@author: chen096
"""
import numpy as np

Ntrain=1000
Ntest=500
Npara=5
Nic=101
Nt=101

def generateSolution(Ncases, it=1, C=2):
    x=np.linspace(0,1,Nic)[None,None,:,None]
    t=np.linspace(0,1,Nt)[:it][None,None,None,:]
    bn=np.random.randn(Ncases, Npara)[:,:,None,None]
    n=np.arange(1,Npara+1)[None,:,None,None]
    u =  ( bn*np.sin(n*np.pi*x) * np.cos(n*np.pi*np.sqrt(C)*t) ).sum(axis=1)
    return u,x[0,0,:,0],t[0,0,0,:]

# train set
u0,x,t=generateSolution(Ntrain, it=1)
u0=u0[:,:,0]
np.savez_compressed('wave_train.npz', u0=u0,x=x,t=t)
    
# test set
for Cval in [1,2,4,8]:
    u,x,t=generateSolution(Ntest, it=Nt, C=Cval)
    u0=u[:,:,0]
    np.savez_compressed(f'wave_test_C={Cval}.npz', u=u, u0=u0, x=x, t=t)