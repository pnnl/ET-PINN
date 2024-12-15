# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:21:56 2022

@author: chen096
"""


import numpy as np
from cases import case, case_params
import matplotlib.pyplot as plt


allCases = case(**case_params)

thiscase=allCases[0]

plt.close("all")
pltdim=0
# Exact
xyE,uE,(xE,yE,uE2D)= thiscase.problem.solve(Nx=1001, Ny=1001)
fE2D               = thiscase.problem.fsource(xyE.reshape(-1,2)).reshape(xE.shape)
fE2D[0,0] = 0

# Exact
plt.figure()
csE=plt.contourf(xE,yE,uE2D, levels=20)
plt.colorbar(csE);
plt.title("Exact")
# plt.figure()
# plt.scatter(xt[:,0],xt[:,1], c=u.flatten(), )

# Exact
plt.figure()
plt.contourf(xE,yE,fE2D, levels=20)
plt.title("Exact")

r_vec=np.linspace(0.999,1.001,1000);
th_vec =np.linspace(0,np.pi/2,100);
xy=np.zeros((len(r_vec),len(th_vec),2))
xy[:,:,0]=r_vec[:,None]*np.cos(th_vec[None,:])
xy[:,:,1]=r_vec[:,None]*np.sin(th_vec[None,:])
fE2D=thiscase.problem.fsource(xy.reshape(-1,2)).reshape(xy.shape[:-1])
plt.figure()
plt.contourf(r_vec,th_vec,fE2D.T, levels=20)
plt.title("source")


assert thiscase.loadnet()
up = thiscase.predict(xyE)
up2D = up.reshape(uE2D.shape)
plt.figure()
cs=plt.contourf(xE,yE, up2D, levels=csE.levels)
plt.colorbar(cs);
plt.title('NN')

plt.figure()
cs=plt.contourf(xE,yE, up2D-uE2D, levels=csE.levels)
plt.colorbar(cs);
plt.title('NN')


plt.figure()
ix=0
plt.plot(yE[:,ix],uE2D[:,ix],'r-',label='Exact')
plt.plot(yE[:,ix],up2D[:,ix],'b-',label='NN')
plt.legend()

plt.figure()
iy=5
plt.plot(xE[iy,:],uE2D[iy,:],'r-',label='Exact')
plt.plot(xE[iy,:],up2D[iy,:],'b-',label='NN')
plt.legend()





