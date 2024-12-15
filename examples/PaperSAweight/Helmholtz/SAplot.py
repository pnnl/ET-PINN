# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:21:43 2022

@author: chen096
"""

import deepxde as dde
import deepxde.backend as bkd
import numpy as np
from cases import case, case_params, SAweightOption
from cases import StepTotal, StepAdam, StepLbfgs, savenet_everyAdam, savenet_everyLBFGS
import matplotlib.pyplot as plt


allCases = case(**case_params)


thiscase=allCases[0]

#plt.close('all')
# residual
thiscase.loadnet()
dataHF=dde.data.dataSet(file=thiscase.fileSystem.dataSave('pde'+'_trainSet'))
dataBC=dde.data.dataSet(file=thiscase.fileSystem.dataSave('bc'+'_trainSet'))
#dataHF = dde.data.datasetDomain(X='pseudo', physicalDomain=thiscase.problem.physicalDomain, length=int(thiscase.nResi*(1-0)))
pde = dde.trainBulletin('pde',dataHF,  1.0, thiscase.net.forward,  thiscase.problem.pde, fileSystem=thiscase.fileSystem,
                        SAweightOption= SAweightOption)
bc  = dde.trainBulletin('bc',dataBC,  1.0, thiscase.net.forward,  thiscase.problem.BC, fileSystem=thiscase.fileSystem,
                        SAweightOption= SAweightOption)
#dataIC=dde.data.dataSet(file=thiscase.fileSystem.dataSave('ic'+'_trainSet'))
#ic = dde.trainBulletin('ic',dataIC,  1.0, thiscase.net.forward_MF,  thiscase.problem.IC)

view = 'pde'
model= dde.trainModel(thiscase.problem, thiscase.net, [pde if view=='pde' else bc],
                                 fileSystem=thiscase.fileSystem)
#plt.close("all")
for bulletin in model.trainBulletins:
    if bulletin.name != view: continue
    bulletin.init()
    bulletin.trainSet.setBatchSampler(0)
    bulletin.nextBatch()
    
    for epoch in [None]: #list(range(0,StepAdam,savenet_everyAdam))+list(range(StepAdam,StepTotal,savenet_everyLBFGS)):
        try:
            #netfile = model.fileSystem.netfile(model.nets[0].name+ "" if epoch is None else ("_epoch="+str(epoch)) )
            thiscase.loadnet(status='best')
            bulletin.SAweight.restore()
        except Exception as e:
            print("error in loading file:",e)
            break
        lossVec, metric_train = bulletin.output_lossVec(*bulletin.feedTrainData)    
        X=bkd.to_numpy(bulletin.feedTrainData[0])
        lossVec   = np.square(bkd.to_numpy(lossVec))
        SAweights = bulletin.SAweight.SAweights
        #SAweights = np.square(SAweights)

# plt.figure()
# plt.plot(X[:,0], SAweights[:,0],'.')
# plt.show()
# plt.figure()
# plt.hist(SAweights, 100)
# plt.show()
fig, ax = plt.subplots()
cm='rainbow'
cV=np.log10(SAweights[:,0]**2)
vmin=cV.min()
vmax=cV.max()
im=ax.scatter(X[:,0],X[:,1],s=10,c=cV,vmin=vmin,vmax=vmax,cmap=cm,marker='^')
th=np.linspace(0,1,100)*np.pi/2
xR=np.cos(th)
yR=np.sin(th)
plt.plot(xR,yR,'k-')
fig.colorbar(im, ax=ax)
plt.title('SAweight')
plt.show()


fig, ax = plt.subplots()
cm='rainbow'
cV=np.log10(lossVec[:,0])
vmin=cV.min()
vmax=cV.max()
im=ax.scatter(X[:,0],X[:,1],s=10,c=cV,vmin=vmin,vmax=vmax,cmap=cm,marker='^')
plt.plot(xR,yR,'k-')
fig.colorbar(im, ax=ax)
plt.title('Residual')
plt.show()

plt.figure()
plt.hist(SAweights[:,0]**2, 50)
plt.show()

plt.figure()
plt.hist(np.log10(lossVec[:,0]**2), 50)
plt.show()