# -*- coding: utf-8 -*-

import ETPINN
import numpy as np
from cases import case, SAweightOption
from cases import StepTotal, StepAdam, StepLbfgs, savenet_everyAdam, savenet_everyLBFGS
from ETPINN.utils.functions import treeToNumpy,treeToTensor
import matplotlib.pyplot as plt

fontsize=14
symbols = ('^','o', 's','D','p','H',)
lines   = ('-', '--',':', '-.','.',)
colors  = ('r','g','b','y','c')
plt.rcParams.update({"pgf.texsystem": "pdflatex",
                     "text.usetex": True,                                           
                     "font.family": "Times New Roman", })
plt.rc('font',family='Times New Roman')


SAweights=(None, 'minMax', 'RBA', 'BRDR', 'BRDR100')
SAweightNames= {None:'Fixed', 'minMax':'SA','RBA':'RBA',
                'BRDR':'BRDR', 'BRDR100':'BRDR+'}
case_params={"repeatID" :12,
             "nResi"    :10000,
             "vis"      : 0.01,
             "SAweight" : ('minMax', 'RBA','BRDR')
            }



cm='rainbow'
def scatter_on_ax(ax, thiscase, epoch, xlabel=True, ylabel=True,
                  show_epoch=False, show_title=False, component='pde'):
    xt,u,(x,t,u2D)     = thiscase.problem.solve(strip_x=10, strip_t=10)  
    xtTensor=treeToTensor(xt).squeeze()
    xtTensor.requires_grad_(True)
    netfile = thiscase.fileSystem.netfile(thiscase.net.name+ "" if epoch is None else ("_epoch="+str(epoch)) )
    thiscase.loadnet(netfile)              
    up = thiscase.net(xtTensor)
    up2D = treeToNumpy(up.reshape(*u2D.shape))
    res = treeToNumpy(thiscase.problem.pde(xtTensor,None,None,up))[0]
    res = res.reshape(*u2D.shape)
    cV=res**2
    cV=np.log10(cV+1E-30)
    cV = u2D-up2D
    vmin=-0.001 #cV.min()
    vmax=0.001 #cV.max()
    # im=ax.scatter(x,t,cV, cmap='viridis', vmin=vmin,vmax=vmax)
    im=ax.scatter(xt[:,0],xt[:,1],c=cV.flatten(), vmin=vmin,vmax=vmax)
    ax.set_xlabel('x',fontsize=fontsize)
    ax.set_ylabel('t',fontsize=fontsize,rotation=0)
    # ax.get_xaxis().set_visible(xlabel)
    epoch = epoch or StepAdam
    label=str(epoch)
    if show_epoch and show_title:
        label = "Epoch \n\n"+str(epoch)
    if show_epoch:
        ax.set_ylabel(label,fontsize=fontsize,rotation=0,
                      labelpad=20)
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
    else:
        ax.get_yaxis().set_visible(ylabel)        
    if show_title:
        ax.set_title(SAweightNames[thiscase.SAweight],fontsize=fontsize )            
    cbar=plt.colorbar(im, ax=ax)
    



if __name__=="__main__":
    epochs = [None] #[400, 800,1600,3200,6400,12800,25600] + [None]
    allCases = case(**case_params)
    plt.close('all')
    fig, axs = plt.subplots(len(epochs),len(allCases), figsize=(8,16))
    axs = axs.reshape(len(epochs),len(allCases))    
    for i,thiscase in enumerate(allCases):
        for j,epoch in enumerate(epochs):
            show_epoch = True if i==0 else False
            show_title = True if j==0 else False
            scatter_on_ax(axs[j,i],thiscase,epoch,
                          xlabel=False,ylabel=False,
                          show_epoch=show_epoch,
                          show_title=show_title)





