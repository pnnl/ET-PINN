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
case_params={"repeatID" :11,
             "nResi"    :25600,
             "SAweight" : ('minMax', 'RBA', 'BRDR', 'BRDR100')[:]
            }


problemName='Allen-Cahn'
cm='rainbow'
def scatter_on_ax(ax, thiscase, epoch, xlabel=True, ylabel=True,
                  show_epoch=False, show_title=False,component='pde'):
    dataHF=ETPINN.data.dataSet(file=thiscase.fileSystem.dataSave(component+'_trainSet'))
    pde = ETPINN.trainBulletin(component,dataHF,  1.0, thiscase.net.forward,  thiscase.problem.pde, fileSystem=thiscase.fileSystem,
                            SAweightOption= SAweightOption[thiscase.SAweight],Xrequires_grad=True)
    model= ETPINN.trainModel(thiscase.problem, thiscase.net, [pde],
                                     fileSystem=thiscase.fileSystem)
    pde_bulletin=model.trainBulletins[0]
    pde_bulletin.init()
    pde_bulletin.trainSet.setBatchSampler(0)
    pde_bulletin.nextBatch()
    netfile = model.fileSystem.netfile(model.nets[0].name+ "" if epoch is None else ("_epoch="+str(epoch)) )
    thiscase.loadnet(netfile)
    pde_bulletin.SAweight.restore(epoch)
    X=treeToNumpy(pde_bulletin.feedTrainData[0]).squeeze()
    SAweights = pde_bulletin.SAweight.SAweights
    
    
    cV=SAweights[:,0]**2 if thiscase.SAweight in ['minMax', 'RBA'] else SAweights[:,0]
    cV=np.log10(cV)
    vmin=cV.min()
    vmax=cV.max()
    im=ax.scatter(X[:,0],X[:,1],s=10,c=cV,vmin=vmin,vmax=vmax,cmap=cm,marker='^')
    ax.set_xlabel('x',fontsize=fontsize)
    ax.set_ylabel('t',fontsize=fontsize,rotation=0)
    ax.get_yaxis().set_visible(ylabel)
    epoch = epoch or StepAdam
    # label=str(epoch)
    # if show_epoch and show_title:
    label = "Epoch="+str(epoch)
    if show_epoch:
        ax.set_xlabel(label,fontsize=fontsize,rotation=0,
                      labelpad=2)
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
    else:
        ax.get_xaxis().set_visible(xlabel)      
    if show_title:
        ax.set_title(SAweightNames[thiscase.SAweight],fontsize=fontsize )    
        # ax.set_title(problemName,fontsize=fontsize )    
    plt.colorbar(im, ax=ax)
def contourf_on_ax(ax, thiscase, xlabel=False, ylabel=False,):
    xt,u,(x,t,u2D)     = thiscase.problem.solve(strip_x=10, strip_t=10)   
    im=ax.contourf(x,t,u2D)
    ax.get_xaxis().set_visible(xlabel) 
    ax.get_yaxis().set_visible(ylabel) 
    ax.set_title('Exact solution',fontsize=fontsize )
    plt.colorbar(im, ax=ax)
    return
if __name__=="__main__":
    epochs = [1000,3000,12000,48000] + [None]
    allCases = case(**case_params)
    # fig, axs = plt.subplots(len(epochs),len(allCases), figsize=(12,10))
    # axs = axs.reshape(len(epochs),len(allCases))
    # for i,thiscase in enumerate(allCases):
    #     for j,epoch in enumerate(epochs):
    #         show_epoch = True #if i==0 else False
    #         show_title = True if j==0 else False
    #         scatter_on_ax(axs[j,i],thiscase,epoch,
    #                       xlabel=False,ylabel=False,
    #                       show_epoch=show_epoch,
    #                       show_title=show_title)
    plt.close('all')
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(12,12))
    axs = gridspec.GridSpec(len(epochs)+1,2*len(allCases), wspace=0.2, hspace=0.22)
    ax_top_mid=fig.add_subplot( axs[0,len(allCases)-1:len(allCases)+1] )
    contourf_on_ax(ax_top_mid, allCases[0])
    for i,thiscase in enumerate(allCases):
        for j,epoch in enumerate(epochs):
            show_epoch = True #if i==0 else False
            show_title = True if j==0 else False
            ax = fig.add_subplot( axs[j+1:j+2,2*i:2*i+2] )
            scatter_on_ax(ax,thiscase,epoch,
                          xlabel=False,ylabel=False,
                          show_epoch=show_epoch,
                          show_title=show_title)






