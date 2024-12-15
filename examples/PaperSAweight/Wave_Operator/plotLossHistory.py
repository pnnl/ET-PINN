# -*- coding: utf-8 -*-

from cases import case, StepAdam
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
fontsize=14
symbols = ('^','o', 's','D','p','H',)
lines   = ('-', '--',':', '-.','.',)
colors  = ('r','g','b','y','c')
plt.rcParams.update({"pgf.texsystem": "pdflatex",
                     "text.usetex": True,                                           
                     "font.family": "Times New Roman", })
plt.rc('font',family='Times New Roman')

SAweights=(None, 'BRDR')
vis = (1,2,4,8)
case_params={"repeatID" :11,
             "vis"      : (1,2,4,8)[1],
             "SAweight" :  (None, 'BRDR')[1:]
            }
cut=slice(None,None,4)
allcases =case(**case_params)
SAweightNames= {None:'Fixed',
                'BRDR':'BRDR',}

def getdata(thiscase):
    historyTrain= thiscase.fileSystem.history("train")
    historyTest= thiscase.fileSystem.history("test")
    outfile    = f"repeatID={thiscase.repeatID}_ksi={vis.index(thiscase.vis)}_useSAweight={SAweights.index(thiscase.SAweight)}.out"
    df_train  = pd.read_csv(historyTrain, sep="\s*,\s*",engine='python')
    df_test   = pd.read_csv(historyTest,  sep="\s*,\s*",engine='python')
    with open(outfile, 'r') as f:
        lines_data=f.readlines()
    scientific_notation_pattern = re.compile(r'[-+]?\d*\.\d+e[-+]?\d+', re.IGNORECASE)
    fun = lambda strs: pd.DataFrame(
                                 [scientific_notation_pattern.findall(line) 
                                 for line in lines_data if line.startswith(strs)]
                                 ).astype(np.float32).values[:,0]
    wIC = fun('weight for IC')
    wIC_t = fun('weight for IC_t')
    wBC   = fun('weight for BC')
    wPDE = fun('weight for pde')*( 2 if thiscase.SAweight=='BRDR100' else 1)
    
    df=pd.merge(df_train,df_test,on='step')
    df.dropna(inplace=True)
    Iter = df['step'].values+1
    error = df['test'].values
    loss_GE= df['res'].values
    loss_IC= df['IC'].values
    loss_BC= df['BC'].values 
    loss_IC_t= df['IC_t'].values 
    return Iter, error, (loss_BC, loss_IC, loss_IC_t, loss_GE),(wBC,wIC, wIC_t,wPDE)
def plotError_on_ax(ax, legend=True, xlabel=True):
    for i,thiscase in enumerate(allcases):
        Iter, error,_,_=getdata(thiscase)
        ax.semilogy(Iter[cut],error[cut],lines[0]+colors[i],
                    label=SAweightNames[thiscase.SAweight])
    ax.set_xlabel('Epoch',fontsize=fontsize)
    ax.get_xaxis().set_visible(xlabel)
    ax.set_ylabel("Error", fontsize=fontsize)
    # ax.set_ylim(bottom=1E-5)
    if legend: ax.legend(loc='best',frameon=False,fontsize=fontsize,ncol=2)
def plotLoss_on_ax(ax,thiscase, names=['PDE', 'BC','BC_x', 'IC'], legend=True, xlabel=True,
                   title=None):
    Iter, _, (loss_IC, loss_BC,loss_BC_x, loss_GE),_=getdata(thiscase)    
    name =SAweightNames[thiscase.SAweight]
    if 'PDE' in names:
        ax.semilogy(Iter[cut],loss_GE[cut],
                     colors[names.index('PDE')]+lines[0],
                     label='PDE, '+name)
    if 'BC' in names:
        ax.semilogy(Iter[cut],loss_BC[cut],
                     colors[names.index('BC')]+lines[0],
                     label='BC, '+name) 
    if 'IC' in names:
        ax.semilogy(Iter[cut],loss_BC_x[cut],
                     colors[names.index('IC')]+lines[0],
                     label='IC, '+name)             
    if 'IC_t' in names:
        ax.semilogy(Iter[cut],loss_IC[cut],
                     colors[names.index('IC_t')]+lines[0],
                     label='IC_t, '+name)           
    if legend: ax.legend(loc='best',frameon=False,fontsize=fontsize,ncol=2)
    ax.set_xlabel('Epoch',fontsize=fontsize)
    ax.get_xaxis().set_visible(xlabel)
    ax.set_ylabel("Loss" if len(names)>1 else names[0]+' loss', fontsize=fontsize)
    if title is not None:
        ax.set_title(title,fontsize=fontsize )   

if __name__ == "__main__":  
    plt.close('all')
    # fig, axs = plt.subplots(1,2, figsize=(10,4))
    # plotError_on_ax(axs[0])
    # plotLoss_on_ax(axs[1])
    # plt.suptitle('Burgers',x=0.5, y=0.92,fontsize=fontsize)

    fig, axs = plt.subplots(1,1, figsize=(8,4))
    thiscase = allcases[0]
    title=None
    plotLoss_on_ax(axs, thiscase, names=['PDE', 'BC','IC', 'IC_t'], 
                   legend=True, 
                   xlabel=True,
                       title=title)  

    # plotWeight_on_ax(axs[4], names=["IC"], legend=False, xlabel=False)
    # plotWeight_on_ax(axs[5], names=['BC'],legend=False, xlabel=True)
    # # plt.suptitle('Burgers',x=0.5, y=0.92,fontsize=fontsize)      
    # plt.subplots_adjust(wspace=0, hspace=0.02)     