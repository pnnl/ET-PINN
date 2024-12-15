# -*- coding: utf-8 -*-

from cases import case,StepAdam
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

SAweights=(None, 'minMax', 'RBA', 'BRDR', 'BRDR100')
case_params={"repeatID" :11,
             "nResi"    :10201,
             "SAweight" : (None, 'minMax', 'RBA', 'BRDR', 'BRDR100')[0:5],             
            }

cut=slice(None,None,5)
allcases =case(**case_params)
SAweightNames= {None:'Fixed', 'minMax':'SA','RBA':'RBA',
                'BRDR':'BRDR', 'BRDR100':'BRDR+'}
def getdata(thiscase):
    historyTrain= thiscase.fileSystem.history("train")
    historyTest= thiscase.fileSystem.history("test")
    outfile    = f"repeatID={thiscase.repeatID}_nResi={thiscase.nResi}_SAweight={SAweights.index(thiscase.SAweight)}.out"
    df_train  = pd.read_csv(historyTrain, sep="\s*,\s*",engine='python')
    df_test   = pd.read_csv(historyTest,  sep="\s*,\s*",engine='python')
    with open(outfile, 'r') as f:
        lines_data=f.readlines()
    scientific_notation_pattern = re.compile(r'[-+]?\d*\.\d+e[-+]?\d+', re.IGNORECASE)
    fun = lambda strs: pd.DataFrame(
                                 [scientific_notation_pattern.findall(line) 
                                 for line in lines_data if line.startswith(strs)]
                                 ).astype(np.float32).values[:,0]
    wBC = fun('weight for BC')*( 100 if thiscase.SAweight=='BRDR100' else 1)
    wPDE = fun('weight for pde')
    df=pd.merge(df_train,df_test,on='step')
    df.dropna(inplace=True)
    Iter = df['step'].values+1
    error = df['test'].values
    loss_GE= df['res'].values
    loss_BC= df['BC'].values 
    return Iter, error, (loss_BC, loss_GE),(wBC,wPDE)
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
def plotLoss_on_ax(ax, names=['PDE', 'BC'], legend=True, xlabel=True):
    for i, thiscase in enumerate(allcases):
        Iter, _, (loss_BC, loss_GE),_=getdata(thiscase)    
        name =SAweightNames[thiscase.SAweight]
        if 'PDE' in names:
            ax.semilogy(Iter[cut],loss_GE[cut],
                         lines[names.index('PDE')]+colors[i],
                         label='PDE, '+name)
        if 'BC' in names:
            ax.semilogy(Iter[cut],loss_BC[cut],
                         lines[names.index('BC')]+colors[i],
                         label='BC, '+name)      
    if legend: ax.legend(loc='best',frameon=False,fontsize=fontsize,ncol=2)
    ax.set_xlabel('Epoch',fontsize=fontsize)
    ax.get_xaxis().set_visible(xlabel)
    ax.set_ylabel("Loss" if len(names)>1 else names[0]+' loss', fontsize=fontsize)
def plotWeight_on_ax(ax, names=['BC'], legend=True, xlabel=True):
    for i, thiscase in enumerate(allcases):
        Iter,_,_,(wBC,wPDE)=getdata(thiscase)   
        rBC = wBC/wPDE
        name =SAweightNames[thiscase.SAweight]
        if 'BC' in names:
            ax.semilogy(Iter[cut],rBC[cut],
                         lines[names.index('BC')]+colors[i],
                         label='BC, '+name)      
    if legend: ax.legend(loc='best',frameon=False,fontsize=fontsize,ncol=2)
    ax.set_xlabel('Epoch',fontsize=fontsize)
    ax.get_xaxis().set_visible(xlabel)
    ylabel=r"$\frac{\overline{w_B}}{\overline{w_R}}$"
    ax.set_ylabel(ylabel, fontsize=fontsize,rotation=0,labelpad=10)
    ax.set_ylim(top=1E3)
    
if __name__ == "__main__":  
    plt.close('all')
    fig, axs = plt.subplots(4,1, figsize=(12,16))
    plotError_on_ax(axs[0], legend=True, xlabel=False)
    plotLoss_on_ax(axs[1], names=['BC'], legend=False, xlabel=False) 
    plotLoss_on_ax(axs[2], names=['PDE'], legend=False, xlabel=False)
    plotWeight_on_ax(axs[3], legend=False, xlabel=True)
    # plt.suptitle('Helmholtz',x=0.5, y=0.92,fontsize=fontsize)      
    plt.subplots_adjust(wspace=0, hspace=0.02)        
    