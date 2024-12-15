# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
fontsize=14
plt.rcParams.update({"pgf.texsystem": "pdflatex",
                     "text.usetex": True,                                           
                     "font.family": "Times New Roman", })
plt.rc('font',family='Times New Roman')

nTest=5
# load error for NTK and CK:
def load(name='CK'):
    Error= []
    for i in range(nTest):
        error0=[]
        for nu in ['1em2', '1em3', '1em4']:        
            file = f'output_NTK_CK/{name}/{nu}/errors_{i}.npy'
            error0.append(np.load(file))
        error0 = np.stack( error0, axis=0) 
        Error.append(error0)
    Error = np.stack( Error, axis=0)    
    return Error
def report(error=None, name='CK' ):
    if error is None:
        error = load(name)
    print(f"Error for {name}")
    print('mean:', error.mean(axis=-1).mean(axis=0))
    print('std:',error.mean(axis=-1).std(axis=0))

report(name='fixed')  
# report(name='NTK')        
report(name='NTK_mod')
report(name='CK')        
# report(name='CK_mod')




error_NTK = load('NTK_mod')
error_CK  = load('CK')
error_BRDR=np.load('error.npz')['Error'].squeeze()
report(error=error_BRDR, name='BRDR')

#error_BRDR =np.load('error.npz')['Error'].squeeze()

plt.close('all')

def errorBox_on_ax(ax,error, xlabel=True, ylabel=True,
                  title=None, xtitle=None,ylim=None):
    flierprops = dict(marker='+', markerfacecolor='red', markeredgecolor='red')
    ax.boxplot(list(error), labels=[ f'Run {i}' for i in range(nTest) ],
                flierprops=flierprops)
 
    if xtitle and not xlabel:
        ax.set_xlabel(xtitle,fontsize=fontsize,rotation=0,
                  labelpad=4)   
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([]) 
    elif xtitle and xlabel:
        ax.set_xlabel(xtitle,fontsize=fontsize,rotation=0,
                  labelpad=4)
    elif xlabel:
        pass        
    else:
        ax.get_xaxis().set_visible(False)  
        
    if ylabel:
        ax.set_ylabel("Error",fontsize=fontsize,rotation=90,
                      labelpad=2)
    else:
        ax.get_yaxis().set_visible(False)  
    if title is not None:
        ax.set_title(title,fontsize=fontsize )  
    if ylim:
        ax.set_ylim(ylim)
# def plot_errorBox(ax,error):    
#     flierprops = dict(marker='+', markerfacecolor='red', markeredgecolor='red')
#     plt.figure(figsize=(8,3))
#     plt.boxplot(list(error), labels=[ f'Run {i}' for i in range(nTest) ],
#                 flierprops=flierprops)
#     plt.ylabel('Error',fontsize=fontsize)

    
nu=[r'$\nu$='+str(v) for v in [0.01, 0.001, 0.0001]]
methods=['NTK','CK','BRDR']
errors =[error_NTK, error_CK, error_BRDR]

fig, axs = plt.subplots( 3, 3, figsize=(12,8))  
for i in range(3):
    for j,method in enumerate(methods):
        all_errors=[ errors[j][:,i,:] for j in range(3)]
        all_errors=np.stack(all_errors)
        ylim=0,all_errors.max()*1.05
        print(ylim)
        title = method if i==0 else None
        xlabel = True if i==2 else False
        ylabel = True if j==0 else False
        xtitle = nu[i] if j==1 else None
        errorBox_on_ax(axs[i,j],errors[j][:,i,:],
                      xlabel=xlabel,ylabel=ylabel,
                      title=title,xtitle=xtitle, ylim=ylim)
plt.subplots_adjust(wspace=0.05, hspace=0.2)
plt.suptitle("(b) Burgers' equation",x=0.5, y=0.05,fontsize=fontsize+4)