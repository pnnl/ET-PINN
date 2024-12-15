# -*- coding: utf-8 -*-

from cases import case
import matplotlib.pyplot as plt
import pandas as pd
fontsize=14
plt.rcParams.update({"pgf.texsystem": "pdflatex",
                     "text.usetex": True,                                           
                     "font.family": "Times New Roman", })
plt.rc('font',family='Times New Roman')

case_params={"repeatID" :(0,1,2,3,4,5,6,7,8,9)[1],
             "nResi"    :1000,
             "ksi"      :(2,4,8),
             "useSAweight":(False, True, )[:2],
            }


if __name__ == "__main__":
    symbols = ('^','o', 's','D','p','H',)
    lines   = ('-', '--',':', '-.','.',)
    colors  = ('r','g','b','y','c')
    allCases = case(**case_params)  
    
    plt.close('all')
    for casei in allCases:
        ind_ksi=case_params['ksi'].index(casei.ksi)
        ind_SA=case_params['useSAweight'].index(casei.useSAweight)
        figsize=(6,8)     
        historyTrain= casei.fileSystem.history("train")
        historyTest= casei.fileSystem.history("test")
        df_train  = pd.read_csv(historyTrain, sep="\s*,\s*",engine='python')
        df_test   = pd.read_csv(historyTest,  sep="\s*,\s*",engine='python')
        df=pd.merge(df_train,df_test,on='step')
        df.dropna(inplace=True)
        Iter = df['step'].values
        error = df['test'].values
        loss_GE= df['res'].values
        loss_BC= df['BC'].values  
        
        cut = slice(None,None,4)
        plt.figure(1, figsize=figsize)
        plt.subplot(len(case_params['ksi']),1,ind_ksi+1)
        weightLabel='SA weight'if casei.useSAweight else 'No weight'
        plt.semilogy(Iter[cut],error[cut],
                     colors[ind_SA%len(colors)],
                     label= weightLabel)
        plt.ylabel("Error", fontsize=fontsize)
        plt.title(f'k={casei.ksi}')
        if ind_ksi==len(case_params['ksi'])-1:
            plt.legend(loc='best',frameon=False,fontsize=fontsize)
        if ind_ksi==len(case_params['ksi'])-1:
            plt.xlabel('Epoch',fontsize=fontsize)                               

        plt.figure(2, figsize=figsize)
        plt.subplot(len(case_params['ksi']),1,ind_ksi+1)    
            
        print(ind_ksi,ind_SA,lines[0]+colors[ind_SA%len(colors)], lines[1]+colors[ind_SA%len(colors)])
        plt.semilogy(Iter[cut],loss_GE[cut],
                     lines[0]+colors[ind_SA%len(colors)],
                     label='PDE, '+weightLabel)
        plt.semilogy(Iter[cut],loss_BC[cut],
                     lines[1]+colors[ind_SA%len(colors)],
                     label='BC , '+weightLabel)            
        plt.ylabel("Loss", fontsize=fontsize)
        plt.title(f'k={casei.ksi}')
        if ind_ksi==len(case_params['ksi'])-1:
            plt.legend(loc='best',frameon=False,fontsize=fontsize)
        if ind_ksi==len(case_params['ksi'])-1:
            plt.xlabel('Epoch',fontsize=fontsize)   