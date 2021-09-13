import matplotlib.pyplot as plt
from utils import *
import numpy as np


'----------------------------------------------------------------------------------------------------------'
def algs_comparison(x,y,w0,dataset,ls_algs,ls_params,min_loss,eps,max_iter,wait,lambda_,ltype,ls_passes,return_losses=True,save=False,path=None,use_labels=False,labels=None,linetypes=None,colors=None):
    
    losses = []
    weights = []
    times = []
    fig, ax = plt.subplots(1,2,figsize=(14,8))

    for k,(alg,params,passes) in enumerate(zip(ls_algs,ls_params,ls_passes)):
        loss,time,weight = wrapper(alg=alg,x=x,y=y,w0=w0,params=params,eps=eps,min_loss=min_loss,return_times=True,return_weights=True,max_iter=max_iter,wait=wait,lambda_=lambda_,ltype=ltype)
        time = np.cumsum(time)
        losses.append(loss)
        times.append(time)
        weights.append(weight)
        if not use_labels:
            label = alg
            ax[0].plot(time,loss-min_loss,label=label)
            ax[1].plot(passes*np.arange(len(loss)),loss-min_loss,label=label)

        else:
            label = labels[k]
            linetype = linetypes[k]
            color = colors[k]
            ax[0].plot(time,loss-min_loss,label=label,linestyle=linetype,color=color)
            ax[1].plot(passes*np.arange(len(loss)),loss-min_loss,label=label,color=color,linestyle=linetype)

        ax[0].set_yscale('log')
        ax[0].set_ylabel(r'$P(w) - P(w^*)$')
        ax[0].set_xlabel(r'$Time\:(s)$')
        ax[0].set_title(r'$Train\:Loss\:vs\:Time:\:{}\:dataset$'.format(dataset))
        
        ax[1].set_yscale('log')
        ax[1].set_ylabel(r'$P(w) - P(w^*)$')
        ax[1].set_xlabel(r'$N.\:\:of\:\:Effective\:\:Calls$')
        ax[1].set_title(r'$Train\:Loss\:vs\:Effective\:Calls:\:{}\:dataset$'.format(dataset))

    ax[0].legend()
    ax[1].legend()
    if save:
        plt.savefig(path)
    if return_losses:
        return losses 
'----------------------------------------------------------------------------------------------------------'

