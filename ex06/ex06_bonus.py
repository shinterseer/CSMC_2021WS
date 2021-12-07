# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 10:48:00 2021

@author: Simon Hinterseer
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plotscript(domain,ax):
    resolution = domain.shape[0]
    h = 1 / (resolution - 1)
    #nodes = np.zeros((resolution,2))
    #values = np.zeros((resolution,1))
    #nodes = np.zeros((resolution,2))
    x = np.zeros(domain.shape)
    y = np.zeros(domain.shape)
    z = domain
    
    min_level = z.min()#*1.1
    max_level = z.max()#*1.1
    levels = np.arange(min_level, max_level, 
                       (max_level - min_level)/20)
    
    for i in range(resolution):
        for j in range(resolution):
            x[i,j] = i * h
            y[i,j] = j * h
    
    qcs = ax.contourf(x,y,z,
                levels,
                cmap=cm.coolwarm,
                origin='lower',
                extend='both',
                )
    #plt.show()
    #plt.contourf(x,y,z)
    return qcs


if __name__ == "__main__":

    # get solution
    n = 100;
    n2 = n*n;
    file = open('solution100.txt')
    solution = file.readlines()
    

    # make 2-dim array out of solution
    domain = np.zeros((n,n))
    for i in range(n2):
        domain[i%n,int(i/n)] = solution[i]
        
       
    #fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
    fig, ax1 = plt.subplots(1,1,figsize=(10,8))

    qcs1 = plotscript(domain,ax1)
    fig.colorbar(qcs1,ax=ax1)
