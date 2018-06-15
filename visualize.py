# This file is part of the master thesis "Variational crimes in the Localized orthogonal decomposition method":
#   https://github.com/TiKeil/Masterthesis-LOD.git
# Copyright holder: Tim Keil 
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MultipleLocator
from matplotlib import cm

from gridlod import util

def drawCoefficient(N, a, greys=False):
    '''
    visualizing the 2d coefficient
    '''
    aCube = a.reshape(N) 
    plt.clf()
    
    if greys:
        cmap = 'Greys'
    else:
        cmap = cm.Greys
            
    plt.imshow(aCube,
               origin='upper', 
               interpolation='none',
               cmap=cmap)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.subplots_adjust(left=0.00,bottom=0.02,right=1,top=0.95,wspace=0.2,hspace=0.2)

def drawCoefficientwt(N, a, greys=False):
    '''
    visualizing the 2d coefficient without a title
    '''
    aCube = a.reshape(N) 
    plt.clf()
    
    if greys:
        cmap = 'Greys'
    else:
        cmap = cm.plasma
            
    plt.imshow(aCube,
               origin='upper', 
               interpolation='none',
               cmap=cmap)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.subplots_adjust(left=0.00,bottom=0.02,right=1,top=0.98,wspace=0.2,hspace=0.2)


def ExtradrawCoefficient(N, a, b, c ,d):
    aCube = a.reshape(N) 
    bCube = b.reshape(N) 
    cCube = c.reshape(N) 
    dCube = d.reshape(N)  
    
    plt.clf()
    
    cmap = plt.cm.viridis
                
    plt.subplot(221)
    plt.title("Original", fontsize=10)
    plt.imshow(aCube,
               origin='upper',   
               interpolation='none',
               cmap=cm.plasma)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.subplots_adjust(left=0.00,bottom=0.02,right=1,top=0.95,wspace=0.00,hspace=0.15)
    
    plt.subplot(222)
    plt.title("Change in value", fontsize=10)
    plt.imshow(bCube,
               origin='upper',  
               interpolation='none',
               cmap=cm.plasma)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.subplots_adjust(left=0.00,bottom=0.02,right=1,top=0.95,wspace=0.00,hspace=0.15)
    
    plt.subplot(223)
    plt.title("Disappearance", fontsize=10)
    plt.imshow(cCube,
               origin='upper', 
               interpolation='none',
               cmap=cm.plasma)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.subplots_adjust(left=0.00,bottom=0.02,right=1,top=0.95,wspace=0.00,hspace=0.15)
    plt.subplot(224)
    plt.title("Shift", fontsize=10)
    plt.imshow(dCube,
               origin='upper', 
               interpolation='none',
               cmap=cm.plasma)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.subplots_adjust(left=0.00,bottom=0.02,right=1,top=0.95,wspace=0.00,hspace=0.15)
    
def d3sol(N, s, String='FinescaleSolution'):
    '''
    3d solution
    ''' 
    fig = plt.figure(String)
    ax = fig.add_subplot(111, projection='3d') 
    
    xp = util.pCoordinates(N)
    X = xp[0:,1:].flatten()
    Y = xp[0:,:1].flatten()
    X = np.unique(X)
    Y = np.unique(Y)
    
    X, Y = np.meshgrid(X, Y)
    
    uLodFine = s.reshape(N+1)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, uLodFine, cmap=cm.coolwarm)
    ymin, ymax = ax.set_zlim()
    ax.set_zticks((ymin,ymax))
    ax.set_zlabel('$z$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.axis('off')
    
    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    
def d3solextra(N, s, fig, ax, ymin, ymax):
    '''
    function for 2d fem example
    '''
    
    xp = util.pCoordinates(N)
    X = xp[0:,1:].flatten()
    Y = xp[0:,:1].flatten()
    X = np.unique(X)
    Y = np.unique(Y)

    X, Y = np.meshgrid(X, Y)

    uLodFine = s.reshape(N+1)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, uLodFine, cmap=cm.coolwarm)
    ymin, ymax = ax.set_zlim()
    ax.set_zlim(ymin,ymax)
    ax.set_zlabel('$z$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.axis('off')

def d3plotter(N, s, String='FinescaleSolution', boundary=None, Blues=None, zmax=None, zmin=None, fig=None, ax=None):

    if fig is None:
        fig = plt.figure(String)
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    
    xp = util.pCoordinates(N)
    X = xp[0:,1:].flatten()
    Y = xp[0:,:1].flatten()
    X = np.unique(X)
    Y = np.unique(Y)
    
    X, Y = np.meshgrid(X, Y)
    
    uLodFine = s.reshape(N+1)
    # Plot the surface.
    if zmin is not None:
        surf = ax.plot_surface(X, Y, uLodFine, cmap=cm.coolwarm, vmin=zmin, vmax=zmax)
    
    if boundary is not None:
        surf = ax.plot_surface(X, Y, uLodFine, cmap=cm.coolwarm, vmin=boundary[0], vmax=boundary[1])
        ax.set_zlim(boundary[0], boundary[1])
    ax.zaxis.set_major_locator(LinearLocator(10))
    if zmin is not None:
        ax.set_zlim(zmin,zmax)
    ax.axis('off')
    ax.grid(False)
    fig.subplots_adjust(left=0.00,bottom=0.00,right=1,top=1,wspace=0.2,hspace=0.2)

def drawPatches(N, a, fig, ax, te):
    aCube = a.reshape(N)
        
    if te == 10:
        ax.plot(5,5,'*w')
    
    major_ticks = np.arange(0, te, 1)                                                
    ax.imshow(aCube, cmap=cm.Blues, extent=[0, te, 0, te])
    
    ax.axis([0, te, 0, te])
    ax.set_xticks(major_ticks)                                                       
    ax.set_yticks(major_ticks)                                                       
    ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    ax.grid(which='both')                                                            
    ax.grid(which='major', linestyle="-", color="black")                                                

def AllshapesSixdrawCoefficient(N, a, b, c , d, e, f):
    aCube = a.reshape(N) 
    bCube = b.reshape(N) 
    cCube = c.reshape(N) 
    dCube = d.reshape(N) 
    eCube = e.reshape(N)
    fCube = f.reshape(N)
    
    plt.clf()
                
    plt.subplot(161)
    plt.title("Shape 1.", fontsize=10)
    plt.imshow(aCube,
               origin='upper',  
               interpolation='none',
               cmap=cm.plasma)
    plt.axis('off')
    plt.subplot(162)
    plt.title("Shape 2.", fontsize=10)
    plt.imshow(bCube,
               origin='upper',   
               interpolation='none',
               cmap=cm.plasma)
    plt.axis('off')
    plt.subplot(163)
    plt.title("Shape 3.", fontsize=10)
    plt.imshow(cCube,
               origin='upper',   
               interpolation='none',
               cmap=cm.plasma)
    plt.axis('off')
    plt.subplot(164)
    plt.title("Shape 4.", fontsize=10)
    plt.imshow(dCube,
               origin='upper',  
               interpolation='none',
               cmap=cm.plasma)
    plt.axis('off')
    plt.subplot(165)
    plt.title("Shape 5.", fontsize=10)
    plt.imshow(eCube,
               origin='upper', 
               interpolation='none',
               cmap=cm.plasma)
    plt.axis('off')
    plt.subplot(166)
    plt.title("Shape 6.", fontsize=10)
    plt.imshow(fCube,
               origin='upper', 
               interpolation='none',
               cmap=cm.plasma)
    plt.axis('off')

def drawCoefficientGrid(N, a, fig, ax, Greys=False):
    aCube = a.reshape(N)  
    te = 16
    major_ticks = np.arange(0, te, 1)
    if Greys:
        ax.imshow(aCube, cmap='Greys', extent=[0, te, 0, te])
    else:
        ax.imshow(aCube, cmap=cm.Blues, extent=[0, te, 0, te])
    ax.axis([0, te, 0, te])
    ax.set_xticks(major_ticks)                                                       
    ax.set_yticks(major_ticks)                                                       
    ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    ax.grid(which='both')                                                            
    ax.grid(which='major', linestyle="-", color="grey")                                                

def drawCoefficientwg(N, a, fig, ax, Greys=False):
    aCube = a.reshape(N) 
    if Greys:
        ax.imshow(aCube, cmap='Greys')
    else:
        ax.imshow(aCube, cmap=cm.Blues)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    
    # or if you want differnet settings for the grids:                               
    fig.subplots_adjust(left=0.02,bottom=0.00,right=0.98,top=1,wspace=0.05,hspace=0.00)
    

def plot_error_indicator(eps,recomputefractionsafe, NWorldCoarse,String):
    eps.sort()
    eps = np.unique(eps)
    es = []
    for i in range(0,np.size(eps)):
        es.append(eps[np.size(eps)-i-1])
    if np.size(recomputefractionsafe) != np.size(es):
        es.append(0)
    plt.figure("Sorted error indicator for " + String)
    elemente = np.arange(np.prod(NWorldCoarse))
    plt.plot(recomputefractionsafe,es,label=String)
    plt.title("Sorted error indicator for " + String)
    plt.ylabel('$e_{u}$',fontsize=20)
    plt.xlabel('Updated correctors in %',fontsize=20)
    plt.subplots_adjust(left=0.15,bottom=0.14,right=0.99,top=0.95,wspace=0.2,hspace=0.2)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tick_params(axis='both', which='minor', labelsize=17)
    #plt.savefig('pic/indi_'+ String + '.pdf')
    
def plot_error_indicator_all(eps, recomputefractionsafe, NWorldCoarse, String):
    eps.sort()
    eps = np.unique(eps)
    es = []
    for i in range(0,np.size(eps)):
        es.append(eps[np.size(eps)-i-1])
    if np.size(recomputefractionsafe) != np.size(es):
        es.append(0)
    plt.figure("Compare sorted error indicators")
    plt.plot(recomputefractionsafe,es,label=String)
    plt.title("Comparison of sorted error indicators")
    plt.ylabel('$e_{u}$',fontsize=20)
    plt.xlabel('Updated correctors in %',fontsize=20)
    plt.legend(loc='upper right',fontsize=17)
    plt.subplots_adjust(left=0.15,bottom=0.14,right=0.99,top=0.95,wspace=0.2,hspace=0.2)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tick_params(axis='both', which='minor', labelsize=17)
    #plt.savefig('pic/compare_indi.pdf')
    
def plot_VCLOD_error(errorbest,errorworst,errorplotinfo,recomputefractionsafe,String):
    size = np.size(recomputefractionsafe)
    iden = []
    for i in range(0,size):
        iden.append(errorworst[0]-((errorworst[0]-errorbest[0])/size) * i)
     
    plt.figure("Errors for " + String)
    errorplotinfo1 = []
    for l in range(0,np.size(errorplotinfo)):
        errorplotinfo1.append(errorplotinfo[l]/errorbest[l])
    
    plt.plot(recomputefractionsafe,errorplotinfo1)
    plt.title("Errors for " + String)
    ymin, ymax = plt.ylim()
    plt.ylabel('$e_{vc}$',fontsize=20)
    plt.xlabel('Updated correctors in %',fontsize=20)
    plt.subplots_adjust(left=0.15,bottom=0.14,right=0.99,top=0.95,wspace=0.2,hspace=0.2)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tick_params(axis='both', which='minor', labelsize=17)
    #plt.savefig('pic/error_'+ String + '.pdf')
    
def plot_VCLOD_error_all(errorbest,errorworst,errorplotinfo,recomputefractionsafe, String):
    size = np.size(recomputefractionsafe)
    iden = []
    for i in range(0,size):
        iden.append(errorworst[0]-((errorworst[0]-errorbest[0])/size) * i)
     
    plt.figure("Errors for " + String)
    errorplotinfo1 = []
    for l in range(0,np.size(errorplotinfo)):
        errorplotinfo1.append(errorplotinfo[l]/errorbest[l])
    
    plt.figure("Compare error")
    plt.plot(recomputefractionsafe,errorplotinfo1, label = String)
    plt.title("VCLOD error comparison")
    ymin, ymax = plt.ylim()
    plt.legend(loc='upper right',fontsize=17)
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tick_params(axis='both', which='minor', labelsize=17)
    plt.subplots_adjust(left=0.15,bottom=0.14,right=0.99,top=0.95,wspace=0.2,hspace=0.2)
    plt.ylabel('$e_{vc}$',fontsize=20)
    plt.xlabel('Updated correctors in %',fontsize=20)
    plt.legend(loc='upper right',fontsize=17) 
    plt.grid()
    #plt.savefig('pic/compare_errors.pdf')