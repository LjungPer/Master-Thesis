import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from gridlod import util, fem, coef, interp
from gridlod.world import World
import lod_wave

'''
Settings
'''

# fine mesh parameters
fine = 1024
NFine = np.array([fine])
NpFine = np.prod(NFine+1)
boundaryConditions = np.array([[0, 0]])
world = World(np.array([256]), NFine/np.array([256]), boundaryConditions)
NWorldFine = world.NWorldCoarse * world.NCoarseElement

# fine grid elements and nodes
xt = util.tCoordinates(NFine).flatten()
xp = util.pCoordinates(NFine).flatten()

# time step parameters
tau = 0.01
numTimeSteps = 100

# ms coefficients
epsA = 2**(-4)
epsB = 2**(-6)
aFine = (2 - np.sin(2 * np.pi * xt / epsA)) ** (-1)
bFine = (2 - np.cos(2 * np.pi * xt / epsB)) ** (-1)

k_0 = np.inf
k_1 = 1000
N = 8

# coarse mesh parameters
NWorldCoarse = np.array([N])
NCoarseElement = NFine / NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

# grid nodes
xpCoarse = util.pCoordinates(NWorldCoarse).flatten()
NpCoarse = np.prod(NWorldCoarse + 1)

'''
Compute multiscale basis
'''

# patch generator and coefficients
IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse,
                                                              NCoarseElement, boundaryConditions)
b_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, bFine)
a_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aFine / tau)

# compute basis correctors
lod = lod_wave.LodWave(b_coef, world, k_0, IPatchGenerator, a_coef)
lod.compute_basis_correctors()

# compute ms basis
basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
basis_correctors = lod.assembleBasisCorrectors()
ms_basis = basis - basis_correctors

'''
Compute finescale system

fs_solutions[i] = {w^i_x}_x
'''

prev_fs_sol = ms_basis
fs_solutions = []
for i in xrange(numTimeSteps):

    # solve system
    lod = lod_wave.LodWave(b_coef, world, k_1, IPatchGenerator, a_coef, prev_fs_sol, ms_basis)
    lod.solve_fs_system(localized=False)

    # store sparse solution
    prev_fs_sol = sparse.csc_matrix(np.array(np.column_stack(lod.fs_list)))
    fs_solutions.append(prev_fs_sol)


n = 0
x = N/2
plt.figure('Corr')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.subplots_adjust(left=0.14, bottom=0.07, right=0.99, top=0.92, wspace=0.1, hspace=0.2)
plt.tick_params(labelsize=14)
plt.plot(xp, fs_solutions[n][:,x].todense(), 'k', label='$w^{%d}_{%d}$' %(n, x), linewidth=1.5)
plt.plot(xpCoarse, 0 * xpCoarse, 'or', label='$x\in \mathcal{N}_H$', markersize=4)
plt.title('Solution correction', fontsize=20)
plt.grid(True, which="both", ls="--")
plt.legend(frameon=False, fontsize=16)
plt.show()

x = N/2
plt.figure('Corr', figsize=(16, 9))
plt.tick_params(labelsize=14)
plt.plot(xp, ms_basis[:,x].todense(), 'k', label='ms', linewidth=2)
plt.plot(xp, basis[:,x].todense(), 'b', label='ms', linewidth=1.5, alpha=0.3)
plt.plot(xp, basis_correctors[:,x].todense(), 'g', label='ms', linewidth=1.5, alpha=0.3)
plt.plot(xpCoarse, 0 * xpCoarse, 'or', label='$x\in \mathcal{N}_H$', markersize=6)
plt.axis('off')
plt.show()









''' Calulcate with lambda for each fine node

new_fs_list = []
for i in range(N+1):

    lams = []
    for j in range(fine + 1):
        if fs_solutions[1][j, i] == 0:
            lams.append(0)
        else:
            lams.append(np.log(fs_solutions[1][j, i] / fs_solutions[0][j, i]) / (-tau))

    #Creates list containing the values of fs_solutions[1][:,8], e.g. fs_solutions[1][64,8] = temp[64]
    temp = []
    for j in range(fine+1):
        temp.append(fs_solutions[0][j,i] * np.exp(-tau*lams[j]))

    new_fs_list.append(temp)
prev_fs_sol = sparse.csc_matrix(np.array(np.column_stack(new_fs_list)))


lam = np.log(fs_solutions[1][fine/2, N/2] / fs_solutions[0][fine/2, N/2]) / (-tau)
'''

''' Calculate and plot exp_fs and fs

temp = []
for i in range(numTimeSteps):
    temp.append(fs_solutions[0] * np.exp(-i*tau*1.15))
n = 90
x = 6
plt.figure('Corr')
plt.subplots_adjust(left=0.10, bottom=0.07, right=0.99, top=0.92, wspace=0.1, hspace=0.2)
plt.tick_params(labelsize=14)
plt.plot(xp, fs_solutions[n][:,x].todense(), 'k', label='$\phi_{%d}$' %(x), linewidth=1.5)
plt.plot(xp, temp[n][:,x].todense(), 'b', label='$\phi_{%d}$' %(x), linewidth=1.5)
plt.plot(xpCoarse, 0 * xpCoarse, 'or', label='$x\in \mathcal{N}_H$', markersize=4)
plt.title('Basis correction', fontsize=20)
plt.grid(True, which="both", ls="--")
plt.legend(frameon=False, fontsize=16)
plt.show()
'''


''' Code to save gif

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
x = 2
inter = 100
fig, ax = plt.subplots(figsize=(16*0.6,9*0.6))
xdata, ydata = [0], [0]
y2data = [0]
y3data = [0]
ln2, = plt.plot([], [], 'r', label='$approx$', animated=True)
ln, = plt.plot([], [], 'b', label='$exact$', animated=True)
ax.grid(color='k', linestyle='-', linewidth=0.3)
plt.legend(frameon=False, fontsize=20)
def init():
    ax.set_xlim(0.35, 0.65)
    ax.set_ylim(-0.001, 0.001)
    return ln,
def update(i):
    if i % 1 == 0:
        xdata[0] = xp
        ydata[0] = fs_solutions[i][:,N/2].todense()
        y2data[0] = exp_fs_solutions[i][:,N/2].todense()
        ln.set_data(xdata, ydata)
        ln2.set_data(xdata, y2data)
        t = tau*(i+1)
        ax.set_title('$t=%.2f$' %(t), fontsize=20)
    return ln, ln2,
ani = FuncAnimation(fig, update, 100,
                    init_func=init, interval=10, blit=True)
ani.save('animation.gif', writer='imagemagick', fps=30)
plt.show()

'''

''' New try with lambdas

lams = np.log(fs_solutions[1]/fs_solutions[0])/(-1*tau)
lams = np.nan_to_num(lams)
lams = np.array(lams)
lams[lams > 5] = 5
lams[lams < 0.01] = 0.01

exp_fs_solutions = []
for i in range(numTimeSteps):
    exp_fs_solutions.append(sparse.csc_matrix(fs_solutions[0].toarray() * np.exp(-i*tau*lams)))
    
'''