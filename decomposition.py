import numpy as np
import scipy.sparse as sparse
from gridlod import util, fem, coef, interp, linalg
from gridlod.world import World
import lod_wave
import matplotlib.pyplot as plt

'''
Settings
'''

# fine mesh parameters
fine = 1024
NFine = np.array([fine])
NpFine = np.prod(NFine + 1)
boundaryConditions = np.array([[0, 0]])
world = World(np.array([256]), NFine / np.array([256]), boundaryConditions)
NWorldFine = world.NWorldCoarse * world.NCoarseElement

# fine grid elements and nodes
xt = util.tCoordinates(NFine).flatten()
xp = util.pCoordinates(NFine).flatten()

# time step parameters
tau = 0.01
numTimeSteps = 1000

# ms coefficients
epsA = 2 ** (-4)
epsB = 2 ** (-6)
aFine = (2 - np.sin(2 * np.pi * xt / epsA)) ** (-1)
bFine = (2 - np.cos(2 * np.pi * xt / epsB)) ** (-1)

# plot A(x)
plt.figure('Coefficient')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.subplots_adjust(left=0.09, bottom=0.08, right=0.99, top=0.91, wspace=0.1, hspace=0.2)
plt.tick_params(labelsize=16)
plt.plot(xt,aFine, linewidth=2, label=r'$A_{\epsilon}(x)$')
plt.yticks((0,np.max(aFine)+np.min(aFine)),fontsize=14)
plt.legend(frameon=False,fontsize=22)
plt.title(r'Multiscale coefficient $A_{\epsilon}(x)$', fontsize=24)
plt.grid(True,which="both")
plt.show()

# plot B(x)
plt.figure('Coefficient')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.subplots_adjust(left=0.09, bottom=0.08, right=0.99, top=0.91, wspace=0.1, hspace=0.2)
plt.tick_params(labelsize=16)
plt.plot(xt,bFine, linewidth=2, label=r'$B_{\epsilon}(x)$')
plt.yticks((0,np.max(bFine)+np.min(bFine)),fontsize=14)
plt.legend(frameon=False,fontsize=22)
plt.title(r'Multiscale coefficient $B_{\epsilon}(x)$', fontsize=24)
plt.grid(True,which="both")
plt.show()

# mesh and localization parameters
k = np.inf
N = 4

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
lod = lod_wave.LodWave(b_coef, world, k, IPatchGenerator, a_coef)
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

    # solve non-localized system
    lod = lod_wave.LodWave(b_coef, world, np.inf, IPatchGenerator, a_coef, prev_fs_sol, ms_basis)
    lod.solve_fs_system()

    # store sparse solution
    prev_fs_sol = sparse.csc_matrix(np.array(np.column_stack(lod.fs_list)))
    fs_solutions.append(prev_fs_sol)

'''
Compute v^n and w^n
'''

# initial value
Uo = xpCoarse * (1 - xpCoarse)

# coarse v^(-1) and v^0
V = [Uo]
V.append(Uo)

# fine v^(-1) and v^0
VFine = [ms_basis * Uo]
VFine.append(ms_basis * Uo)

# exact solution
UFine = [ms_basis * Uo]
UFine.append(ms_basis * Uo)

# initial value w^0
Wo = np.zeros(NpFine)
WFine = [Wo]


# compute ms matrices
S = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aFine)
K = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, bFine)
M = fem.assemblePatchMatrix(NWorldFine, world.MLocFine)

SmsFull = ms_basis.T * S * ms_basis
KmsFull = ms_basis.T * K * ms_basis
MmsFull = ms_basis.T * M * ms_basis

free = util.interiorpIndexMap(NWorldCoarse)

SmsFree = SmsFull[free][:, free]
KmsFree = KmsFull[free][:, free]
MmsFree = MmsFull[free][:, free]

RmsFreeList = []
for i in xrange(numTimeSteps):
    n = i + 1

    # linear system
    A = (1. / tau) * SmsFree + KmsFree
    b = (1. / tau) * SmsFree * V[n][free]

    # store ms matrix R^{ms',h}_{H,i,k}
    RmsFull = ms_basis.T * S * fs_solutions[i]
    RmsFree = RmsFull[free][:, free]
    RmsFreeList.append(RmsFree)

    # add sum to linear system
    if i is not 0:
        for j in range(i):
            b += (1. / tau) * RmsFreeList[j] * V[n - 1 - j][free]

    # solve system
    VFree = linalg.linSolve(A, b)
    VFull = np.zeros(NpCoarse)
    VFull[free] = VFree

    # append solution for current time step
    V.append(VFull)
    VFine.append(ms_basis * VFull)

    # evaluate w^n
    w = 0
    if i is not 0:
        for j in range(0, i + 1):
            w += fs_solutions[j] * V[n - j]
    WFine.append(w)

'''
Compute exact solution
'''

# fine free indices
boundaryMap = boundaryConditions == 0
fixedFine = util.boundarypIndexMap(NWorldFine, boundaryMap)
freeFine = np.setdiff1d(np.arange(NpFine), fixedFine)

SFree = S[freeFine][:, freeFine]
KFree = K[freeFine][:, freeFine]

for i in range(numTimeSteps):
    n = i + 1

    # reference system
    A = (1. / tau) * SFree + KFree
    b = (1. / tau) * SFree * UFine[n][freeFine]

    # solve system
    UFineFree = linalg.linSolve(A, b)
    UFineFull = np.zeros(NpFine)
    UFineFull[freeFine] = UFineFree

    # append solution
    UFine.append(UFineFull)

# print L2-error
print np.sqrt(np.dot((UFine[-1] - VFine[-1] - WFine[-1]), (UFine[-1] - VFine[-1] - WFine[-1])))
