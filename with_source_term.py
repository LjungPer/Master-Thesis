import numpy as np
import scipy.sparse as sparse
from gridlod import util, fem, coef, interp, linalg, pg
from gridlod.world import World
import lod_wave
import matplotlib.pyplot as plt

'''
Settings
'''

# fine mesh parameters
fine = 256
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
numTimeSteps = 200


# ms coefficients
epsA = 2 ** (-4)
epsB = 2 ** (-6)
aFine = (2 - np.sin(2 * np.pi * xt / epsA)) ** (-1)
bFine = (2 - np.cos(2 * np.pi * xt / epsB)) ** (-1)

# mesh and localization parameters
k = np.inf
NList = [2, 4, 8, 16, 32, 64]

error = []
errorFEM = []
errorLod = []

solutions = []
fem_solutions = []
lod_solutions = []

for N in NList:
    print N

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
        print i

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

    # standard LOD
    Ulod = [Uo]
    Ulod.append(Uo)
    UlodFine = [ms_basis * Uo]
    UlodFine.append(ms_basis * Uo)

    # standard FEM
    UFEM = [Uo]
    UFEM.append(Uo)
    UFEMFine = [ms_basis * Uo]
    UFEMFine.append(ms_basis * Uo)

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

    free = util.interiorpIndexMap(NWorldCoarse)

    SmsFree = SmsFull[free][:, free]
    KmsFree = KmsFull[free][:, free]

    # load vector
    f = np.ones(NpFine)
    LFull = M * f
    LmsFull = ms_basis.T * LFull
    LmsFree = LmsFull[free]

    RmsFreeList = []
    for i in xrange(numTimeSteps):
        n = i + 1

        # linear system
        A = (1. / tau) * SmsFree + KmsFree
        b = LmsFree + (1. / tau) * SmsFree * V[n][free]

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
    Compute standard LOD solution
    '''

    pglod = pg.PetrovGalerkinLOD(world, k, IPatchGenerator, 0)
    pglod.updateCorrectors(b_coef, clearFineQuantities=False)

    pg_basis_correctors = pglod.assembleBasisCorrectors()
    pg_mod_basis = basis - pg_basis_correctors

    SFull = pg_mod_basis.T * S * basis
    KFull = pg_mod_basis.T * K * basis

    SFree = SFull[free][:, free]
    KFree = KFull[free][:, free]

    LFull = basis.T * LFull
    LFree = LFull[free]

    for i in xrange(numTimeSteps):
        n = i + 1

        # standard FEM system
        A = (1. / tau) * SFree + KFree
        b = LFree + (1. / tau) * SFree * Ulod[n][free]

        # solve system
        UlodFree = linalg.linSolve(A, b)
        UlodFull = np.zeros(NpCoarse)
        UlodFull[free] = UlodFree

        # append solution
        Ulod.append(UlodFull)
        UlodFine.append(pg_mod_basis * UlodFull)

    '''
    Compute standard FEM solution
    '''

    # standard FEM matrices
    SFull = basis.T * S * basis
    KFull = basis.T * K * basis

    SFree = SFull[free][:, free]
    KFree = KFull[free][:, free]

    for i in xrange(numTimeSteps):

        n = i + 1

        # standard FEM system
        A = (1. / tau) * SFree + KFree
        b = LFree + (1. / tau) * SFree * UFEM[n][free]

        # solve system
        UFEMFree = linalg.linSolve(A, b)
        UFEMFull = np.zeros(NpCoarse)
        UFEMFull[free] = UFEMFree

        # append solution
        UFEM.append(UFEMFull)
        UFEMFine.append(basis * UFEMFull)



    '''
    Compute exact solution
    '''

    # fine free indices
    boundaryMap = boundaryConditions == 0
    fixedFine = util.boundarypIndexMap(NWorldFine, boundaryMap)
    freeFine = np.setdiff1d(np.arange(NpFine), fixedFine)

    SFree = S[freeFine][:, freeFine]
    KFree = K[freeFine][:, freeFine]

    f = np.ones(NpFine)
    LFineFull = M * f
    LFineFree = LFineFull[freeFine]

    for i in range(numTimeSteps):
        n = i + 1

        # reference system
        A = (1. / tau) * SFree + KFree
        b = LFineFree + (1. / tau) * SFree * UFine[n][freeFine]

        # solve system
        UFineFree = linalg.linSolve(A, b)
        UFineFull = np.zeros(NpFine)
        UFineFull[freeFine] = UFineFree

        # append solution
        UFine.append(UFineFull)

    error.append(np.sqrt(np.dot((UFine[-1] - VFine[-1] - WFine[-1]), (UFine[-1] - VFine[-1] - WFine[-1]))))
    errorFEM.append(np.sqrt(np.dot((UFine[-1] - UFEMFine[-1]), (UFine[-1] - UFEMFine[-1]))))
    errorLod.append(np.sqrt(np.dot((UFine[-1] - UlodFine[-1]), (UFine[-1] - UlodFine[-1]))))

    solutions.append(VFine[-1] + WFine[-1])
    lod_solutions.append(UlodFine[-1])
    fem_solutions.append(UFEMFine[-1])

#Plot solutions
plt.figure('FEM-Solutions')
plt.subplots_adjust(left=0.01, bottom=0.04, right=0.99, top=0.95, wspace=0.1, hspace=0.2)
plt.plot(xp, UFine[-1], 'k', label='ref')
plt.plot(xp, solutions[-2], '--', label='New LOD')
plt.plot(xp, lod_solutions[-2], '--', label='PG-LOD')
plt.plot(xp, fem_solutions[-2], '--', label='FEM')
plt.title('Solutions at $1/H=$ ' + str(NList[-2]), fontsize=18  )
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
                labelleft='off')
plt.grid(True, which="both", ls="--")

plt.legend(frameon=False, fontsize=15)
plt.show()

# plot errors
plt.figure('Error comparison')
plt.subplots_adjust(left=0.09, bottom=0.07, right=0.99, top=0.92, wspace=0.1, hspace=0.2)
plt.tick_params(labelsize=14)
plt.loglog(NList, error, '--s', basex=2, basey=2, label='New LOD')
plt.loglog(NList, errorLod, '--s', basex=2, basey=2, label='PG-LOD $k=\Omega$')
plt.loglog(NList, errorFEM, '--s', basex=2, basey=2, label='FEM')
plt.grid(True, which="both", ls="--")
plt.ylabel('$L^2$-error', fontsize=14)
plt.xlabel('$1/H$', fontsize=14)
plt.title('$L^2$-error at $t=%.2f$' % (numTimeSteps * tau), fontsize=16)
plt.legend(fontsize=12)

plt.show()