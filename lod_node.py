import numpy as np
from gridlod import fem, util, linalg

class schurComplementSolver:
    def __init__(self, NCache=None):
        if NCache is not None:
            self.cholCache = linalg.choleskyCache(NCache)
        else:
            self.cholCache = None

    def solve(self, A, I, bList, fixed, NPatchCoarse=None, NCoarseElement=None):
        return linalg.schurComplementSolve(A, I, bList, fixed, NPatchCoarse, NCoarseElement, self.cholCache)


def ritzProjectionToFinePatchWithGivenSaddleSolver(world,
                                                   iPatchWorldCoarse,
                                                   NPatchCoarse,
                                                   APatchFull,
                                                   bPatchFullList,
                                                   IPatch,
                                                   saddleSolver):
    d = np.size(NPatchCoarse)
    NPatchFine = NPatchCoarse * world.NCoarseElement
    NpFine = np.prod(NPatchFine + 1)

    # Find what patch faces are common to the world faces, and inherit
    # boundary conditions from the world for those. For the other
    # faces, all DoFs fixed (Dirichlet)
    boundaryMapWorld = world.boundaryConditions == 0

    inherit0 = iPatchWorldCoarse == 0
    inherit1 = (iPatchWorldCoarse + NPatchCoarse) == world.NWorldCoarse

    boundaryMap = np.ones([d, 2], dtype='bool')
    boundaryMap[inherit0, 0] = boundaryMapWorld[inherit0, 0]
    boundaryMap[inherit1, 1] = boundaryMapWorld[inherit1, 1]

    # Using schur complement solver for the case when there are no
    # Dirichlet conditions does not work. Fix if necessary.
    assert (np.any(boundaryMap == True))

    fixed = util.boundarypIndexMap(NPatchFine, boundaryMap)

    # projectionsList = saddleSolver.solve(APatch, IPatch, bPatchList)

    projectionsList = saddleSolver.solve(APatchFull, IPatch, bPatchFullList, fixed, NPatchCoarse, world.NCoarseElement)

    return projectionsList

class nodeCorrector:
    def __init__(self, world, k, iNodeWorldCoarse, saddleSolver=None):
        self.k = k
        self.iNodeWorldCoarse = iNodeWorldCoarse
        self.world = world

        d = np.size(iNodeWorldCoarse)
        NWorldCoarse = world.NWorldCoarse
        iPatchWorldCoarse = np.maximum(0, iNodeWorldCoarse - k).astype('int64')
        iEndPatchWorldCoarse = np.minimum(NWorldCoarse - 1, iNodeWorldCoarse + k - 1).astype('int64') + 1
        self.NPatchCoarse = iEndPatchWorldCoarse - iPatchWorldCoarse
        self.iNodePatchCoarse = np.array([iNodeWorldCoarse - iPatchWorldCoarse])
        self.iPatchWorldCoarse = np.array([iPatchWorldCoarse])

        if saddleSolver == None:
            self._saddleSolver = schurComplementSolver()
        else:
            self._saddleSolver = saddleSolver

    @property
    def saddleSolver(self):
        return self._saddleSolver

    @saddleSolver.setter
    def saddleSolver(self, value):
        self._saddleSolver = value

    def compute_localized_node_correction(self, b_patch, a_patch, IPatch, ms_basis, prev_fs_sol, node_index):
        '''Compute the fine correctors over the node based patch.

        Compute the correctors, for all z \in V^f(U(\omega_{x,k})):

        a(\phi_x, z) + \tau b(\phi_x, z) = a(\lambda_x, z) + \tau b(\lambda_x, z)
        '''

        world = self.world
        NCoarseElement = world.NCoarseElement
        NPatchCoarse = self.NPatchCoarse
        iPatchWorldCoarse = self.iPatchWorldCoarse

        NPatchFine = NPatchCoarse * NCoarseElement
        NtFine = np.prod(NPatchFine)
        NpFine = np.prod(NPatchFine + 1)

        iPatchWorldFine = iPatchWorldCoarse * NCoarseElement
        patchpIndexMap = util.lowerLeftpIndexMap(NPatchFine, world.NWorldFine)
        patchpStartIndex = util.convertpCoordinateToIndex(world.NWorldFine, iPatchWorldFine)

        patch_indices = patchpStartIndex + patchpIndexMap

        b_patch = b_patch.aFine
        a_patch = a_patch.aFine

        assert (np.size(b_patch) == NtFine)

        S_patch = fem.assemblePatchMatrix(NPatchFine, world.ALocFine, b_patch)
        K_patch = fem.assemblePatchMatrix(NPatchFine, world.ALocFine, a_patch)
        bPatchFull = np.zeros(NpFine)

        if prev_fs_sol is None:
            bPatchFull -= S_patch * ms_basis.toarray()[:, node_index][patch_indices]
        if prev_fs_sol is not None:
            bPatchFull += K_patch * prev_fs_sol.toarray()[:, node_index][patch_indices]

        fs_patch_solution = ritzProjectionToFinePatchWithGivenSaddleSolver(world,
                                                                        self.iPatchWorldCoarse,
                                                                        NPatchCoarse,
                                                                        S_patch + K_patch,
                                                                        [bPatchFull],
                                                                        IPatch,
                                                                        self.saddleSolver)

        fs_solution = np.zeros(world.NpFine)
        fs_solution[patch_indices] += fs_patch_solution[0]

        return fs_solution

    def compute_node_correction(self, b_patch, a_patch, IPatch, prev_fs_sol):
        '''Compute the fine correctors over full domain

        Computes the correction:

        a(Q^h\lambda_x, z) + \tau b(Q^h\lambda_x, z) = a(\lambda_x, z) + \tau b(\lambda_x, z)
        '''

        world = self.world
        NCoarseElement = world.NCoarseElement
        NPatchCoarse = self.NPatchCoarse

        NPatchFine = NPatchCoarse * NCoarseElement
        NtFine = np.prod(NPatchFine)
        NpCoarse = np.prod(NPatchCoarse + 1)
        NpFine = np.prod(NPatchFine + 1)

        b_patch = b_patch.aFine
        a_patch = a_patch.aFine

        assert (np.size(b_patch) == NtFine)

        SPatchFull = fem.assemblePatchMatrix(NPatchFine, world.ALocFine, b_patch)
        KPatchFull = fem.assemblePatchMatrix(NPatchFine, world.ALocFine, a_patch)
        bPatchFullList = []
        for node_index in range(NpCoarse):

            bPatchFull = np.zeros(NpFine)
            bPatchFull += KPatchFull * prev_fs_sol.toarray()[:,node_index]
            bPatchFullList.append(bPatchFull)

        correctorsList = ritzProjectionToFinePatchWithGivenSaddleSolver(world,
                                                                        self.iPatchWorldCoarse,
                                                                        NPatchCoarse,
                                                                        SPatchFull + KPatchFull,
                                                                        bPatchFullList,
                                                                        IPatch,
                                                                        self.saddleSolver)
        return correctorsList