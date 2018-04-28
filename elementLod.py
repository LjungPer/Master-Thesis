import numpy as np
from gridlod import fem, util, linalg

# Saddle point problem solver
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


class FineScaleInformation:
    def __init__(self, coefficientPatch, correctorsList):
        self.coefficient = coefficientPatch
        self.correctorsList = correctorsList


class CoarseScaleInformation:
    def __init__(self, Kij, Kmsij, muTPrime, correctorFluxTF, basisFluxTF, rCoarse=None):
        self.Kij = Kij
        self.Kmsij = Kmsij
        # self.LTPrimeij = LTPrimeij
        self.muTPrime = muTPrime
        self.rCoarse = rCoarse
        self.correctorFluxTF = correctorFluxTF
        self.basisFluxTF = basisFluxTF

class elementCorrector:
    def __init__(self, world, k, iElementWorldCoarse, saddleSolver=None):
        self.k = k
        self.iElementWorldCoarse = iElementWorldCoarse[:]
        self.world = world

        # Compute (NPatchCoarse, iElementPatchCoarse) from (k, iElementWorldCoarse, NWorldCoarse)
        d = np.size(iElementWorldCoarse)
        NWorldCoarse = world.NWorldCoarse

        iPatchWorldCoarse = np.maximum(0, iElementWorldCoarse - k).astype('int64')
        iEndPatchWorldCoarse = np.minimum(NWorldCoarse - 1, iElementWorldCoarse + k).astype('int64') + 1
        self.NPatchCoarse = iEndPatchWorldCoarse - iPatchWorldCoarse
        self.iElementPatchCoarse = iElementWorldCoarse - iPatchWorldCoarse
        self.iPatchWorldCoarse = iPatchWorldCoarse

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


    def compute_element_corrector(self, b_patch, a_patch, IPatch, ARhsList):
        '''Compute the fine correctors over the patch.

        Compute the correctors

        a(Q_T \lambda_x, z)_{U_K(T)} + \tau b(Q_T \lambda_x, z)_{U_K(T)} = a(\lambda_x, z)_T + \tau b(\lambda_x, z)_T
        '''

        numRhs = len(ARhsList)

        world = self.world
        NCoarseElement = world.NCoarseElement
        NPatchCoarse = self.NPatchCoarse

        NPatchFine = NPatchCoarse * NCoarseElement
        NtFine = np.prod(NPatchFine)
        NpFine = np.prod(NPatchFine + 1)

        b_patch = b_patch.aFine
        a_patch = a_patch.aFine
        assert (np.size(a_patch) == NtFine)

        iElementPatchCoarse = self.iElementPatchCoarse
        elementFinetIndexMap = util.extractElementFine(NPatchCoarse,
                                                       NCoarseElement,
                                                       iElementPatchCoarse,
                                                       extractElements=True)

        elementFinepIndexMap = util.extractElementFine(NPatchCoarse,
                                                       NCoarseElement,
                                                       iElementPatchCoarse,
                                                       extractElements=False)

        SElementFull = fem.assemblePatchMatrix(NCoarseElement, world.ALocFine, b_patch[elementFinetIndexMap])
        KElementFull = fem.assemblePatchMatrix(NCoarseElement, world.ALocFine, a_patch[elementFinetIndexMap])
        SPatchFull = fem.assemblePatchMatrix(NPatchFine, world.ALocFine, b_patch)
        KPatchFull = fem.assemblePatchMatrix(NPatchFine, world.ALocFine, a_patch)

        bPatchFullList = []
        for rhsIndex in range(numRhs):
            bPatchFull = np.zeros(NpFine)
            bPatchFull[elementFinepIndexMap] += SElementFull * ARhsList[rhsIndex]
            bPatchFull[elementFinepIndexMap] += KElementFull * ARhsList[rhsIndex]
            bPatchFullList.append(bPatchFull)

        correctorsList = ritzProjectionToFinePatchWithGivenSaddleSolver(world,
                                                                        self.iPatchWorldCoarse,
                                                                        NPatchCoarse,
                                                                        SPatchFull + KPatchFull,
                                                                        bPatchFullList,
                                                                        IPatch,
                                                                        self.saddleSolver)

        return correctorsList


    def compute_corrector(self, coefficientPatch, corrCoefficientPatch, IPatch):
        '''Compute the fine correctors over the patch.

        Compute the correctors Q_T\lambda_x (T is given by the class instance)

        and store them in the self.fsi object, together with the extracted A|_{U_k(T)}
        '''
        d = np.size(self.NPatchCoarse)
        ARhsList = map(np.squeeze, np.hsplit(self.world.localBasis, 2 ** d))

        correctorsList = self.compute_element_corrector(coefficientPatch, corrCoefficientPatch, IPatch, ARhsList)

        self.fsi = FineScaleInformation(coefficientPatch, correctorsList)
