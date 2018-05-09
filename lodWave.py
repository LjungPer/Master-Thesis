import numpy as np
import nodeLod
from gridlod import util
import scipy.sparse as sparse
from copy import deepcopy
import elementLod


class LodWave:
    def __init__(self, b_coef, world, k, IPatchGenerator, a_coef, prev_fs_sol = None, ms_basis = None):
        self.world = world
        self.k = k
        self.IPatchGenerator = IPatchGenerator
        self.a_coef = a_coef
        self.b_coef = b_coef
        self.prev_fs_sol = None
        self.ms_basis = None
        self.fs_list = None
        self.basis_correctors = None
        self.ecList = None
        self.ecListOrigin = None

        if prev_fs_sol is not None:
            self.prev_fs_sol = prev_fs_sol

        if ms_basis is not None:
            self.ms_basis = ms_basis


    def solve_fs_system(self, localized=False):
        '''
        Computes the finescale system and returns {w^n_x}_x, where

        a(w^n_x,  z) + \tau b(w^n_x, z) = a(w^{n-1}_x, z), for all z \in V^f
        '''

        world = self.world
        k = self.k
        IPatchGenerator = self.IPatchGenerator
        b_coef = self.b_coef
        a_coef = self.a_coef
        ms_basis = self.ms_basis
        prev_fs_sol = self.prev_fs_sol

        NpCoarse = np.prod(world.NWorldCoarse + 1)

        if localized:
            fs_list = []
            for node_index in range(NpCoarse):
                ecT = nodeLod.nodeCorrector(world, k, node_index)

                b_patch = b_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
                a_patch = a_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
                IPatch = IPatchGenerator(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

                fs_solution = ecT.compute_localized_node_correction(b_patch, a_patch, IPatch, ms_basis,
                                                                    prev_fs_sol, node_index)
                fs_list.append(fs_solution)
        else:
            ecT = nodeLod.nodeCorrector(world, k, 0)

            b_patch = b_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
            a_patch = a_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
            IPatch = IPatchGenerator(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

            fs_list = ecT.compute_node_correction(b_patch, a_patch, IPatch, prev_fs_sol)

        self.fs_list = fs_list


    def compute_basis_correctors(self):
        '''
        Computes the basis correctors Q_T\lambda_x for every element T.
        '''

        world = self.world
        k = self.k
        IPatchGenerator = self.IPatchGenerator
        b_coef = self.b_coef
        a_coef = self.a_coef

        NtCoarse = np.prod(world.NWorldCoarse)

        ecListOrigin = [None] * NtCoarse
        ecComputeList = []
        
        for element_index in range(NtCoarse):

            iElement = util.convertpIndexToCoordinate(world.NWorldCoarse - 1, element_index)
            ecComputeList.append((element_index, iElement))

        ecTList = []
        for element_index, iElement in ecComputeList:
            ecT = elementLod.elementCorrector(world, k, iElement)

            b_patch = b_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
            a_patch = a_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
            IPatch = IPatchGenerator(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

            ecT.compute_corrector(b_patch, a_patch, IPatch)
            ecTList.append(ecT)

        for ecResult, ecCompute in zip(ecTList, ecComputeList):
            ecListOrigin[ecCompute[0]] = ecResult

        self.ecList = deepcopy(ecListOrigin)


    def assembleBasisCorrectors(self):
        '''
        Constructs {Q\lambda_x}_x by the sum Q\lambda_x = \sum_T Q_T\lambda_x
        '''

        if self.basis_correctors is not None:
            return self.basis_correctors

        assert (self.ecList is not None)

        world = self.world
        NWorldCoarse = world.NWorldCoarse
        NCoarseElement = world.NCoarseElement
        NWorldFine = NWorldCoarse * NCoarseElement

        NtCoarse = np.prod(NWorldCoarse)
        NpCoarse = np.prod(NWorldCoarse + 1)
        NpFine = np.prod(NWorldFine + 1)

        TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
        TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse - 1, NWorldCoarse)

        cols = []
        rows = []
        data = []
        ecList = self.ecList
        for TInd in range(NtCoarse):
            ecT = ecList[TInd]
            assert (ecT is not None)
            assert (hasattr(ecT, 'fsi'))

            NPatchFine = ecT.NPatchCoarse * NCoarseElement
            iPatchWorldFine = ecT.iPatchWorldCoarse * NCoarseElement

            patchpIndexMap = util.lowerLeftpIndexMap(NPatchFine, NWorldFine)
            patchpStartIndex = util.convertpCoordinateToIndex(NWorldFine, iPatchWorldFine)

            colsT = TpStartIndices[TInd] + TpIndexMap
            rowsT = patchpStartIndex + patchpIndexMap
            dataT = np.hstack(ecT.fsi.correctorsList)

            cols.extend(np.repeat(colsT, np.size(rowsT)))
            rows.extend(np.tile(rowsT, np.size(colsT)))
            data.extend(dataT)

        basis_correctors = sparse.csc_matrix((data, (rows, cols)), shape=(NpFine, NpCoarse))

        self.basis_correctors = basis_correctors
        return basis_correctors

