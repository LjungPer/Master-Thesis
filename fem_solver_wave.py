# FEM solver for weak form of strongly damped wave equation

import numpy as np
from gridlod import util, fem, linalg

def solveCoarse_fem(world, aFine, bFine, MbFine, U, tau, boundaryConditions, i):
    NWorldCoarse = world.NWorldCoarse
    NWorldFine = world.NWorldCoarse * world.NCoarseElement
    NCoarseElement = world.NCoarseElement

    NpFine = np.prod(NWorldFine + 1)
    NpCoarse = np.prod(NWorldCoarse + 1)

    if MbFine is None:
        MbFine = np.zeros(NpFine)

    boundaryMap = boundaryConditions == 0
    fixedCoarse = util.boundarypIndexMap(NWorldCoarse, boundaryMap=boundaryMap)
    freeCoarse = np.setdiff1d(np.arange(NpCoarse), fixedCoarse)

    AFine = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aFine)
    BFine = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, bFine)
    MFine = fem.assemblePatchMatrix(NWorldFine, world.MLocFine)

    bFine = MFine * MbFine

    basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
    ACoarse = basis.T * (AFine * basis)
    BCoarse = basis.T * (BFine * basis)
    MCoarse = basis.T * (MFine * basis)
    bCoarse = basis.T * bFine

    ACoarseFree = ACoarse[freeCoarse][:, freeCoarse]
    BCoarseFree = BCoarse[freeCoarse][:, freeCoarse]
    MCoarseFree = MCoarse[freeCoarse][:, freeCoarse]
    bCoarseFree = bCoarse[freeCoarse]

    A = (1./tau**2) * MCoarseFree + (1./tau) * ACoarseFree + BCoarseFree
    if i == 0:
        b = bCoarseFree + (1./tau) * ACoarseFree * U[i][freeCoarse] + (1./tau) * MCoarseFree * ((2./tau) * U[i][freeCoarse])
    else:
        b = bCoarseFree + (1./tau) * ACoarseFree * U[i][freeCoarse] + (1./tau) * MCoarseFree * ((2./tau) * U[i][freeCoarse] - (1./tau)*U[i-1][freeCoarse])
    uCoarseFree = linalg.linSolve(A, b)
    uCoarseFull = np.zeros(NpCoarse)
    uCoarseFull[freeCoarse] = uCoarseFree
    uCoarseFull = uCoarseFull

    return uCoarseFull


def solveFine_fem(world, aFine, bFine, f, uSol, tau, boundaryConditions, i):
    NWorldCoarse = world.NWorldCoarse
    NWorldFine = world.NWorldCoarse*world.NCoarseElement
    NpFine = np.prod(NWorldFine+1)
    prevU = uSol[-1]
    if f is None:
        f = np.zeros(NpFine)

    boundaryMap = boundaryConditions == 0
    fixedFine = util.boundarypIndexMap(NWorldFine, boundaryMap=boundaryMap)
    freeFine  = np.setdiff1d(np.arange(NpFine), fixedFine)

    AFine = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aFine)
    BFine = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, bFine)
    MFine = fem.assemblePatchMatrix(NWorldFine, world.MLocFine)

    bFine = MFine * f

    AFineFree = AFine[freeFine][:, freeFine]
    BFineFree = BFine[freeFine][:, freeFine]
    MFineFree = MFine[freeFine][:, freeFine]
    bFineFree = bFine[freeFine]

    A = (1./tau**2) * MFineFree + (1./tau) * AFineFree + BFineFree
    #b = bFineFree + (1./tau) * AFineFree * prevU[freeFine] + (1./tau) * MFineFree * ((1./tau) * prevU[freeFine] + prevV[freeFine])
    if i == 0:
        b = bFineFree + (1. / tau) * AFineFree * prevU[freeFine] + (1. / tau) * MFineFree * (
        (2. / tau) * prevU[freeFine])
    else:
        b = bFineFree + (1. / tau) * AFineFree * prevU[freeFine] + (1. / tau) * MFineFree * (
            (2. / tau) * prevU[freeFine] - (1./tau) * uSol[i-1][freeFine])

    uFineFree = linalg.linSolve(A, b)
    uFineFull = np.zeros(NpFine)
    uFineFull[freeFine] = uFineFree
    uFineFull = uFineFull

    return uFineFull

def solveDampedFine_fem(world, aFine, bFine, f, uSol, tau, boundaryConditions, i):
    NWorldCoarse = world.NWorldCoarse
    NWorldFine = world.NWorldCoarse*world.NCoarseElement
    NpFine = np.prod(NWorldFine+1)
    prevU = uSol[-1]
    if f is None:
        f = np.zeros(NpFine)

    boundaryMap = boundaryConditions == 0
    fixedFine = util.boundarypIndexMap(NWorldFine, boundaryMap=boundaryMap)
    freeFine  = np.setdiff1d(np.arange(NpFine), fixedFine)

    AFine = fem.assemblePatchMatrix(NWorldFine, world.MLocFine, aFine)
    BFine = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, bFine)
    MFine = fem.assemblePatchMatrix(NWorldFine, world.MLocFine)

    bFine = MFine * f

    AFineFree = AFine[freeFine][:, freeFine]
    BFineFree = BFine[freeFine][:, freeFine]
    MFineFree = MFine[freeFine][:, freeFine]
    bFineFree = bFine[freeFine]

    A = (1./tau**2) * MFineFree + (1./tau) * AFineFree + BFineFree
    if i == 0:
        b = bFineFree + (1. / tau) * AFineFree * prevU[freeFine] + (1. / tau) * MFineFree * (
        (2. / tau) * prevU[freeFine])
    else:
        b = bFineFree + (1. / tau) * AFineFree * prevU[freeFine] + (1. / tau) * MFineFree * (
            (2. / tau) * prevU[freeFine] - (1./tau) * uSol[i-1][freeFine])

    uFineFree = linalg.linSolve(A, b)
    uFineFull = np.zeros(NpFine)
    uFineFull[freeFine] = uFineFree
    uFineFull = uFineFull

    return uFineFull
