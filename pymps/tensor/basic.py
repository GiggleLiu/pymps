import numpy as np
import pdb
from numpy.linalg import norm
from scipy.linalg import svd, lu

__all__ = ['dpl', 'ldu']


def dpl(A, axis, tol=1e-12):
    '''DeParallelize, MT = A.'''
    # norm over remaining axes
    axes = [i for i in range(A.ndim) if i!=axis]
    axes = tuple(axes)
    nA = norm(A, axis=axes)

    # check for zeros
    nz_inds = np.where(nA > tol)[0]
    AA = np.asarray(A).take(nz_inds, axis=axis)
    nA = nA[nz_inds]
    Au = AA / nA.reshape([-1] + [1] * (AA.ndim - axis - 1))
    Au = Au.view(np.ndarray)

    # Au=AA.mul_axis(1./norm(AA,axis=axes),axis=axis)
    groups = []
    unique_cols = []
    ratios = []
    mask = np.zeros(AA.shape[axis], dtype='bool')
    for ci in range(AA.shape[axis]):
        if mask[ci]:
            continue
        v1 = Au.take([ci], axis=axis)
        remaining_indices = np.where(~mask)[0]
        au = Au.take(remaining_indices, axis=axis)
        overlap = (v1.conj() * au).sum(axis=axes)
        seq_cs = np.where(1 - abs(overlap) < tol)[0]
        eq_cs = remaining_indices[seq_cs]

        groups.append(eq_cs)
        ratios.append(nA[eq_cs] / nA[ci] * overlap[seq_cs])
        unique_cols.append(ci)
        # update mask
        mask[eq_cs] = True
    M = AA.take(unique_cols, axis=axis)
    T = np.zeros((M.shape[axis], A.shape[axis]),
                 dtype='float64' if A.dtype != 'complex128' else 'complex128')
    for i, (cols, ratio) in enumerate(zip(groups, ratios)):
        T[i, nz_inds[cols]] = ratio
    return (M, T) if axis >= 1 else (T.T, M)


def ldu(A):
    '''
    LDU decomposition.

    Args:
        A (1darray): square matrix as input.

    Return:
        tuple: (L, D, U).
    '''
    L, U = lu(A, permute_l=True)
    D = norm(U, axis=1)
    nzmask = D != 0
    U[nzmask] /= D[nzmask, np.newaxis]
    return L, D, U


