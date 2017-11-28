'''
Find the Block Matrix structure automatically.
'''

import numpy as np
import scipy.sparse as sps

from . import fblock as flib
from .blockmarker import BlockMarker

__all__ = ['get_blockmarker']


def _pyblockize(indices, indptr):
    '''
    Cluster indices into groups, which forms block diagonal forms.
    The python version.

    Args:
        indices (1D array): The y indices.
        indptr (1D array): The row indicator in c(b)sr_matrix form.

    Returns:
        <BlockMarker>, The collection of block indices.
    '''
    n = len(indptr) - 1
    nitem = len(indices)
    mask = -1 * ones(n, dtype='int32')
    collection = []
    for i in range(n):
        if mask[i] != -1:
            continue

        def enclosure(l, ii):
            l.append(ii)
            x0 = indptr[ii]
            jjs = indices[x0:indptr[ii + 1]]
            for jj in jjs:
                if not (jj in l):
                    enclosure(l, jj)
        li = []
        enclosure(li, i)
        mask[li] = i
        collection.append(li)
    pm = np.concatenate(collection)
    Nr = np.cumsum([0] + [len(ci) for ci in collection])
    return BlockMarker(pm, Nr)


def get_blockmarker(cm, py=False):
    '''
    Cluster indices into groups, which forms block diagonal forms.
    The Fortran version is used unless py is True.

    Args:
        cm (csr_matrix/bsr_matrix): the target matrix.

    Returns:
        <BlockMarker>, The collection of block indices.
    '''
    indices, indptr = cm.indices, cm.indptr
    if py:
        return _pyblockize(indices, indptr)
    pm, Nr, nblock = flib.fblockizeh(indices=indices, indptr=indptr)
    return BlockMarker(Nr=Nr[:nblock + 1]), pm
