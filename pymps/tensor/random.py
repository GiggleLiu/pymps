from __future__ import division
import numpy as np
import pdb

from ..spaceconfig import SuperSpaceConfig
from ..toolbox.linear import typed_randn
from ..blockmarker import block_diag, SimpleBMG, BlockMarker, trunc_bm
from .zero_flux import zero_flux_blocks
from .tensor import Tensor, BLabel
from .btensor import BTensor

__all__ = ['random_tensor', 'random_bbtensor', 'random_btensor', 'random_bdmatrix', 'random_zeroflux_tensor']

def random_bbtensor(sites=None, labels=None, nnzblock=100, dtype='complex128'):
    '''
    Generate a random Block-BTensor.

    Parameters:
        :labels: list/None, the labels.
        :sites: int, the number of sites(prop to blocks).
        :nnzblock: int, the approximate number of non zeros entries.

    Return:
        <BTensor>
    '''
    spaceconfig = SuperSpaceConfig([1, 2, 1])
    # get block markers
    if sites is None:
        sites = [2, 3, 4]
    ndim = len(sites)
    if labels is None:
        labels = ['a_%s' % i for i in range(ndim)]
    bms = [SimpleBMG(spaceconfig=spaceconfig, qstring='QM').random_bm(
        nsite=sites[i]) for i in range(ndim)]

    # get unique nzblocks.
    nbs = [bm.nblock for bm in bms]
    nzblocks = np.concatenate([np.random.randint(0, nb, nnzblock)[
                              :, np.newaxis] for nb in nbs], axis=1)
    b = np.ascontiguousarray(nzblocks).view(
        np.dtype((np.void, nzblocks.dtype.itemsize * nzblocks.shape[1])))
    idx = np.unique(b, return_index=True)[1]
    nzblocks = nzblocks[idx]

    # get entries
    data = dict((tuple(blk), typed_randn(dtype,
        [bm.blocksize(blki) for blki, bm in zip(blk, bms)])) for blk in nzblocks)

    # generate BTensor
    return BTensor(data, labels=[BLabel(lb, bm) for lb, bm in zip(labels, bms)])


def random_tensor(shape=None, labels=None, dtype='complex128'):
    '''
    Generate a random Tensor.

    Parameters:
        :shape: the shape of tensor.
        :labels: the labels of axes.

    Return:
        <Tensor>
    '''
    if shape is None:
        shape = (20, 30, 40)
    ndim = len(shape)
    if labels is None:
        labels = ['a_%s' % i for i in range(ndim)]
    data = typed_randn(dtype, shape)

    # generate Tensor
    return Tensor(data, labels=labels)


def random_btensor(bms, label_strs=None, fill_rate=0.2, dtype='complex128'):
    '''
    Generate a random Block Dense Tensor.

    Parameters:
        :bms: list, the block markers.
        :label_strs: list/None, the labels.
        :fill_rate: propotion of the number of filled blocks.

    Return:
        <Tensor>
    '''
    # get block markers
    ndim = len(bms)
    if label_strs is None:
        labels = [BLabel('a_%s' % i, bm) for i, bm in enumerate(bms)]
    else:
        labels = [BLabel(lb, bm) for lb, bm in zip(label_strs, bms)]
    ts = Tensor(np.zeros([bm.N for bm in bms]), labels=labels)
    # insert datas
    nnzblock = int(fill_rate * np.prod([bm.nblock for bm in bms]))
    for i in range(nnzblock):
        target = ts[tuple(
            [bm.get_slice(np.random.randint(0, bm.nblock)) for bm in bms])]
        target[...] = typed_randn(dtype, target.shape)

    # generate Blocked Tensor
    return ts


def random_bdmatrix(bm=None, dtype='complex128'):
    '''
    Generate a random Block Diagonal 2D Tensor.

    Parameters:
        :bm: <BlockMarker>

    Return:
        <Tensor>,
    '''
    cells = [typed_randn(dtype, [ni, ni]) * 2 / ni for ni in bm.nr]
    ts = Tensor(block_diag(*cells), labels=[BLabel('r', bm), BLabel('c', bm)])
    return ts


def random_zeroflux_tensor(bms_or_sites, trunc_rate=0.2, dtype='complex128', bmg=None, signs=None, zero=None, info=None):
    '''
    random tensor with zero flux.
    '''
    ndim = len(bms_or_sites)
    if bmg is None:
        spaceconfig = SuperSpaceConfig([1,2,1])
        bmg = SimpleBMG(spaceconfig=spaceconfig, qstring='QM')
    if signs is None:
        signs = [1]*ndim//2 + [-1]*(ndim-ndim//2)
    bms = bms_or_sites
    for i in range(ndim):
        if isinstance(bms[i], int):
            bms[i] = bmg.random_bm(nsite=bms[i])
    nzbs = zero_flux_blocks([bm.qns for bm in bms], signs, bmg, zero=zero)
    ts = Tensor(np.zeros([bm.N for bm in bms], dtype=dtype),
            labels=[BLabel('s_%d'%i, bm) for i,bm in enumerate(bms)])
    for b in zip(*nzbs):
        ts.set_block(b, typed_randn(dtype, [bm.blocksize(ib) for ib, bm in zip(b, bms)]))
    if info is not None:
        info['bmg'] = bmg
    return ts

