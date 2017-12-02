import numpy as np
import pdb
from numpy import array, einsum
from functools import reduce

from .tensor import tdot, Tensor, BLabel
from .btensor import BTensor

__all__ = ['check_validity_tensor', 'gen_eincode', 'contract',
           'tensor_block_diag']


def gen_eincode(*labels):
    '''
    Generate einsum string from contraction strings of tensors.

    labels:
        The labels for contraction.
    '''
    all_list = sum(labels, [])
    unique_list, legs = [], []
    for item in all_list:
        if item in unique_list:
            legs.remove(item)
        else:
            unique_list.append(item)
            legs.append(item)
    mapping = dict((label, chr(97 + i)) for i, label in enumerate(unique_list))
    tokens = []
    for lbs in labels:
        tokens.append(''.join([mapping[l] for l in lbs]))
    token_r = ''.join([mapping[l] for l in legs])
    return '%s->%s' % (','.join(tokens), token_r), legs


def contract(*tensors):
    '''
    Contract a collection of tensors

    Parameters:
        :tensors: <Tensor>s, A list of <Tensor> instances.

    Return:
        <Tensor>
    '''
    if len(tensors) == 0:
        raise ValueError('Not enough parameters.')
    if len(tensors) == 1:
        tensors = tensors[0]
    labels = [t.labels for t in tensors]
    # asign dummy tokens.
    eincode, leglabels = gen_eincode(*labels)
    return Tensor(einsum(eincode, *tensors), labels=leglabels)


def check_validity_tensor(ts):
    '''Check if it is a valid tensor.'''
    valid = True
    for i in range(np.ndim(ts)):
        if hasattr(ts.labels[i], 'bm'):
            if not ts.shape[i] == ts.labels[i].bm.N:
                valid = False
    if isinstance(ts.data, dict):
        # check data
        for blk, d in ts.data.items():
            if not tuple(lb.bm.blocksize(bi) for lb, bi in zip(ts.labels, blk)) == d.shape:
                valid = False
    return valid


def tensor_block_diag(tensors, axes, nlabels=None):
    '''
    Tensor block diagonalization.

    Parameters:
        :tensors: list, list of tensors(Tensor/BTensor).
        :axes: tuple, target axes, tensor dimension in axes not in target axes should be identical.
        :nlabels: list, strings for new labels.

    Return:
        tensor,
    '''
    ndim = tensors[0].ndim
    axes = [axis if axis >= 0 else axis + ndim for axis in axes]
    if nlabels is None:
        nlabels = [tensors[0].labels[axi] for axi in axes]
    # check uniform shape for remaining axes.
    for i, lbs in enumerate(zip([t.labels for t in tensors])):
        if i not in axes and not all([lb1.bm == lb2.bm for lb1, lb2 in zip(lbs[:-1], lbs[1:])]):
            raise ValueError
    # new labels
    newlabel = tensors[0].labels[:]
    for ax, nlb in zip(axes, nlabels):
        if hasattr(newlabel[ax], 'bm'):
            newlabel[ax] = BLabel(nlb, reduce(
                lambda x, y: x + y, [ts.labels[ax].bm for ts in tensors]))
        else:
            newlabel[ax] = nlb

    if isinstance(tensors[0], Tensor):
        # get new shapes and offsets
        shapes = array([t.shape for t in tensors])
        ntensor, ndim = shapes.shape
        offsets = np.concatenate([np.zeros([1, len(axes)], dtype='int32'), np.cumsum(
            shapes[:, axes], axis=0)], axis=0)
        newshape = shapes[0]
        newshape[axes] = offsets[-1]

        ts = Tensor(
            np.zeros(newshape, dtype=tensors[0].dtype), labels=newlabel)
        sls0 = [slice(None)] * ndim
        for it, t in enumerate(tensors):
            sls = sls0[:]
            for iax, ax in enumerate(axes):
                sls[ax] = slice(offsets[it, iax], offsets[it + 1, iax])
            ts[sls] = t
        return ts
    elif isinstance(tensors[0], BTensor):
        ndata = {}
        offset = np.zeros(ndim, dtype='int32')
        for it, t in enumerate(tensors):
            for i, (blk, data) in enumerate(t.data.items()):
                nblk = tuple(blk + offset)
                ndata[nblk] = data
            offset[axes] += [t.labels[ax].bm.nblock for ax in axes]
        return BTensor(ndata, newlabel)

