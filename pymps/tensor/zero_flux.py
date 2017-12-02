import numpy as np
import pdb

from ..blockmarker import trunc_bm
from .btensor import BTensor
from .tensor import _same_diff_labels, Tensor

__all__ = ['is_zero_flux', 'clip_nonzero_flux', 'zero_flux_blocks', 'nonzero_flux_blocks', 'btdot']


def is_zero_flux(ts, signs, bmg):
    '''
    a tensor is zero flux or not.

    Parameters:
        :ts: <Tensor>,
        :signs: list, flow directions.
        :bmg: <BlockMarkerGenerator>,

    Return:
        bool, true if the flow is quantum number conserving.
    '''
    ts = ts.merge_axes(slice(0, ts.ndim), bmg=bmg, signs=signs)
    if isinstance(ts, BTensor):
        bm = ts.labels[0].bm
        return np.all(concatenate([bm.qns[k] for k in list(ts.data.keys())]) == 0)
    else:
        kpmask = (abs(ts) > 1e-10)
        cbm = trunc_bm(ts.labels[0].bm, kpmask)
        return np.all(cbm.qns == 0)


def clip_nonzero_flux(ts, signs, bmg, zero=None):
    '''
    clip non-zero flux terms.

    Parameters:
        :ts: <Tensor>,
        :signs: list, flow directions.
        :bmg: <BlockMarkerGenerator>,

    Return:
        bool, true if the flow is quantum number conserving.
    '''
    nzbs = nonzero_flux_blocks([l.bm.qns for l in ts.labels], signs, bmg, zero)
    for blk in zip(*nzbs):
        if isinstance(ts, BTensor) and blk in ts.data:
            del(ts.data[blk])
        else:
            ts.set_block(blk, 0)

def zero_flux_blocks(qns_list, signs, bmg, zero=None):
    '''zero flux blocks, assume qns are compact'''
    return np.where(_zero_flux_mask(qns_list, signs, bmg, zero))


def nonzero_flux_blocks(qns_list, signs, bmg, zero=None):
    '''nonzero flux blocks, assume qns are compact'''
    return np.where(~_zero_flux_mask(qns_list, signs, bmg, zero))


def _zero_flux_mask(qns_list, signs, bmg, zero):
    '''non-zero blocks with respect to flow quations, 
    qns are compact'''
    if zero is None:
        zero = np.zeros(len(bmg.per), dtype='int32')
    for k, (per_k, target_k) in enumerate(zip(bmg.per, zero)):
        mesh_list = np.meshgrid(*[qns[:,k] for qns in qns_list], indexing='ij')
        acc = np.sum([m*s for m, s in zip(mesh_list, signs)], axis=0)
        if per_k != bmg.INF:
            acc = acc%per_k
        mask_ = acc==target_k
        mask = mask_ if k==0 else mask&mask_
    return mask
 

def btdot(tensor1, tensor2, signs1, signs2, bmg):
    '''
    Tensor dot between two tensors, faster than contract in most case?

    Args:
        tensor1,tensor2 (:obj:`Tensor`): two tensors to contract.

    Returns:
        :obj:`Tensor`: output tensor.
    '''
    inner1, inner2, outer1, outer2 = _same_diff_labels(tensor1.labels, tensor2.labels)
    # output array
    out_shape = [tensor1.shape[i] for i in outer1]+[tensor2.shape[i] for i in outer2]
    out_arr = np.zeros(out_shape, dtype=np.find_common_type((tensor1.dtype, tensor2.dtype), ()))
    out_arr = Tensor(out_arr, labels=[tensor1.labels[i] for i in outer1]+[tensor2.labels[i] for i in outer2])

    # get non-zero blocks for tensor1 and tensor2
    nz_table1 = _gen_index_table(zero_flux_blocks([l.bm.qns for l in tensor1.labels], signs1, bmg), key_axes=inner1)
    nz_table2 = _gen_index_table(zero_flux_blocks([l.bm.qns for l in tensor2.labels], signs2, bmg), key_axes=inner2)

    for k, l1 in nz_table1.items():
        if k in nz_table2:
            l2 = nz_table2[k]
            for b1, o1 in l1:
                for b2, o2 in l2:
                    out_b = o1+o2
                    bdata = np.tensordot(tensor1.get_block(b1), tensor2.get_block(b2), axes=(inner1, inner2))
                    out_arr.set_block(out_b, bdata)
    return out_arr


def _gen_index_table(nzblocks, key_axes):
    val_axes = [i for i in range(len(nzblocks)) if i not in key_axes]
    key_blocks = [nzblocks[i] for i in key_axes]
    val_blocks = [nzblocks[i] for i in val_axes]
    d = {}
    for blk, key, val in zip(zip(*nzblocks), zip(*key_blocks), zip(*val_blocks)):
        if key in d:
            d[key].append((blk, val))
        else:
            d[key] = [(blk, val)]
    return d
