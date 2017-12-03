cimport numpy as np
import numpy as np

from ..tensor import _same_diff_labels

ctypedef fused DTYPE_t:
    np.float64_t
    np.complex128_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def btdot_{{dtype_token}}(np.ndarray[DTYPE_t] t1,
        np.ndarray[DTYPE_t] t2,
        np.ndarray[DTYPE_t] Nr1,
        Nr2, signs1, signs2, per, outarr):
    inner1, inner2, outer1, outer2 = _same_diff_labels(t1.labels, t2.labels)
    # output array
    out_shape = [t1.shape[i] for i in outer1]+[t2.shape[i] for i in outer2]
    out_arr = np.zeros(out_shape, dtype=np.find_common_type((t1.dtype, t2.dtype), ()))
    out_arr = Tensor(out_arr, labels=[t1.labels[i] for i in outer1]+[t2.labels[i] for i in outer2])

    # get non-zero blocks for t1 and t2
    nz_table1 = _gen_index_table(zero_flux_blocks([l.bm.qns for l in t1.labels], signs1, bmg), key_axes=inner1)
    nz_table2 = _gen_index_table(zero_flux_blocks([l.bm.qns for l in t2.labels], signs2, bmg), key_axes=inner2)

    for k, l1 in nz_table1.items():
        if k in nz_table2:
            l2 = nz_table2[k]
            for b1, o1 in l1:
                for b2, o2 in l2:
                    out_b = o1+o2
                    bdata = np.tensordot(t1.get_block(b1), t2.get_block(b2), axes=(inner1, inner2))
                    out_arr.set_block(out_b, bdata)


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
