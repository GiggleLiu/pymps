'''
Block Matrix related operations.
'''
import numpy as np
from scipy.linalg import eigh, eigvalsh, svd
from scipy.sparse.linalg import eigsh
import scipy.sparse as sps

from .blockmarker import BlockMarker

__all__ = ['block_diag', 'eigbh', 'svdb', 'sign4bm', 'join_bms']


def block_diag(*arrs):
    """
    [ReWrite of scipy.linalg.block_diag]
    Create a block diagonal matrix from provided arrays.
    Given the inputs `A`, `B` and `C`, the output will have these
    arrays arranged on the diagonal::
        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]
    Args
    ----------
    A, B, C, ... : array_like, up to 2-D
        Input arrays.  A 1-D array or array_like sequence of length `n` is
        treated as a 2-D array with shape ``(1,n)``.
    Returns
    -------
    D : ndarray
        Array with `A`, `B`, `C`, ... on the diagonal.  `D` has the
        same dtype as `A`.
    Notes
    -----
    If all the input arrays are square, the output is known as a
    block diagonal matrix.
    Empty sequences (i.e., array-likes of zero size) are ignored.
    Examples
    --------
    >>> from scipy.linalg import block_diag
    >>> A = [[1, 0],
    ...      [0, 1]]
    >>> B = [[3, 4, 5],
    ...      [6, 7, 8]]
    >>> C = [[7]]
    >>> block_diag(A, B, C)
    array([[1, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 3, 4, 5, 0],
           [0, 0, 6, 7, 8, 0],
           [0, 0, 0, 0, 0, 7]])
    >>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  2.,  3.,  0.,  0.],
           [ 0.,  0.,  0.,  4.,  5.],
           [ 0.,  0.,  0.,  6.,  7.]])
    """
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                         "greater than 2: %s" % bad_args)

    shapes = np.array([a.shape for a in arrs])
    out_dtype = np.find_common_type([arr.dtype for arr in arrs], [])
    out = np.zeros(np.sum(shapes, axis=0), dtype=out_dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out


def eigbh(cm, bm, return_vecs=True):
    '''
    Get the eigenvalues and eigenvectors for matrice with block structure specified by block marker.

    Args:
        cm (csr_matrix/bsr_matrix): the input matrix.
        return_vecs (bool): return the eigenvectors or not.
        bm (<BlockMarkerBase>): the block marker.

    Returns:
        (eigenvalues,eigenvectors) if return vecs==True.
        (eigenvalues) if return vecs==False.
    '''
    EL, UL = [], []
    is_sparse = sps.issparse(cm)
    for i in range(bm.nblock):
        mi = bm.extract_block(cm, (i, i))
        if is_sparse:
            mi = mi.toarray()
        if return_vecs:
            ei, ui = eigh(mi)
            EL.append(ei)
            UL.append(ui)
        else:
            ei = eigvalsh(mi)
            EL.append(ei)
    if return_vecs:
        return np.concatenate(EL), block_diag(*UL)
    else:
        return np.concatenate(EL)


def svdb(cm, bm, bm2=None, mapping_rule=None):
    '''
    Get the svd decomposition for matrix with block structure specified by block markers.

    Args:
        cm (csr_matrix/bsr_matrix): the input matrix.
        :bm/bm2: <BlockMarkerBase>, the block marker, bm2 specifies the column block marker.
        mapping_rule (function, the mapping between left block and right blocks): using qns.

    Returns:
        (eigenvalues,eigenvectors) if return vecs==True.
        (eigenvalues,) if return vecs==False.
    '''
    SL, UL, VL, SL2 = [], [], [], []
    is_sparse = sps.issparse(cm)
    if bm2 is None:
        bm2 = bm
    extb1, extb2 = bm.extract_block, bm2.extract_block
    qns1, qns2 = bm.qns, list(bm2.qns)

    um_l, m_r = [], []  # un-matched blocks for left and matched for right
    for i, lbi in enumerate(qns1):
        lbj = mapping_rule(lbi) if mapping_rule is not None else lbi
        try:
            j = qns2.index(lbj)
            m_r.append(j)
        except:
            um_l.append(i)
            size = bm.blocksize(bm.index_qn(lbi)[0])
            UL.append(identity(size))
            SL.append(sps.csr_matrix((size, size)))
            continue
        mi = extb2(extb1(cm, (bm.index_qn(lbi)[0],), axes=(
            0,)), (bm2.index_qn(lbj)[0],), axes=(1,))
        if is_sparse:
            mi = mi.toarray()
        if mi.shape[0] == 0 or mi.shape[1] == 0:
            ui, vi = np.zeros([mi.shape[0]] * 2), np.zeros([mi.shape[1]] * 2)
            SL.append(sps.csr_matrix(tuple([mi.shape[0]] * 2)))
            SL2.append(sps.csr_matrix(tuple([mi.shape[1]] * 2)))
        else:
            ui, si, vi = svd(mi, full_matrices=False)
            if mi.shape[1] > mi.shape[0]:
                si1, si2 = si, append(si, np.zeros(mi.shape[1] - mi.shape[0]))
            elif mi.shape[1] < mi.shape[0]:
                si1, si2 = append(si, np.zeros(mi.shape[0] - mi.shape[1])), si
            else:
                si1 = si2 = si
            SL.append(sps.diags(si1, 0))
            SL2.append(sps.diags(si2, 0))
        UL.append(ui)
        VL.append(vi)
    for j in range(bm2.nblock):
        if not j in m_r:
            m_r.append(j)
            size = bm2.nr[j]
            VL.append(np.identity(size))
            SL.append(sps.csr_matrix((0, 0)))
            SL2.append(sps.csr_matrix(tuple([size] * 2)))
    order = np.argsort(m_r)
    # reorder S and V, and build matrices
    m_l = np.ones(len(SL), dtype='bool')
    m_l[um_l] = False
    um_r = np.ones(len(SL), dtype='bool')
    um_r[m_r] = False

    Smat = np.ndarray([len(SL)] * 2, dtype='O')
    Smat[m_l, m_r] = np.array(SL)[m_l]
    Smat[um_l, um_r] = np.array(SL)[um_l]
    Smat2 = np.ndarray([len(SL2)] * 2, dtype='O')
    Smat2[np.arange(len(SL2)), m_r] = SL2
    VL = np.array(VL)[order]
    return sps.block_diag(UL), sps.bmat(Smat), sps.block_diag(VL), sps.bmat(Smat2)

######################### Sign Problem ###############################


def sign4qns(qns, bmg):
    '''
    Get the sign for qns.

    Args:
        qns (<BasicBlockMarker>): the qns.
        bmg (<BlockMarkerGenerator>): the block marker generator.

    Returns:
        1D array, the signs.
    '''
    if np.shape(qns)[1] == 0:
        return np.ones(len(qns))
    qns = np.asarray(qns)
    qstring = bmg.qstring
    for S in ['P', 'Q', 'M']:
        if S in qstring:
            i = qstring.index(S)
            res = 1 - 2 * (qns[..., i] % 2)
            break
    return res


def sign4bm(bm, bmg, full_length=True):
    '''
    Get the sign for blockmarker.

    Args:
        bm (<BasicBlockMarker>): the block marker.
        bmg (<BlockMarkerGenerator>): the block marker generator.
        full_length (bool): expand the signs to matrix length.

    Returns:
        1D array, the signs.
    '''
    res = sign4qns(bm.qns, bmg)
    # expand to full length
    if full_length:
        res = np.repeat(res, bm.nr, axis=0)
    return res

######################### Join Two BlockMarkers  ###############################


def join_bms(bms, **kwargs):
    '''
    Join two <BasicBlockMarker>(merge two axes).

    Args:
        bms (<BasicBlockMarker>): the target qned block marker.

    Returns:
        <BlockMarker>, the new block marker.
    '''
    # inflate and get size-1 qns(robuster!)
    qns = bms[0].inflate().qns
    for i, bm in enumerate(bms[1:]):
        if i < len(bms) - 2:
            bm = bm.inflate()
        lbi = bm.qns
        lbi = np.tile(lbi, [len(qns), 1])
        qns = np.concatenate([np.repeat(qns, bm.nblock, axis=0), lbi], axis=1)

    Nr = np.append([0], np.cumsum(
        [bms[-1].nr] * np.prod([bm.N for bm in bms[:-1]])))
    return BlockMarker(Nr=Nr, qns=qns)
