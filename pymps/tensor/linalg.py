import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd, lu, block_diag
import pdb

from ..blockmarker import BlockMarker
from .tensor import BLabel, Tensor
from .btensor import BTensor

__all__ = ['svdbd_map', 'svdbd']


def svdbd_map(A, mapping_rule=None, full_matrices=False):
    '''
    Get the svd decomposition for dense tensor with block structure.

    Parameters:
        :A: 2D<Tensor>, the input matrix, with <BLabel>s.
        :mapping_rule: function, the mapping between left block and right blocks, using labels.

    Return:
        (eigenvalues,eigenvectors) if return vecs==True.
        (eigenvalues,) if return vecs==False.
    '''
    # check datas
    if mapping_rule is None:
        def mapping_rule(x): return x
    bm1, bm2 = A.labels[0].bm, A.labels[1].bm
    extb1, extb2 = bm1.extract_block, bm2.extract_block
    qns1, qns2 = bm1.labels, list(bm2.labels)
    SL, UL, VL, SL2 = [], [], [], []

    um_l, m_r = [], []  # un-matched blocks for left and matched for right
    for i, lbi in enumerate(qns1):
        lbj = mapping_rule(lbi) if mapping_rule is not None else lbi
        try:
            j = qns2.index(lbj)
            m_r.append(j)
        except:
            um_l.append(i)
            size = bm1.blocksize(bm1.index_qn(lbi)[0])
            UL.append(identity(size))
            SL.append(sps.csr_matrix((size, size)))
            continue
        mi = extb2(extb1(A, (bm1.index_qn(lbi).item(),), axes=(0,)),
                   (bm2.index_qn(lbj).item(),), axes=(1,))
        if mi.shape[0] == 0 or mi.shape[1] == 0:
            ui, vi = np.zeros([mi.shape[0]] * 2), np.zeros([mi.shape[1]] * 2)
            SL.append(sps.csr_matrix(tuple([mi.shape[0]] * 2)))
            SL2.append(sps.csr_matrix(tuple([mi.shape[1]] * 2)))
        else:
            ui, si, vi = svd(mi, full_matrices=full_matrices,
                             lapack_driver='gesvd')
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
            VL.append(identity(size))
            SL.append(sps.csr_matrix((0, 0)))
            SL2.append(sps.csr_matrix(tuple([size] * 2)))
    order = argsort(m_r)
    # reorder S and V, and build matrices
    m_l = ones(len(SL), dtype='bool')
    m_l[um_l] = False
    um_r = ones(len(SL), dtype='bool')
    um_r[m_r] = False

    Smat = ndarray([len(SL)] * 2, dtype='O')
    Smat[m_l, m_r] = array(SL)[m_l]
    Smat[um_l, um_r] = array(SL)[um_l]
    Smat2 = ndarray([len(SL2)] * 2, dtype='O')
    Smat2[arange(len(SL2)), m_r] = SL2
    VL = array(VL)[order]
    return block_diag(*UL), array(sps.bmat(Smat)), block_diag(*VL), array(sps.bmat(Smat2))


def svdbd(A, cbond_str='X', kernel='svd'):
    '''
    Get the svd decomposition for dense tensor with block structure.

    Parameters:
        :A: 2D<Tensor>, the input matrix, with <BLabel>s.
        :cbond_str: str, the labes string for center bond.
        :kernel: 'svd'/'ldu', the kernel of svd decomposition.

    Return:
        (U,S,V) that U*S*V = A
    '''
    # check and prepair datas
    bm1, bm2 = A.labels[0].bm, A.labels[1].bm
    # add support for null block marker
    if bm1.qns.shape[1] == 0:
        if kernel == 'svd':
            U, S, V = svd(A, full_matrices=False, lapack_driver='gesvd')
        elif kernel == 'ldu':
            U, S, V = ldu(A)
        else:
            raise ValueError()
        center_label = BLabel(cbond_str, BlockMarker(
            qns=np.zeros([1, 0], dtype='int32'), Nr=array([0, len(S)])))
        U = Tensor(U, labels=[A.labels[0].bm, center_label])
        V = Tensor(V, labels=[center_label, A.labels[1].bm])
        return U, S, V
    qns1, qns2 = bm1.qns, bm2.qns
    qns1_1d = qns1.copy().view([('', qns1.dtype)] * qns1.shape[1])
    qns2_1d = qns2.copy().view([('', qns2.dtype)] * qns2.shape[1])
    common_qns_1d = np.intersect1d(qns1_1d, qns2_1d)
    common_qns_2d = common_qns_1d.view(
        bm1.qns.dtype).reshape(-1, bm1.qns.shape[-1])
    cqns1 = tuple(bm1.index_qn(lbi).item() for lbi in common_qns_2d)
    cqns2 = tuple(bm2.index_qn(lbi).item() for lbi in common_qns_2d)

    # do SVD
    UL, SL, VL = [], [], []
    for c1, c2 in zip(cqns1, cqns2):
        cell = A.get_block((c1, c2))
        if kernel == 'svd':
            Ui, Si, Vi = svd(cell, full_matrices=False, lapack_driver='gesvd')
        elif kernel == 'ldu':
            Ui, Si, Vi = ldu(cell)
        else:
            raise ValueError()
        UL.append(Ui)
        SL.append(Si)
        VL.append(Vi)

    # get center BLabel and S
    nr = [len(si) for si in SL]
    Nr = np.append([0], np.cumsum(nr))
    b0 = BLabel(cbond_str, BlockMarker(Nr=Nr, qns=common_qns_2d))
    S = np.concatenate(SL)

    # get U, V
    if isinstance(A, Tensor):
        # get correct shape of UL
        ptr = 0
        for i, lbi_1d in enumerate(qns1_1d):
            if lbi_1d != common_qns_1d[ptr]:
                UL.insert(i, np.zeros([bm1.blocksize(i), 0], dtype=A.dtype))
            elif ptr != len(common_qns_1d) - 1:
                ptr = ptr + 1

        # the same for VL
        ptr = 0
        for i, lbi_1d in enumerate(qns2_1d):
            if lbi_1d != common_qns_1d[ptr]:
                VL.insert(i, np.zeros([0, bm2.blocksize(i)], dtype=A.dtype))
            elif ptr != len(common_qns_1d) - 1:
                ptr = ptr + 1
        U, V = Tensor(block_diag(
            *UL), labels=[A.labels[0], b0]), Tensor(block_diag(*VL), labels=[b0, A.labels[1]])
    elif isinstance(A, BTensor):
        U = BTensor(dict(((b1, b2), data) for b2, (b1, data)
                         in enumerate(zip(cqns1, UL))), labels=[A.labels[0], b0])
        V = BTensor(dict(((b1, b2), data) for b1, (b2, data)
                         in enumerate(zip(cqns2, VL))), labels=[b0, A.labels[1]])

    # detect a shape error raised by the wrong ordering of block marker.
    if A.shape[0] != U.shape[0]:
        raise Exception('Error! 1. check block markers!')
    return U, S, V

