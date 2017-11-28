'''
Contraction Handler classed for MPS and MPO. 
'''
from numpy import *
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix
from functools import reduce
import pdb
import time

from ..tensor import Tensor
from ..toolbox.linear import fast_svd, kron_csr
from ..construct.opstring import UNSETTLED, OpUnit, OpString, OpCollection
from .mpslib import mps_sum
from .mps import mPS

__all__ = ['expect_onsite', 'get_expect', 'op_mul_mps',
           'USVobj', 'XUSVobj', 'get_expect_ivmps', 'G_Gong']


def expect_onsite(opunit, ket, bra=None):
    '''
    Get the expectation value of the on-site term.

    Args:
        opunit (<OpUnit>): the on-site operator.
        ket (<MPS>): the ket.
        bra (<MPS>/None, the bra): "same" as ket if is `None`.

    Returns:
        number, the expectation value of opunit on this ket.
    '''
    l = opunit.siteindex
    if l == UNSETTLED:
        raise Exception('Please set the site index of opunit!')
    if ket.l != l:
        ket >> l - ket.l
    site_axis = ket.site_axis
    M = ket.get(l, attach_S='B')
    if bra is None:
        MH = M.make_copy(labels=[
                         li + '\'' if i == site_axis else li for i, li in enumerate(M.labels)]).conj()
    else:
        MH = bra.get(l, attach_S='B')
        MH.labels = [li + '\'' if i ==
                     site_axis else li for i, li in enumerate(M.labels)]
    O = Tensor(opunit.get_data(), labels=[
               MH.labels[site_axis], M.labels[site_axis]])
    res = MH * O * M
    return res


def expect_opstring(ket, opstring):
    '''
    Get the expectation value of the opstring.

    Args:
        ket (<MPS>): the ket.
        opunit (<OpUnit>): the on-site operator.

    Returns:
        number, the expectation value of opunit on this ket.
    '''
    sites = opstring.siteindices
    assert(len(sites) > 1)
    lmin = sites.min()
    lmax = sites.max()
    if lmin == UNSETTLED:
        raise Exception('Please set the site index of opunit!')
    if ket.l != lmin:
        ket >> lmin - ket.l
    site_axis = ket.site_axis
    rlink_axis = ket.rlink_axis
    llink_axis = ket.llink_axis
    for i in range(lmin, lmax + 1):
        M = ket.get(lmin, attach_S='B')
        if i == lmin:
            MH = M.make_copy(labels=[li + '\'' if i in [site_axis, rlink_axis]
                                     else li for i, li in enumerate(M.labels)]).conj()
            O = Tensor(opunit.get_data(), labels=[
                       MH.labels[site_axis], M.labels[site_axis]])
            res = MH * O * M
        elif i == lmax:
            MH = M.make_copy(labels=[li + '\'' if i in [site_axis, llink_axis]
                                     else li for i, li in enumerate(M.labels)]).conj()
            O = Tensor(opunit.get_data(), labels=[
                       MH.labels[site_axis], M.labels[site_axis]])
            res = res * MH * O * M
        elif i in sites:
            MH = M.make_copy(
                labels=[li + '\'' for i, li in enumerate(M.labels)]).conj()
            O = Tensor(opunit.get_data(), labels=[
                       MH.labels[site_axis], M.labels[site_axis]])
            res = res * MH * O * M
        else:
            MH = M.make_copy(labels=[li + '\'' if i in [llink_axis, rlink_axis]
                                     else li for i, li in enumerate(M.labels)]).conj()
            res = res * MH * M
    return res


def get_expect(op, ket, bra=None, sls=slice(None), memorial=False):
    '''
    Get the expectation value of an operator.

    Args:
        op (<OpUnit>/<OpString>/<OpCollection>): the operator.
        ket (<MPS>): the ket.
        bra (<MPS>/None, the bra): "same" as ket if is `None`.
        sls (slice): the interval to perform contraction.
        memorial (bool): return contraction history if True.

    Returns:
        number, the expectation value of operator on this ket.
    '''
    if isinstance(op, OpCollection):
        return sum([get_expect(opi, ket) for opi in op.ops], axis=0)
    elif op is None:
        sites = []
    sites = op.siteindices
    site_axis = ket.site_axis
    rlink_axis = ket.rlink_axis
    llink_axis = ket.llink_axis
    if bra is None:
        bra = ket.tobra()

    res = None
    memory = []
    attach_S = 'B' if ket.l == (
        sls.start if sls.start is not None else 0) else 'A'
    for i in range(ket.nsite)[sls]:
        items = []
        M = ket.get(i, attach_S=attach_S)
        MH = bra.get(i, attach_S=attach_S)
        MH.labels[llink_axis] = M.labels[llink_axis] + '\''
        MH.labels[rlink_axis] = M.labels[rlink_axis] + '\''
        if i in sites:
            opunit = op if isinstance(op, OpUnit) else op.query(i)[0]
            MH.labels[site_axis] = M.labels[site_axis] + '\''
            O = Tensor(opunit.get_data(), labels=[
                       MH.labels[site_axis], M.labels[site_axis]])
            items.append(O)
        else:
            MH.labels[site_axis] = M.labels[site_axis]
        items.append(M)
        items.insert(0, MH)
        for item in items:
            res = (res * item) if res is not None else item
        memory.append(res)
    if memorial:
        return memory
    return res


def get_expect_ivmps(op, ket):
    '''
    Get the expectation value of op per site.

    Args:
        op (<OpUnit>/<OpString>/<OpCollection>):
        :ket: <IVMPS>
    '''
    npart = ket.npart
    if isinstance(op, OpCollection):
        return sum([get_expect_ivmps(opi, ket) for opi in op.ops], axis=0)
    elif op is None:
        sites = []
        lmin, lmax = 0, npart
    sites = op.siteindices
    lmin, lmax = sites[0], sites[-1] + 1
    site_axis = 0

    res = None
    for i in range(lmin, lmax):
        items = []
        M = ket.GL[i].mul_axis(ket.LL[i], axis=rlink_axis)
        if i == lmin:
            # some spetial treatment for first site.
            M = M.mul_axis(ket.LL[(i - 1) % npart], axis=llink_axis)
            MH = M.conj()
            MH.labels[llink_axis] = M.labels[llink_axis]
        else:
            MH = M.conj()
            MH.labels[llink_axis] = M.labels[llink_axis] + '\''
        MH.labels[site_axis] = M.labels[site_axis] + '\''
        if i == lmax - 1:
            MH.labels[rlink_axis] = M.labels[rlink_axis]
        else:
            MH.labels[rlink_axis] = M.labels[rlink_axis] + '\''
        if i in sites:
            opunit = op if isinstance(op, OpUnit) else op.query(i)[0]
            MH.labels[site_axis] = M.labels[site_axis] + '\''
            O = Tensor(opunit.get_data(), labels=[
                       MH.labels[site_axis], M.labels[site_axis]])
            items.append(O)
        else:
            MH.labels[site_axis] = M.labels[site_axis]
        items.append(M)
        items.insert(0, MH)
        for item in items:
            res = (res * item) if res is not None else item
    return res


def get_expect_ivmps_chain(op, ket):
    '''
    Get the expectation value of op per site.

    Args:
        op (<OpUnit>/<OpString>/<OpCollection>):
        ket (<IVMPS>): chain version.
    '''
    if isinstance(op, OpCollection):
        return sum([get_expect_ivmps_chain(opi, ket) for opi in op.ops], axis=0)
    sites = op.siteindices
    site_axis = 0

    # keep environments orthogonal
    # turn the ivmps into a AB chain mps: A <- link1 -> B <- link2 -> A <- link1 -> B ...
    res = None
    for sitei in range(min(sites), max(sites) + 1):
        ti = sitei % ket.npart
        res = ket.tensors[ti]
        if i == lmin:
            # some spetial treatment for first site.
            M = M.mul_axis(ket.LL[(i - 1) % npart], axis=llink_axis)
            MH = M.conj()
            MH.labels[llink_axis] = M.labels[llink_axis]
        else:
            MH = M.conj()
            MH.labels[llink_axis] = M.labels[llink_axis] + '\''
        MH.labels[site_axis] = M.labels[site_axis] + '\''
        if i == lmax - 1:
            MH.labels[rlink_axis] = M.labels[rlink_axis]
        else:
            MH.labels[rlink_axis] = M.labels[rlink_axis] + '\''
        if i in sites:
            opunit = op if isinstance(op, OpUnit) else op.query(i)[0]
            MH.labels[site_axis] = M.labels[site_axis] + '\''
            O = Tensor(opunit.get_data(), labels=[
                       MH.labels[site_axis], M.labels[site_axis]])
            items.append(O)
        else:
            MH.labels[site_axis] = M.labels[site_axis]
        items.append(M)
        items.insert(0, MH)
        for item in items:
            res = (res * item) if res is not None else item
    return res


def op_mul_mps(op, ket):
    '''
    Get the expectation value of the on-site term.

    Args:
        op (<OpUnit>/<OpString>): the on-site operator.
        ket (<MPS>): the ket.

    Returns:
        new <MPS>, which is not canonicalized!
    '''
    sites = op.siteindices
    nsite = ket.nsite
    site_axis = ket.site_axis

    ML = []
    for i in range(nsite):
        # get M matrices.
        M = ket.ML[i]
        # product
        if i in sites:
            opunit = op if isinstance(op, OpUnit) else op.query(i)[0]
            O = Tensor(opunit.get_data(), labels=[
                       'x_%s' % i, M.labels[site_axis]])
            M = O * M
            M = M.chorder([1, 0, 2])
        ML.append(M)
    res = mPS(ML, ket.l, S=ket.S, labels=['s', 'a'])
    return res


class USVobj(LinearOperator):
    '''
    Linear Operator in the USV form, with S low dimensional.

        ----U----
            |
            S
            |
        ----V----

    Construct:
        USVobj(U,S,V)

    Attributes:
        :U/V: <Tensor>, with 3 indices (llink, site, rlink).
        S (tensor/ndarray/None): None is for the identity matrix.
    '''

    def __init__(self, U, S, V):
        # check for data
        assert(ndim(U) == 3 and ndim(V) == 3)
        assert(isinstance(S, Tensor) or (isinstance(
            S, ndarray) and ndim(S) == 1) or (S is None))
        self.U, self.S, self.V = U, S, V
        # super(USVobj,self).__init__((U.shape[0]*U.shape[2],V.shape[0]*V.shape[2]),matvec=self._matvec,matmat=self._matmat,rmatvec=self._rmatvec,dtype=U.dtype)
        super(USVobj, self).__init__(
            shape=(U.shape[0] * U.shape[2], V.shape[0] * V.shape[2]), dtype=U.dtype)

    def __str__(self):
        return 'U(%s|%s)S(%s|%s)V(%s|%s)' % (','.join(self.U.labels),
                                             ','.join(str(x)
                                                      for x in self.U.shape),
                                             ','.join(self.S.labels) if isinstance(
                                                 self.S, Tensor) else '-',
                                             ','.join(str(x)
                                                      for x in self.S.shape) if self.S is not None else 1,
                                             ','.join(self.V.labels),
                                             ','.join(str(x) for x in self.V.shape))

    def _matvec(self, x):
        '''matrix vector multiplication, x should be an 1D array.'''
        V = self.V
        x = Tensor(x.reshape(V.shape[0], V.shape[2]),
                   labels=[V.labels[0], V.labels[2]])
        return self.tdot(x)

    def tdot(self, x):
        '''
        Tensor dot.

        Args:
            :x: <Tensor>.  
        '''
        res = self.V * x
        if ndim(res) != ndim(self.V) + ndim(x) - 4:
            raise ValueError('Can not contract V=%s and x=%s properly!' % (
                self.V.__repr__(), x.__repr__()))
        if self.S is not None:
            res = self.S * res
        res = self.U * res
        return res

    def _rmatvec(self, x):
        U = self.U
        x = Tensor(x.reshape(U.shape[0], U.shape[2]),
                   labels=[U.labels[0], U.labels[2]])
        return self.rtdot(x)

    def rtdot(self, x):
        '''
        Right tensor dot

        Args:
            :x: <Tensor>.  
        '''
        res = self.U * x
        if ndim(res) != ndim(self.U) + ndim(x) - 4:
            raise ValueError('Can not contract U=%s and x=%s properly!' % (
                self.V.__repr__(), x.__repr__()))
        if self.S is not None:
            res = res * self.S
        return res * self.V

    def _matmat(self, X):
        V = self.V
        X = Tensor(X.reshape(V.shape[0], V.shape[2], X.shape[1]), labels=[
                   V.labels[0], V.labels[2], random.random()])
        return self.tdot(X)

    def _rmatmat(self, X):
        U0 = self.usvs[0].U
        U1 = self.usvs[-1].U
        X = Tensor(X.reshape(U0.shape[0], U1.shape[2]), labels=[
                   U0.labels[0], U1.labels[2]])
        return self.rtdot(X)

    def _adjoint(self):
        return USVobj(self.V.conj(), self.S.conj(), self.U.conj())

    def rdot(self, x):
        '''Right dot product.'''
        if ndim(x) == 1:
            return self._rmatvec(x)
        else:
            return self._rmatmat(x)

    def join(self, target):
        '''Attach one more pieces at right hand side.'''
        if isinstance(target, USVobj):
            return XUSVobj(self, target)
        elif isinstance(target, XUSVobj):
            return XUSVobj(*([self] + target.usvs))
        else:
            raise TypeError('Can not join %s and %s' %
                            (self.__class__, target.__class__))

    def toarray(self):
        '''Transform to an array.'''
        return asarray(self.totensor()).reshape(self.shape)

    def totensor(self):
        '''Transform to a <Tensor>.'''
        if isinstance(self.S, Tensor):
            res = self.U * self.S * self.V
        elif isinstance(self.S, ndarray):
            res = self.U * self.S[:, newaxis] * self.V
        else:
            res = self.U * self.V
        return res


class XUSVobj(LinearOperator):
    '''
    Linear Operator in the X-USV form, with S low dimensional.

        ----U----U2----
            |     |
            S    S2      ...
            |     |
        ----V----V2----

    Construct:
        BiUSVobj(usv1,usv2)

    Attributes:
        :args: <USVobj>.
    '''

    def __init__(self, *args):
        # check for data
        self.usvs = args
        for usv1, usv2 in zip(self.usvs[:-1], self.usvs[1:]):
            assert(usv1.U.shape[2] == usv2.U.shape[0]
                   and usv1.V.shape[2] == usv2.V.shape[0])
            assert(usv1.U.labels[2] == usv2.U.labels[0]
                   and usv1.V.labels[2] == usv2.V.labels[0])
        UL, SL, VL = [], [], []
        for usv in self.usvs:
            UL.append(usv.U)
            VL.append(usv.V)
            SL.append(usv.S)
        self.UL, self.VL, self.SL = UL, VL, SL
        # super(XUSVobj,self).__init__((self.usvs[0].U.shape[0]*self.usvs[-1].U.shape[2],\
        #        self.usvs[0].V.shape[0]*self.usvs[-1].V.shape[2]),matvec=self._matvec,rmatvec=self._rmatvec,matmat=self._matmat,dtype=self.usvs[0].U.dtype)
        super(XUSVobj, self).__init__((self.usvs[0].U.shape[0] * self.usvs[-1].U.shape[2],
                                       self.usvs[0].V.shape[0] * self.usvs[-1].V.shape[2]), dtype=self.usvs[0].U.dtype)

    def __str__(self):
        return ' // '.join([usv.__str__() for usv in self.usvs])

    @property
    def nusv(self):
        '''Number of <USVobj>s'''
        return len(self.usvs)

    def _matvec(self, x):
        '''matrix vector multiplication, x should be an array.'''
        x = Tensor(x.reshape([self.usvs[0].V.shape[0], self.usvs[-1].V.shape[2]]),
                   labels=[self.usvs[0].V.labels[0], self.usvs[-1].V.labels[2]])
        return self.tdot(x)

    def tdot(self, x):
        '''
        Tensor dot.

        Args:
            :x: <Tensor>.  
        '''
        res = x
        resdim, nusv = ndim(x) - 2, self.nusv
        # reorder VL to get best performance, assuming p<<m.
        if prod(self.VL[0].shape[1:]) > prod(self.VL[-1].shape[:2]):
            reverse = True
            VL, SL = self.VL[::-1], self.SL[::-1]
        else:
            reverse = False
            VL, SL = self.VL, self.SL
        VL = [S * V if isinstance(S, Tensor) else (S[:, newaxis] *
                                                   V if isinstance(S, ndarray) else V) for S, V in zip(SL, VL)]

        for V in VL:
            res = V * res
        if ndim(res) != self.nusv + resdim:
            raise ValueError('Can not contract V(%s) and x=%s properly!' % (
                self.shape[1], x.__repr__()))

        # reorder UL to get best performance, start from diffcult side.
        if prod(self.UL[0].shape[1:]) < prod(self.UL[-1].shape[:2]):
            reverse = True
            UL = self.UL[::-1]
        else:
            reverse = False
            UL = self.UL
        for U in UL:
            res = U * res
        if not reverse:
            res = res.chorder([1, 0] + list(range(2, ndim(res))))
        return res

    def _rmatvec(self, x):
        x = Tensor(x.reshape([self.usvs[0].U.shape[0], self.usvs[-1].U.shape[2]]),
                   labels=[self.usvs[0].U.labels[0], self.usvs[-1].U.labels[2]])
        return self.rtdot(x)

    def rtdot(self, x):
        '''
        Tensor dot.

        Args:
            :x: <Tensor>.  
        '''
        nusv = self.nusv
        res = x
        # reorder UL to get best performance, assuming p<<m.
        if prod(self.UL[0].shape[1:]) > prod(self.UL[-1].shape[:2]):
            reverse = True
            UL, SL = self.UL[::-1], self.SL[::-1]
        else:
            reverse = False
            UL, SL = self.UL, self.SL
        UL = [U * S if isinstance(S, Tensor) else (U * S[:, newaxis]
                                                   if isinstance(S, ndarray) else U) for S, U in zip(SL, UL)]

        for U in UL:
            res = res * U
        if ndim(res) != nusv + ndim(x) - 2:
            raise ValueError('Can not contract U(%s) and x=%s properly!' % (
                self.shape[0], x.__repr__()))
        # for i,S in enumerate(self.SL):
        #    if isinstance(S,Tensor):
        #        res=S*res
        #    elif isinstance(S,ndarray):
        #        res=S[tuple([slice(None)]+[newaxis]*(nusv-i-1))]*res
        if prod(self.VL[0].shape[1:]) < prod(self.VL[-1].shape[:2]):
            reverse = True
            VL = self.VL[::-1]
        else:
            reverse = False
            VL = self.VL
        for V in VL:
            res = res * V
        if reverse:
            resdim = ndim(res)
            res = res.chorder(list(range(resdim - 2)) +
                              [resdim - 1, resdim - 2])
        return res

    def _matmat(self, X):
        V0 = self.usvs[0].V
        V1 = self.usvs[-1].V
        X = Tensor(X.reshape(V0.shape[0], V1.shape[2], X.shape[1]), labels=[
                   V0.labels[0], V1.labels[2], str(random.random())])
        res = asarray(self.tdot(X)).reshape([-1, X.shape[-1]])
        return res

    def _rmatmat(self, X):
        U0 = self.usvs[0].U
        U1 = self.usvs[-1].U
        X = Tensor(X.reshape(X.shape[0], U0.shape[0], U1.shape[2]), labels=[
                   str(random.random()), U0.labels[0], U1.labels[2]])
        return asarray(self.rtdot(X)).reshape([X.shape[0], -1])

    def rdot(self, x):
        '''Right dot product.'''
        if ndim(x) == 1:
            res = self._rmatvec(x)
        else:
            res = self._rmatmat(x)
        return res

    def _adjoint(self):
        return XUSVobj(*[usv.adjoint() for usv in self.usvs])

    def compress(self, d, dlabel='d'):
        '''Compress <XSUVobj> to <USVobj>.'''
        V0, V1 = self.VL[0], self.VL[-1]
        U0, U1 = self.UL[0], self.UL[-1]
        U, S, V = fast_svd(self, d)
        Ut = Tensor(U.reshape([U0.shape[0], U1.shape[-1], d]),
                    labels=[U0.labels[0], U1.labels[-1], dlabel])
        Vt = Tensor(V.reshape([d, V0.shape[0], V1.shape[-1]]),
                    labels=[dlabel, V0.labels[0], V1.labels[-1]])
        usv = USVobj(Ut.chorder([0, 2, 1]), S, Vt.chorder([1, 0, 2]))
        return usv

    def join(self, target):
        '''Attach one more pieces at right hand side.'''
        if isinstance(target, USVobj):
            return XUSVobj(*(self.usvs + [target]))
        elif isinstance(target, XUSVobj):
            return XUSVobj(*(self.usvs + target.usvs))
        else:
            raise TypeError('Can not join %s and %s' %
                            (self.__class__, target.__class__))

    def toarray(self):
        '''Transform to an array.'''
        return asarray(self.totensor()).reshape(self.shape)

    def totensor(self):
        '''
        Transform to a <Tensor>, Contract in USV-USV-USV... fashion.
        '''
        res = 1
        for usv in self.usvs:
            U, S, V = usv.U, usv.S, usv.V
            if isinstance(S, Tensor):
                res = res * U * S * V
            elif isinstance(S, ndarray):
                res = res * U * (S[:, newaxis] * V)
            else:
                res = res * U * V
        # chorder
        res = res.chorder([0, 2, 1, 3])
        return res


def G_Gong(ket, bra, sls, attach_S='', dense=False, return_histo=False):
    '''
    Calculate transfer matrix `Gong`.

    Args:
        :ket/bra: <MPS>s,
        sls (slice): the target segment.
        attach_S (str, 'A'):'B' or ''.
        dense (bool): get dense version if True.
        return_histo (bool): return contraction history if True.

    Returns:
        4darray/csr_matrix, in the dense version, the axes are arranged in bra to ket order.
        int the sparse version, the axes are arranged in left to right order.
    '''
    # check data
    if bra.labels[0] != ket.labels[0] or bra.labels[1] == ket.labels[1]:
        raise ValueError(
            'Make site labels identical and bond labels different please!')
    if bra.is_ket or not ket.is_ket:
        raise ValueError()

    MHL, ML = bra.get_all(attach_S=attach_S)[
        sls], ket.get_all(attach_S=attach_S)[sls]
    res = None
    memory = []
    for MH, M in zip(MHL, ML):
        if dense:
            res = (MH * M) if res is None else (MH * res * M)
        else:
            Gi = reduce(add, [kron_csr(csr_matrix(MH[:, i, :]), csr_matrix(
                M[:, i, :])) for i in range(ket.hndim)])
            res = res.dot(Gi) if res is not None else Gi
        memory.append(res)
    if len(ML) == 0:
        raise ValueError()
    return res if not return_histo else (res, memory)


def G_Wang(mpo, ket, bra, sls, attach_S='', dense=False, return_histo=False):
    '''
    Get the expectation of MPO.

    Args:
        op (<MPO>):
        :ket/bra: <MPS>, ket and bra.
        sls (slice): the interval to perform contraction.
        attach_S (str, 'A'):'B' or ''.
        dense (bool): get dense version if True.
        return_histo (bool): return contraction history if True.

    Returns:
        ndarray/csr_matrix,
    '''
    # check data
    if not (bra.labels[0] == mpo.labels[0]
            and len(set([bra.labels[1], ket.labels[1]] + mpo.labels)) == 5
            and mpo.labels[1] == ket.labels[0]):
        raise ValueError(
            'Make site labels identical and bond labels different please!')
    if bra.is_ket or not ket.is_ket:
        raise ValueError()

    res = None
    memory = []
    MHL, OL, ML = bra.get_all(attach_S=attach_S)[sls], O.get_all()[
        sls], ket.get_all(attach_S=attach_S)[sls]
    for MH, O, M in zip(MHL, OL, ML):
        res = (MH * res * O * M) if res is not None else (MH * O * M)
        memory.append(res)
    if return_histo:
        return res, memory
    return res
