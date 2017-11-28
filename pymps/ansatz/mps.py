'''
Matrix Product State.
'''

from __future__ import division
from numpy import *
from scipy.linalg import svd, norm, block_diag
import pdb
import copy
import numbers
from abc import ABCMeta, abstractmethod
from functools import reduce

from ..toolbox.utils import inherit_docstring_from
from ..tensor.tensor import BLabel, Tensor, TensorBase
from ..tensor.tensorlib import tensor_block_diag, check_validity_tensor

__all__ = ['MPSBase', 'MPS', 'BMPS', 'mPS']

ZERO_REF = 1e-12


def _mps_sum(mpses, labels=('s', 'a')):
    '''
    Summation over <MPS>es.

    Args:
        mpses (list of <MPS>): instances to be added.
        labels (list of str): the new labels for added state.

    Returns:
        <MPS>, the added MPS.
    '''
    if len(mpses) == 1:
        return mpses[0]
    elif len(mpses) == 0:
        raise ValueError('At least 1 <MPS> instances are required.')
    assert(all(diff([mps.l for mps in mpses]) == 0))  # the l should be same
    mps0 = mpses[0]
    l = mps0.l
    MLs = [mps.get_all(attach_S='') for mps in mpses]
    hndim = mps0.hndim
    nsite = mps0.nsite
    site_axis = mps0.site_axis

    ML = [tensor_block_diag(mis, axes=(2,) if i == 0 else (
        (0,) if i == nsite - 1 else (0, 2))) for i, mis in enumerate(zip(*MLs))]

    # get S matrix
    if l == 0 or l == nsite:
        S = ones(1, dtype='complex128')
    else:
        S = ones(sum([len(mps.S) for mps in mpses]))
    return mPS(ML, l, S=S, labels=labels, bmg=mps0.bmg if hasattr(mps0, 'bmg') else None)


def _autoset_bms(TL, bmg, bml0=None, reorder=False, check_conflicts=False):
    '''
    Auto setup blockmarkers for <MPS> and <MPO>.

    Args:
        TL (list): a list of tensors.
        bmg (<BlockMarkerGenerator>):
        bml0 (<BlockMarker>, initial blockmarker at left end): bmg.bm0 by default.
        reorder (bool): make sure quantum numbers in bonds are ordered.
        check_conflicts (bool):

    Returns:
        list, tensors with block marker set.
    '''
    nsite = len(TL)
    if nsite == 0:
        return
    bm1 = bmg.bm1_
    is_mps = ndim(TL[0]) == 3
    pm = slice(None)
    for i in range(nsite):
        # cell is a tensor tensor(al,sup,ar), first permute left axes to match the transformation of block marker
        cell = TL[i]
        if reorder:
            cell = cell[pm]
        # get bml
        if i == 0:
            bml = bmg.bm0 if bml0 is None else bml0
        else:
            bml = TL[i - 1].labels[-1].bm

        # setup left, site labels.
        cell.labels[:2] = [
            BLabel(cell.labels[0], bml),
            BLabel(cell.labels[1], bm1)]
        if not is_mps:
            cell.labels[2] = BLabel(cell.labels[2], bm1)

        # get bmr
        cell.autoflow(axis=-1, bmg=bmg, signs=[1, 1, -1] if is_mps else [
                      1, 1, -1, -1], check_conflicts=check_conflicts)
        if reorder:
            cell, (pm,) = cell.b_reorder(axes=(-1,), return_pm=True)

        TL[i] = cell
    return TL


def _auto_label(TL, labels, bm1=None, bms=None):
    is_mps = TL[0].ndim == 3
    for i, M in enumerate(TL):
        lbs = ['%s_%s' % (labels[-1], i), '%s_%s' %
               (labels[0], i), '%s_%s' % (labels[-1], i + 1)]
        if bms is not None:
            bmis = [bms[i]] + [bm1] * (M.ndim - 2) + [bms[i + 1]]
            lbs = [BLabel(s, b) for s, b in zip(lbs, bmis)]
        if not is_mps:
            lbs.insert(2, '%s_%s' % (labels[1], i))
        if isinstance(M, TensorBase):
            M.chlabel(lbs)
        elif isinstance(M, ndarray):
            TL[i] = Tensor(M, labels=lbs)
        else:
            raise TypeError
    return TL


def _replace_cells(mpx, sls, cells):
    start, stop = sls.start, sls.stop
    ncell = len(cells)
    is_mps = hasattr(mpx, 'S')
    if is_mps:
        if mpx.l >= stop:
            mpx.l += ncell - stop - start
        elif mpx.l > start:
            mpx.l = start
            mpx.S = identity(mpx.check_bond(start))
    if is_mps:
        # insert cells
        mpx.ML = mpx.ML[:start] + list(cells) + mpx.ML[stop:]
    else:
        mpx.OL = mpx.OL[:start] + list(cells) + mpx.OL[stop:]
    # adjust labels of original cells
    _auto_label(mpx.ML if is_mps else mpx.OL, labels=mpx.labels)

    # adjust block markers
    if hasattr(mpx, 'bmg'):
        bm0 = mpx.get(start - 1).labels[-1].bm if start > 0 else (
            mpx.get(start).labels[0].bm if ncell == 0 else None)
        _autoset_bms(mpx.ML[start:] if is_mps else mpx.OL[start:],
                     bmg=mpx.bmg, bml0=bm0, reorder=False, check_conflicts=False)


class MPSBase(object, metaclass=ABCMeta):
    '''
    The Base class of Matrix Product state.

    Attributes:
        :hndim: The number of channels on each site.
        :nsite: The number of sites.
        :site_axis/llink_axis/rlink_axis: The specific axes for site, left link, right link.
        :state: The state in the normal representation.
    '''

    @property
    def site_axis(self):
        '''int, axis of site index.'''
        return 1

    @property
    def llink_axis(self):
        '''int, axis of left link index.'''
        return 0

    @property
    def rlink_axis(self):
        '''int, axis of right link index.'''
        return 2

    @property
    @abstractmethod
    def hndim(self):
        '''int, number of state in a single site.'''
        pass

    @property
    @abstractmethod
    def nsite(self):
        '''int, number of sites.'''
        pass

    @property
    @abstractmethod
    def state(self):
        '''1d array, vector representation of this MPS'''
        pass

    @abstractmethod
    def tobra(self, labels):
        '''
        Get the bra conterpart.

        Args:
            labels (list): label strings for site and bond.

        Returns:
            <MPS>,
        '''
        pass

    @abstractmethod
    def toket(self, labels):
        '''
        Get the ket conterpart.

        Args:
            labels (list): label strings for site and bond.

        Returns:
            <MPS>,
        '''
        pass


class MPS(MPSBase):
    '''
    Matrix product states.

    Attributes:
        ML (list of 3D array): the sequence of A/B-matrices.
        :l/S: int, 1D array, the division point of left and right scan, and the singular value matrix at the division point.
            Also, it is the index of the non-unitary M-matrix(for the special case of l==N, S is attached to the right side of N-1-th matrix)
        is_ket (bool): It is a ket if True else bra.
        labels (len-2 list of str, the labels for auto-labeling in MPS [site): link].
    '''

    def __init__(self, ML, l, S, is_ket=True, labels=['s', 'a']):
        assert(ndim(S) == 1)
        assert(len(labels) == 2)
        self.ML = ML
        self.labels = list(labels)
        self.l = l
        self.S = S
        self.is_ket = is_ket

    def __str__(self):
        string = '<MPS,%s>\n' % (self.nsite)
        string += '\n'.join(['  A[s=%s] (%s x %s) (%s,%s,%s)' % (
            a.shape[self.site_axis], a.shape[self.llink_axis], a.shape[self.rlink_axis],
            a.labels[self.llink_axis], a.labels[self.site_axis], a.labels[self.rlink_axis]
        ) for a in self.ML[:self.l]]) + '\n'
        string += '  S      %s\n' % (self.S.shape,)
        string += '\n'.join(['  B[s=%s] (%s x %s) (%s,%s,%s)' % (
            a.shape[self.site_axis], a.shape[self.llink_axis], a.shape[self.rlink_axis],
            a.labels[self.llink_axis], a.labels[self.site_axis], a.labels[self.rlink_axis]
        ) for a in self.ML[self.l:]])
        return string

    def __add__(self, target):
        if isinstance(target, MPS):
            return _mps_sum([self, target], labels=self.labels[:])
        else:
            raise TypeError('Can not add <MPS> with %s' % target.__class__)

    def __radd__(self, target):
        if isinstance(target, MPS):
            return target.__add__(self)
        else:
            raise TypeError('Can not add %s with <MPS>' % target.__class__)

    def __sub__(self, target):
        if isinstance(target, MPS):
            return self.__add__(-target)
        else:
            raise TypeError('Can not subtract <MPS> with %s' %
                            target.__class__)

    def __rsub__(self, target):
        if isinstance(target, MPS):
            return target.__sub__(self)
        else:
            raise TypeError('Can not subtract %s with <MPS>' %
                            target.__class__)

    def __mul__(self, target):
        hndim = self.hndim
        site_axis = self.site_axis
        if isinstance(target, numbers.Number):
            mps = self.toket(labels=self.labels[:]) if self.is_ket else self.tobra(
                labels=self.labels[:])
            mps.S = self.S * target
            return mps
        elif isinstance(target, MPS):
            if self.is_ket or not target.is_ket:
                raise Exception(
                    'Not implemented for multipling ket on the left side.')
            S = identity(1)
            for mi, tmi in zip(self.get_all(attach_S=''), target.get_all(attach_S='')):
                # need some check!
                S = sum([mi.take(j, axis=site_axis).toarray().T.dot(S).dot(
                    tmi.take(j, axis=site_axis).toarray()) for j in range(hndim)], axis=0)
            return S
        else:
            raise TypeError('Can not multiply <MPS> with %s' %
                            target.__class__)

    def __rmul__(self, target):
        if isinstance(target, numbers.Number):
            return self.__mul__(target)
        elif isinstance(target, MPS):
            return target.__mul__(self)
        else:
            raise TypeError('Can not multiply %s with <MPS>' %
                            target.__class__)

    def __imul__(self, target):
        if isinstance(target, numbers.Number):
            self.S = self.S * target
            return self
        else:
            raise TypeError('Can not i-multiply by %s' % target.__class__)

    def __neg__(self):
        return -1 * self

    def __truediv__(self, target):
        hndim = self.hndim
        if isinstance(target, numbers.Number):
            mps = self.toket(labels=self.labels[:]) if self.is_ket else self.tobra(
                labels=self.labels[:])
            mps.S = self.S / target
            return mps
        else:
            raise TypeError('Can not divide <MPS> with %s' % target.__class__)

    def __itruediv__(self, target):
        if isinstance(target, numbers.Number):
            self.S = self.S / target
            return self
        else:
            raise TypeError('Can not i-divide by %s' % target.__class__)

    def __lshift__(self, k):
        '''Left move l-index by k.'''
        tol = ZERO_REF
        maxN = Inf
        if isinstance(k, tuple):
            k, tol, maxN = k
        return self.canomove(-k, tol=tol, maxN=maxN)

    def __rshift__(self, k):
        '''Right move l-index by k.'''
        tol = ZERO_REF
        maxN = Inf
        if isinstance(k, tuple):
            k, tol, maxN = k
        return self.canomove(k, tol=tol, maxN=maxN)

    @property
    def hndim(self):
        '''The number of state in a single site.'''
        return self.ML[0].shape[self.site_axis]

    @property
    @inherit_docstring_from(MPSBase)
    def nsite(self):
        return len(self.ML)

    @property
    @inherit_docstring_from(MPSBase)
    def state(self):
        ML = self.get_all()
        res = reduce(lambda x, y: x * y, ML)
        return asarray(res.ravel())

    def get(self, siteindex, attach_S='', *args, **kwargs):
        '''
        Get the tensor element for specific site.

        Args:
            siteindex (int): the index of site.
            attach_S (bool): attach the S-matrix to

                * ''  -> don't attach to any block.
                * 'A' -> right most A block.
                * 'B' -> left most B block.

        Returns:
            <Tensor>,
        '''
        assert(attach_S in ['A', 'B', ''])
        if siteindex > self.nsite or siteindex < 0:
            raise ValueError('l=%s out of bound!' % siteindex)
        res = self.ML[siteindex]
        if attach_S == 'A' and siteindex == self.l - 1:
            res = res.mul_axis(self.S, self.rlink_axis)
        elif attach_S == 'B' and siteindex == self.l:
            res = res.mul_axis(self.S, self.llink_axis)
        return res

    def set(self, siteindex, A, *args, **kwargs):
        '''
        Get the matrix for specific site.

        Args:
            siteindex (int): the index of site.
            A (<Tensor>): the data
        '''
        if siteindex > self.nsite or siteindex < 0:
            raise ValueError('l=%s out of bound!' % siteindex)
        self.ML[siteindex] = A

    def insert(self, pos, cells):
        '''
        Insert cells into MPS.

        Args:
            cells (list): tensors.
        '''
        _replace_cells(self, slice(pos, pos), cells)

    def remove(self, start, stop):
        '''
        Remove a segment from MPS.

        Args:
            start (int):
            stop (int):
        '''
        _replace_cells(self, slice(start, stop), [])

    def check_link(self, l):
        '''
        The bond dimension for l-th link.

        Args:
            l (int): the bond index.

        Returns:
            int, the bond dimension.
        '''
        if l == self.nsite:
            return self.ML[-1].shape[self.rlink_axis]
        elif l >= 0 and l < self.nsite:
            return self.ML[l].shape[self.llink_axis]
        else:
            raise ValueError('Link index out of range!')

    def get_all(self, attach_S=''):
        '''
        Get the concatenation of A and B sectors.

        Args:
            attach_S (bool): attach the S-matrix to

                * ''  -> don't attach to any block.
                * 'A' -> right most A block.
                * 'B' -> left most B block.

        Returns:
            list,
        '''
        assert(attach_S in ['A', 'B', ''])
        ML = self.ML[:]
        l = self.l
        nsite = self.nsite
        if attach_S == '':
            attach_S = 'A' if l != 0 else 'B'
        # fix no S cases
        if l == 0 and attach_S == 'A':
            attach_S = 'B'
        if l == nsite and attach_S == 'B':
            attach_S = 'A'

        if attach_S == 'A':
            ML[l - 1] = ML[l - 1].mul_axis(self.S, self.rlink_axis)
        elif attach_S == 'B':
            ML[l] = ML[l].mul_axis(self.S, axis=self.llink_axis)
        return ML

    def canomove(self, nstep, tol=ZERO_REF, maxN=Inf):
        '''
        Move l-index by one with specific direction.

        Args:
            nstep (int): move l nstep towards right.
            tol (float): the tolerence for compression.
            maxN (int): the maximum dimension.

        Returns:
            float, approximate truncation error.
        '''
        use_bm = hasattr(self, 'bmg')
        nsite = self.nsite
        hndim = self.hndim
        llink_axis, rlink_axis, site_axis = self.llink_axis, self.rlink_axis, self.site_axis
        # check and prepair data
        if self.l + nstep > nsite or self.l + nstep < 0:
            raise ValueError('Illegal Move!')
        right = nstep > 0
        acc = 1.
        for i in range(abs(nstep)):
            # prepair the tensor, Get A,B matrix
            self.l = (self.l + 1) if right else (self.l - 1)
            if right:
                A = self.ML[self.l - 1].mul_axis(self.S, llink_axis)
                if self.l == nsite:
                    S = sqrt((A**2).sum())
                    self.S = array([S])
                    self.ML[-1] = A / S
                    return 1 - acc
                B = self.ML[self.l]
            else:
                B = self.ML[self.l].mul_axis(self.S, rlink_axis)
                if self.l == 0:
                    S = sqrt((B**2).sum())
                    self.S = array([S])
                    self.ML[0] = B / S
                    return 1 - acc
                A = self.ML[self.l - 1]
            cbond_str = B.labels[llink_axis]
            # contract AB,
            AB = A * B
            U, S, V = AB.svd(cbond=2, cbond_str=cbond_str, bmg=self.bmg if hasattr(
                self, 'bmg') else None, signs=[1, 1, -1, 1])
            bdim = len(S)

            # truncation
            if maxN < S.shape[0]:
                tol = max(S[maxN], tol)
            kpmask = S > tol
            acc *= (1 - sum(S[~kpmask]**2))
            # unpermute blocked U,V and get c label
            self.ML[self.l - 1], self.S, self.ML[self.l] = U.take(
                kpmask, axis=-1), S[kpmask], V.take(kpmask, axis=0)
        return 1 - acc

    def use_bm(self, bmg, sharedata=True):
        '''
        Use <BlockMarker> to indicate block structure.

        Args:
            bmg (<BlockMarkerGenerator>): the generator of block markers.
            sharedata (bool): the new <BMPS> will share the data with current one if True.

        Returns:
            <BMPS>,
        '''
        mps = mPS([ai.make_copy(copydata=not sharedata) for ai in self.ML], self.l,
                  self.S if sharedata else self.S[...], is_ket=self.is_ket, labels=self.labels[:], bmg=bmg)
        return mps

    @inherit_docstring_from(MPSBase)
    def toket(self, labels=None):
        if labels is None:
            labels = self.labels[:]

        mps = mPS([ai.make_copy(copydata=False) if self.is_ket else ai.conj() for ai in self.ML], self.l,
                  self.S if self.is_ket else self.S.conj(), is_ket=True, labels=labels)
        return mps

    @inherit_docstring_from(MPSBase)
    def tobra(self, labels=None):
        # get new labels,
        if labels is None:
            labels = self.labels[:]

        mps = mPS([ai.make_copy(copydata=False) if not self.is_ket else ai.conj() for ai in self.ML], self.l,
                  self.S if not self.is_ket else self.S.conj(), is_ket=False, labels=labels)
        return mps

    def tovidal(self):
        '''
        Transform to the Vidal form.
        '''
        llink_axis, rlink_axis = self.llink_axis, self.rlink_axis
        nsite = self.nsite
        hndim = self.hndim
        LL = []
        GL = []
        factor = norm(self.S)
        S = self.S / factor
        LL.append(S)
        for i in range(self.l - 1, -1, -1):
            UL = U0 * LL[-1] if i != self.l - 1 else diag(S)
            A = self.ML[i]
            A = (A.reshape([-1, A.shape[2]]).dot(UL)
                 ).reshape([-1, A.shape[2] * hndim])
            U0, L, V = svd(A, full_matrices=False)
            LL.append(L)
            GL.append(V.reshape(self.ML[i].shape) /
                      LL[-2].reshape([-1] + [1] * (2 - rlink_axis)))
        LL, GL = LL[::-1], GL[::-1]
        for i in range(nsite - self.l):
            LV = LL[-1][:, newaxis] * V0 if i != 0 else diag(S)
            B = self.ML[self.l + i]
            B = LV.dot(B.reshape([B.shape[0], -1])
                       ).reshape([B.shape[0] * hndim, -1])
            U, L, V0 = svd(B, full_matrices=False)
            LL.append(L)
            GL.append(U.reshape(self.ML[self.l + i].shape) /
                      LL[-2].reshape([-1] + [1] * (2 - llink_axis)))
        if self.l > 0:
            factor = factor * U0
        if self.l < nsite:
            factor = factor * V0
        vmps = VidalMPS(
            GL, LL[1:-1], labels=self.labels[:], factor=factor.item())
        return vmps

    def chlabel(self, labels):
        '''
        Change the label of specific axis.

        Parametrs:
            labels (list): the new labels.
        '''
        self.labels = labels
        _auto_label(self.ML, labels)

    def query(self, serie):
        '''
        Query the magnitude of a state.

        Args:
            serie (1d array): sgimas(indices of states on sites).

        Returns:
            number, the amplitude.
        '''
        state = identity(1)
        site_axis = self.site_axis
        for si, Mi in list(zip(serie, self.get_all(attach_S='B')))[::-1]:
            state = Mi.take(si, axis=site_axis).dot(state)
        return state.item()

    def compress(self, tol=1e-8, maxN=200, niter=3):
        '''
        Compress this state.

        Args:
            tol (float): the tolerence used for compressing.
            maxN (int): the maximum retained states.
            niter (int): number of iterations.

        Returns:
            float, approximate truncation error.
        '''
        nsite, l = self.nsite, self.l
        M = self.check_link(nsite // 2)
        dM = max(M - maxN, 0)
        acc = 1.
        for i in range(niter):
            m1 = maxN + int(dM * ((niter - i - 0.5) / niter))
            m2 = maxN + int(dM * ((niter - i - 1.) / niter))
            acc *= 1 - (self >> (nsite - l, tol, m1))
            acc *= 1 - (self << (nsite - l, tol, m2))
            acc *= 1 - (self << (l, tol, m1))
            acc *= 1 - (self >> (l, tol, m2))
        print('Compression of MPS Done!')
        return 1 - acc

    def recanonicalize(self, left=True, tol=1e-12, maxN=Inf):
        '''
        Trun this MPS into canonical form.

        Args:
            left (bool, use left canonical if True): else right canonical
            tol (float): the tolerence.
            maxN (int): the maximum retained states.
        '''
        nsite, l = self.nsite, self.l
        if not left:
            self >> (nsite - l, tol, maxN)
            self << (nsite, tol, maxN)
            self >> (l, tol, maxN)
        else:
            self << (l, tol, maxN)
            self >> (nsite, tol, maxN)
            self << (nsite - l, tol, maxN)
        return self


class BMPS(MPS):
    '''
    MPS with block structure.

    Attributes:
        bmg (<BlockMarkerGenerator>): the block infomation manager.

        *see <MPS> for more.*
    '''

    def __init__(self, ML, l, S, bmg, **kwargs):
        super(BMPS, self).__init__(ML, l, S, **kwargs)
        self.bmg = bmg

    def unuse_bm(self, sharedata=True):
        '''
        Get the non-block version of current ket.

        Args:
            sharedata (bool): the new <MPS> will share the data with current one if True.

        Returns:
            <MPS>,
        '''
        mps = mPS([ai.make_copy(copydata=not sharedata) for ai in self.ML], self.l,
                  self.S if sharedata else self.S[...], is_ket=self.is_ket, labels=self.labels[:])
        return mps

    @inherit_docstring_from(MPS)
    def toket(self, labels=None):
        if labels is None:
            labels = self.labels[:]

        mps = mPS([ai.make_copy(copydata=False) if self.is_ket else ai.conj() for ai in self.ML], self.l,
                  self.S if self.is_ket else self.S.conj(), is_ket=True, labels=labels, bmg=self.bmg)
        return mps

    @inherit_docstring_from(MPS)
    def tobra(self, labels=None):
        if labels is None:
            labels = self.labels[:]

        mps = mPS([ai.make_copy(copydata=False) if not self.is_ket else ai.conj() for ai in self.ML], self.l,
                  self.S if not self.is_ket else self.S.conj(), is_ket=False, labels=labels, bmg=self.bmg)
        return mps


def mPS(ML, l, S, is_ket=True, labels=['s', 'a'], bmg=None, bms=None):
    '''
    Construct MPS.
    '''
    _auto_label(ML, labels, bms=bms, bm1=None if bmg is None else bmg.bm1_)
    if bmg is None:  # a normal MPS
        return MPS(ML, l, S, is_ket=is_ket, labels=labels)
    else:
        if isinstance(ML[0], Tensor) and bms is None:
            _autoset_bms(ML, bmg)
        return BMPS(ML, l, S, is_ket=is_ket, labels=labels, bmg=bmg)
