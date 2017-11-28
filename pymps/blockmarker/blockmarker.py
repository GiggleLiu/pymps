'''
Turn a symmetric matrix into block diagonal form.
'''

from numpy import *
from abc import ABCMeta, abstractmethod
import scipy.sparse as sps

from . import fblock as flib
from ..spaceconfig import SuperSpaceConfig, SpinSpaceConfig

__all__ = ['BlockMarker', 'BlockMarkerGenerator', 'SimpleBMG', 'trunc_bm']


def _group_qns(qns):
    '''
    Group up qns and get the size of each qn, the array version.

    Args:
        qns_sorted (list): the sorted qns in 1D array.
        N (int): the size of qns_sorted.

    Returns:
        tuple of (gqns, nnr), nnr are the sizes of each block and gqns are the qns of groups.
    '''
    if isinstance(qns[0], (int, int32, bool)):
        nnr, gqns, N = flib.fgroup_iqns(qns)
        return gqns[:N], nnr[:N]
    elif isinstance(qns[0], (float, float64)):
        nnr, gqns, N = flib.fgroup_dqns(qns)
        return gqns[:N], nnr[:N]
    elif hasattr(qns[0], '__iter__'):
        # support the null qns
        if qns.shape[1] == 0:
            return qns[:1], array([len(qns)])
        dtype = array(qns[0]).dtype
        nnr, gqns, N = flib.fgroup_aqns(qns)
        gqns = array(gqns[:N], dtype=dtype)
        return gqns, nnr[:N]
    elif isinstance(qns[0], str):
        qn_pre = nan
        temp = 0
        gqns, nnr = [], []
        for i, nqn in enumerate(qns):
            if i == 0 or not all(nqn == qn_pre):
                gqns.append(tuple(nqn))
                if i != 0:
                    nnr.append(temp)
                temp = 1
                qn_pre = nqn
            else:
                temp += 1
        nnr.append(temp)
        return gqns, nnr
    else:
        raise TypeError('can not group qns of type %s' % type(qns[0]))


class BlockMarker(object):
    '''
    The BlockMarker to tell block structure.

    Construct:
        BlockMarker(pm,Nr,qns)

    Attributes:
        Nr (1D array): the division points.
        qns (list): the qns of blocks.
        N (integer, the matrix dimension): readonly.
        nr (1D array(int32), the subblock dimensions): readonly.
        nblock (integer, the number of blocks): readonly.
    '''

    def __init__(self, Nr, qns=None):
        self.Nr = asarray(Nr)
        nblock = self.nblock
        if qns is None:
            qns = arange(len(Nr) - 1)[:, newaxis]
        assert(ndim(qns) == 2 and len(qns) == nblock)
        self.qns = asarray(qns)

    @property
    def N(self):
        '''The matrix dimension.'''
        return self.Nr[-1]

    @property
    def nr(self):
        '''The block sizes.'''
        return diff(self.Nr)

    @property
    def nblock(self):
        '''Number of blocks.'''
        return len(self.Nr) - 1

    def __str__(self):
        nr = self.nr
        return '<BlockMarker>(%s)\n%s' % (self.nblock, '\t'.join(['%s(%s)' % (qn, n) for qn, n in zip(self.qns, nr)]))

    def __repr__(self):
        nr = self.nr
        return '<BlockMarker>(block: %s, size: %s)' % (self.nblock, self.N)

    def __eq__(self, target):
        return all(target.Nr == self.Nr) and all(target.qns == self.qns)

    def __add__(self, target):
        Nr = concatenate([self.Nr, self.N + target.Nr[1:]])
        qns = concatenate([self.qns, target.qns], axis=0)
        return BlockMarker(Nr=Nr, qns=qns)

    def __radd__(self, target):
        return target.__add__(self)

    def get_slice(self, i):
        '''
        Get the slice of specific block.

        Args:
            i (int): the block index.

        Returns:
            slice,
        '''
        return slice(self.Nr[i], self.Nr[i + 1])

    def has_qn(self, qn):
        '''Has a qn or not.'''
        return any(all(self.qns == qn, axis=1))

    def index_qn(self, qn):
        '''Get the index of specific qn.'''
        if ndim(qn) < ndim(self.qns):
            return where(all(self.qns == qn, axis=1))[0]
        else:
            nv = self.qns.shape[-1]
            if nv == 0:
                return zeros(len(qn), dtype=self.qns.dtype)
            matches = [(self.qns[:, i:i + 1] == qn[:, i]) for i in range(nv)]
            match = matches[0]
            for i in range(1, nv):
                match = match & matches[i]
            bidpos = where(match)
            bid = bidpos[0][argsort(bidpos[1])]
            return bid

    def blocksize(self, i):
        '''
        Query the block size

        Args:
            :i: The specific index.

        Returns:
            integer, the size of specific block.
        '''
        return self.Nr[i + 1] - self.Nr[i]

    def ind2b(self, i):
        '''
        Parse index to block-id and inner-id

        Args:
            i (int): the index.

        Returns:
            tuple, (bid,iid)
        '''
        bid = searchsorted(self.Nr, i + 1) - 1
        iid = i - self.Nr[bid]
        return bid, iid

    def b2ind(self, bid, iid):
        '''
        Parse index to block-id and inner-id

        Args:
            bid (ndarray): iid is the index
            iid (int): the index in the block.

        Returns:
            int, the matrix index.
        '''
        ind = self.Nr[bid] + iid
        return ind

    def extract_block(self, A, ij, axes=None):
        '''
        Fetch the specific block by qn from target_matrix.

        Args:
            A (ndarray): the target matrix/vector/tensor.
            ij (tuple): the block indices/qns
            axes (tuple): the axes to perform blockization.

        Returns:
            matrix, the specific sub-block.
        '''
        # prepair indices
        assert(axes is None or len(ij) == len(axes))

        Nr = self.Nr
        nblock = self.nblock
        mdim = ndim(A)
        is_sparse = sps.issparse(A)
        if axes is None:
            axes = arange(mdim)
        assert(all(array(ij) < nblock) and len(ij) == len(axes))

        if not is_sparse:
            slices = [slice(None)] * mdim
            for i, axis in zip(ij, axes):
                slices[axis] = slice(Nr[i], Nr[i + 1])
            return A[tuple(slices)]
        else:  # mdim==2
            res = A
            for i, axis in zip(ij, axes):
                if axis == 0:
                    res = res.tocsr()[Nr[i]:Nr[i + 1]]
                else:
                    res = res.tocsc()[:, Nr[i]:Nr[i + 1]]
            return res

    def check_blockdiag(self, A, tol=1e-8):
        '''
        Check a matrix is block diagonal or not.

        Args:
            A (matrix): the input matrix.
            tol (float): the tolerence.

        Returns:
            bool, True if matrix A is block-diagonal.
        '''
        A = sps.coo_matrix(abs(A) > tol)
        rows = A.row
        cols = A.col
        for ib in range(self.nblock):
            start, end = self.Nr[ib], self.Nr[ib + 1]
            rowmask = (rows >= start) & (rows < end)
            colmask = (cols >= start) & (rows < end)
            if not all(rowmask == colmask):
                return False
        return True

    def inflate(self):
        '''Expand all qns into 1 qn-1 bond representation.'''
        N = self.N
        Nr = arange(N + 1)
        qns = repeat(self.qns, self.nr, axis=0)
        return BlockMarker(Nr=Nr, qns=qns)

    def sort(self, reverse=False, return_info=False):
        '''
        Get blockmarker with sorted qns.

        Args:
            reverse (bool): reverse order if true.
            return_info (bool, return information(`pm`):`pm_b`) if true.

        Returns:
            <BlockMarker>,
        '''
        if self.qns.shape[1] == 0 or self.qns.shape[0] == 0:
            bm = BlockMarker(self.Nr, self.qns)
            if return_info:
                info = dict(pm=arange(self.N), pm_b=arange(self.nblock))
                return bm, info
            else:
                return bm
        pm_b = lexsort(self.qns.T[::-1])
        if reverse:
            pm_b = pm_b[::-1]
        qns = self.qns[pm_b]
        nr = self.nr[pm_b]
        bm = BlockMarker(append([0], cumsum(nr)), qns)
        if return_info:
            info = dict(pm=concatenate(
                array(split(arange(self.N), self.Nr[1:-1]))[pm_b]), pm_b=pm_b)
            return bm, info
        else:
            return bm

    def compact_form(self, return_info=False):
        '''
        Recombine indices and turn same indices into one.

        Args:
            return_info (bool): return grouping information.

        Returns:
            <BlockMarker>,
        '''
        if self.qns.shape[0] == 0 or self.qns.shape[1] == 0:
            return BlockMarker(self.Nr, self.qns)
        # group qns and get nr
        unqns, nnr = _group_qns(self.qns)
        NNr = cumsum(nnr)
        Nr_sorted = cumsum(self.nr)
        Nr = append([0], Nr_sorted[NNr - 1])
        bm = BlockMarker(Nr=Nr, qns=unqns)
        if return_info:
            return bm, append([0], NNr)
        else:
            return bm


class BlockMarkerGenerator(object):
    '''
    The <BlockMarker> Generator.

    Attributes:
        start_qn (integer/tuple/str): the initial qn.
        qns1 (1Darray): the qns for a single site.
        spaceconfig (<SpaceConfig>): the Hilbert space for single site.
    '''
    INF = 10000

    def __init__(self, spaceconfig, start_qn, qns1):
        self.spaceconfig = spaceconfig
        self.start_qn = start_qn
        self.qns1 = asarray(qns1)

    @property
    def bm1(self):
        '''
        The Block marker for single site.
        '''
        # cope with null block marker
        if self.qstring == '':
            return BlockMarker(Nr=array([0, self.spaceconfig.hndim]), qns=zeros([1, 0], dtype=self.qns1.dtype))
        # first get the permutation and sort the qns
        nqns = asarray(self.qns1)
        pm = lexsort(nqns.T[::-1])
        nqns_sorted = nqns[pm]
        # group qns and get nr
        unqns, nnr = _group_qns(nqns_sorted)
        Nr = append([0], cumsum(nnr))
        return BlockMarker(Nr=Nr, qns=unqns)

    @property
    def bm0(self):
        '''
        The Block marker for empty.
        '''
        qns = array([self.start_qn])
        return BlockMarker(Nr=array([0, 1], dtype=self.qns1.dtype), qns=qns)

    @property
    def bm1_(self):
        '''
        The Block marker for single site, the primary version.
        '''
        # group qns and get nr
        unqns, nnr = _group_qns(self.qns1)
        return BlockMarker(Nr=append([0], cumsum(nnr)), qns=unqns)

    @abstractmethod
    def bcast_add(self, lbs1, lbs2):
        '''
        add qns (lbs1,lbs2), this is "kronicker type" add.

        Args:
            :lbs1,lbs2: ndarray, qns to be added.

        Returns:
            ndarray, new qns.
        '''
        pass

    @abstractmethod
    def bcast_sub(self, lbs1, lbs2):
        '''
        subtract qns (lbs1,lbs2), this is "kronicker type" subtract.

        Args:
            :lbs1,lbs2: ndarray, qns to be subtracted.

        Returns:
            ndarray, new qns.
        '''
        pass

    @abstractmethod
    def neg_bm(self, bm):
        '''
        Get the negative block marker.

        Args:
            bm (<BlockMarker>):

        Returns:
            <BlockMarker>,
        '''
        pass

    @abstractmethod
    def join_bms(self, bms, signs=None):
        '''
        Join two blockmarkers in kronecker form(merge two axes).

        Args:
            bms (<BasicBlockMarker>): the target qned block marker.
            signs (1darray): the signs for each blockmarker.

        Returns:
            (<BlockMarker>,pm), the new block marker and the permutation serie.
        '''
        pass

    def shift_bm(self, bm, qn):
        '''
        Shift quantum numbers.

        Args:
            bm (<BlockMarker>):
            qn (1darray/number): the shift of quantum numbers.
        '''
        bm.qns = self.bcast_add(bm.qns, qn)
        return bm

    def invert_bm(self, bm, qn):
        '''
        Invert quantum numbers.

        Args:
            bm (<BlockMarker>):
        '''
        bm.qns = self.trim_qns(-bm.qns)
        return bm

    def update1(self, bm, right_side=True):
        '''
        Update the blockmarker after expansion of 1 site.

        Args:
            bm (<BlockMarker>/None, the block marker before update): leave `None` for first update.
            right_side (bool): Expand the new site at right hand site if True.

        Returns:
            (<BlockMarker>, pm), the updated block marker and permutation matrix.
        '''
        if bm is None:
            bm = self.bm0
        return self.join_bms([bm, self.bm1_]) if right_side else self.join_bms([self.bm1_, bm])

    def random_bm(self, nsite=10, trunc_rate=0):
        '''
        Generate a random BlockMarker.

        Args:
            nsite (int): update for n times.
            trunc_rate (float): the rate for truncation.

        Returns:
            <BlockMarkerGenerator>, <BlockMarker>
        '''

        # get generator and initial block marker
        bm = self.bm1

        # expand and truncate for several times.
        for i in range(nsite - 1):
            kpmask = ones(bm.N, dtype='bool')
            if trunc_rate != 0:
                kpmask[random.random(bm.N) < trunc_rate] = False
                if not any(kpmask):
                    kpmask[0] = True
            bm = self.update1(trunc_bm(bm, kpmask), right_side=True)
            bm = bm.sort().compact_form()
        return bm


class SimpleBMG(BlockMarkerGenerator):
    '''
    Simple block marker generator, with additive quantum number.

    Attributes:
        per (1Darray): periodicity for each quantum number.
        qstring (str, one or multiple in ['M','Q','P'):'R']

            * 'M': nup-ndn, or 2*Sz
            * 'Q': nup+ndm,
            * 'P': charge parity, Q%2
            * 'R': spin parity, M%4
            * '': null.

        spaceconfig (<SpaceConfig>): the Hilbert space for single site.
        :bm0,bm1,bm1_: <BlockMarker>, the block marker for 0-site, 1-site and 1-site reordered.(readonly)
    '''

    def __init__(self, qstring, spaceconfig):
        self.qstring = qstring
        hndim = spaceconfig.hndim
        nstr = len(qstring)
        spinspace = len(spaceconfig.config) == 2
        config = spaceconfig.ind2config(arange(hndim))
        # get the single site quantum number.
        qns1 = zeros([hndim, nstr], dtype='int32')
        self.per = zeros(nstr, dtype='int32')
        for istr, qtype in enumerate(qstring):
            if qtype == 'M' or qtype == 'R':
                self.per[istr] = self.INF if qtype == 'M' else 4
                # for quantum number nup-ndn
                nspin = spaceconfig.nspin
                for i in range(nspin):
                    spini = nspin - 2 * i - 1  # nspin/2.-i-0.5
                    if spinspace:
                        qns1[:, istr] += sum(spini * (config == i), axis=-1)
                    else:
                        imask = spaceconfig.subspace(spinindex=i)
                        qns1[:, istr] += config[:, imask].sum(axis=-1) * spini
                    if qtype == 'R':
                        qns1[:, istr] %= 4
            elif qtype == 'Q' or qtype == 'P':
                self.per[istr] = self.INF if qtype == 'Q' else 2
                qns1[:, istr] = sum(config, axis=-1) % self.per[istr]
        super(SimpleBMG, self).__init__(spaceconfig, zeros(
            len(self.qstring), dtype='int32'), qns1)

    def __str__(self):
        return '<SimpleBMG(%s=>%s)>' % (self.qstring, self.qns1.shape[0])

    def bcast_add(self, lbs1, lbs2):
        res = (asarray(lbs1)[:, newaxis, :] + asarray(lbs2)
               [newaxis, :, :]).reshape([-1, len(self.qstring)])
        return self.trim_qns(res)

    def bcast_sub(self, lbs1, lbs2):
        res = (asarray(lbs1)[:, newaxis, :] - asarray(lbs2)
               [newaxis, :, :]).reshape([-1, len(self.qstring)])
        return self.trim_qns(res)

    def neg_bm(self, bm):
        return BlockMarker(self.trim_qns(-bm.qns), Nr=bm.Nr)

    def join_bms(self, bms, signs=None):
        # check datas
        if signs is None:
            signs = ones(len(bms), dtype='int32')
        if len(signs) != len(bms) or len(bms) < 2:
            raise ValueError
        # support null block marker
        if self.qstring == '':
            Nr = array([0, prod([bm.N for bm in bms])])
            return BlockMarker(Nr=Nr, qns=zeros([1, 0], dtype=self.qns1.dtype))

        # inflate and get size-1 qns(robuster!)
        qns = bms[0].inflate().qns * signs[0]
        for i, (bm, sign) in enumerate(zip(bms[1:], signs[1:])):
            if i < len(signs) - 2:
                bm = bm.inflate()
            qns = self.bcast_add(qns, bm.qns * sign)

        Nr = append([0], cumsum(concatenate(
            [bms[-1].nr] * prod([bm.N for bm in bms[:-1]]))))
        return BlockMarker(Nr=Nr, qns=qns)

    def trim_qns(self, qns):
        '''
        Trim qns, to cope data out of range, qn>self.per.

        Args:
            :qns: 2d array.

        Returns:
            2d array, trimed qns.
        '''
        pmask = self.per != self.INF
        qns[..., pmask] = mod(qns[..., pmask], self.per[pmask])
        return qns


def trunc_bm(bm, kpmask):
    '''
    Truncate the <BlockMarker>.

    Args:
        bm (<BlockMarker>): the block marker.
        kpmask (1D array of bool): the mask of the kept part.

    Returns:
        <BlockMarker>, the block marker after truncation.
    '''
    assert(ndim(kpmask) == 1)
    Nr = bm.Nr
    nr = []
    for start, end in zip(Nr[:-1], Nr[1:]):
        nr.append(sum(kpmask[start:end]))
    nr = array(nr)
    bkp = nr > 0
    Nr2 = append([0], cumsum(nr[bkp]))
    return BlockMarker(Nr=Nr2, qns=array(bm.qns)[bkp])
