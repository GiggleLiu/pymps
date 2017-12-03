'''
TensorBase and dense Tensor Class.
'''

import numpy as np
from numpy import array
from scipy.linalg import svd, eigh, eig
from abc import ABCMeta, abstractmethod
import copy
import pdb
import numbers
import itertools

from .basic import ldu, dpl
from ..blockmarker import join_bms, BlockMarker, block_diag

__all__ = ['TensorBase', 'Tensor', 'tdot', 'BLabel']

ZERO_REF = 1e-12


class BLabel(str):
    '''
    Label string with block marker.

    Args:
        value (str): the string as a label.
        bm (:obj:`BlockMarker`): block marker.

    Attibutes:
        value (str): the string as a label.
        bm (:obj:`BlockMarker`): block marker.
    '''
    def __new__(cls, value, bm):
        obj = str.__new__(cls, value)
        obj.bm = bm
        return obj

    def __getnewargs__(self):
        return (str(self), self.bm)

    def __getstate__(self):
        pass

    # def __copy__(self):
    #    return BLabel(self,self.bm)

#    def __deepcopy__(self, memo):
#        return BLabel(self,copy.deepcopy(self.bm))

    def chbm(self, bm):
        '''Get a new <BLabel> with different block marker.'''
        return BLabel(str(self), bm)

    def chstr(self, s):
        '''Get a new <BLabel> with different string.'''
        return BLabel(s, self.bm)


def tdot(tensor1, tensor2):
    '''
    Tensor dot between two tensors, faster than contract in most case?

    Args:
        tensor1,tensor2 (:obj:`Tensor`): two tensors to contract.

    Returns:
        :obj:`Tensor`: output tensor.
    '''
    inner1, inner2, outer1, outer2 = _same_diff_labels(tensor1.labels, tensor2.labels)
    res = np.tensordot(tensor1, tensor2, axes=(inner1, inner2))
    res = Tensor(res, labels=[tensor1.labels[i] for i in outer1]+[tensor2.labels[i] for i in outer2])
    return res


def _same_diff_labels(lb1s, lb2s):
    # detect same labels
    inner1, inner2 = [], []
    outer1 = []
    for i1, lb1 in enumerate(lb1s):
        if lb1 in lb2s:
            i2 = lb2s.index(lb1)
            inner1.append(i1)
            inner2.append(i2)
        else:
            outer1.append(i1)
    outer2 = [i2 for i2 in range(len(lb2s)) if i2 not in inner2]
    return inner1, inner2, outer1, outer2


class TensorBase(object, metaclass=ABCMeta):
    '''
    The base abstract class for tensor.
    '''

    @property
    def ndim(self):
        '''The dimension.'''
        pass

    @abstractmethod
    def todense(self):
        '''Parse this tensor to dense version - <Tensor>.'''
        pass

    @abstractmethod
    def toarray(self):
        '''Parse this tensor into an array.'''
        pass

    @abstractmethod
    def mul_axis(self, vec, axis):
        '''
        Multiply a vector on specific axis.

        Args:
            vec (1d array): the vector.
            axis (int/str): the axis or label.

        Returns:
            <TensorBase>
        '''
        pass

    @abstractmethod
    def make_copy(self, labels=None, copydata=True):
        '''
        Make a copy of this tensor.

        Args:
            labels (list/None, the new labels): use the old ones if None.
            copydata (bool): copy the data to the new tensor if True.

        Returns:
            <TensorBase>
        '''
        pass

    @abstractmethod
    def take(self, key, axis):
        '''
        Take subspace from this Tensor.

        Args:
            key (0-d/1-d array): the key
            axis (int): the axis to take.

        Returns:
            <TensorBase>
        '''
        pass

    @abstractmethod
    def chorder(self, order):
        '''
        Reorder the axes of this tensor.

        Args:
            order (tuple): the new order of the axes.

        Returns:
            <TensorBase>
        '''
        pass

    @abstractmethod
    def merge_axes(self, sls, nlabel=None, signs=None, bmg=None):
        '''
        Merge multiple axes into one.

        Args:
            sls (slice): the axes to merge.
            nlabel (str/None, the new label): addition of old labels if None.
            signs (1darray): the signs(flow direction) for each merged block marker.

        Returns:
            <TensorBase>
        '''
        pass

    @abstractmethod
    def split_axis(self, axis, nlabels, dims=None):
        '''
        Split one axis into multiple.

        Args:
            axis (int/str): the axes to merge.
            dims (tuple, the new dimensions): can be None if nlabels contain block markers.
            nlabels (list): the new labels.

        Returns:
            <TensorBase>
        '''
        pass

    @abstractmethod
    def get_block(self, block):
        '''
        Query data in specific block.

        Args:
            block (tuple): the target block.

        Returns:
            ndarray, the data.
        '''
        pass


    @abstractmethod
    def set_block(self, block):
        '''
        Query data in specific block.

        Args:
            block (tuple): the target block.

        Returns:
            ndarray, the data.
        '''
        pass


    @abstractmethod
    def eliminate_zeros(self, tol=ZERO_REF):
        '''
        Remove zeros or exremely small elements.

        Args:
            tol (float): the tolerance.

        Returns:
            <TensorBase>, self.
        '''
        pass

    def chlabel(self, nlabel, axis=None):
        '''
        Change the label in place.

        Args:
            axis (int/None): the axis to change.
            nlabel (str): the new label.

        Returns:
            self
        '''
        if axis is None:
            if len(nlabel) != self.ndim:
                raise ValueError
            for i in range(self.ndim):
                self.chlabel(nlabel[i], axis=i)
            return self
        if axis < 0:
            axis = self.ndim - axis
        labels = self.labels[:]
        if hasattr(labels[axis], 'bm') and not hasattr(nlabel, 'bm'):
            labels[axis] = labels[axis].chstr(nlabel)
        else:
            labels[axis] = nlabel
        self.labels = labels
        return self

    def svd(self, cbond, cbond_str='_O_', kernel='svd', signs=None, bmg=None):
        '''
        Get the svd decomposition for dense tensor with block structure.

        Args:
            cbond (int): the bound to perform svd.
            cbond_str (str): the labes string for center bond.
            kernel ('svd'/'ldu'/'dpl_r'/'dpl_c'): the kernel of svd decomposition.
            signs (list):
            bmg (<BlockMarkerGenerator>):

        Returns:
            (U,S,V) that U*S*V = A
        '''
        kernel_type = 0  # 0->USV type, 1->QR type, 2->USU type
        if kernel == 'svd':
            def csvd(cell): return svd(
                cell, full_matrices=False, lapack_driver='gesvd')
        elif kernel == 'ldu':
            def csvd(cell): return ldu(cell)
        elif kernel == 'dpl_r':
            def csvd(cell): return dpl(cell, axis=0)
            kernel_type = 1
        elif kernel == 'dpl_c':
            def csvd(cell): return dpl(cell, axis=1)
            kernel_type = 1
        elif kernel == 'eigh':
            def csvd(cell):
                data = eigh(cell)
                return (data[1], data[0], data[1].T.conj())
            kernel_type = 2
        else:
            raise ValueError()
        from .btensor import BTensor

        # cope two exteme cases, no bm and null bm.
        if isinstance(self.labels[0], BLabel):
            if signs is None or bmg is None:
                raise ValueError
            # first, find the block structure and make it block diagonal
            M = self.merge_axes(sls=slice(cbond, self.ndim), nlabel='_X_', signs=signs[cbond:], bmg=bmg).merge_axes(
                sls=slice(0, cbond), nlabel='_Y_', signs=signs[:cbond], bmg=bmg)
            if M.labels[0].bm.qns.shape[1] == 0:
                data = csvd(M)
                U, V = data[0], data[-1]
                center_label = BLabel(cbond_str, BlockMarker(
                    qns=np.zeros([1, 0], dtype='int32'), Nr=array([0, V.shape[0]])))
                U = Tensor(U, labels=[M.labels[0], center_label])
                V = Tensor(V, labels=[center_label, M.labels[1]])
                return (U, data[1], V) if kernel_type!=1 else (U, V)
        else:
            M = self.merge_axes(sls=slice(cbond, self.ndim), nlabel='_X_').merge_axes(
                sls=slice(0, cbond), nlabel='_Y_')
            # perform svd and roll back to original non-block structure
            data = csvd(M)
            U, V = data[0], data[-1]
            U = Tensor(U.reshape(
                self.shape[:cbond] + (U.shape[-1],)), labels=self.labels[:cbond] + [cbond_str])
            V = Tensor(V.reshape(
                (V.shape[0],) + self.shape[cbond:]), labels=[cbond_str] + self.labels[cbond:])
            return (U, data[1], V) if kernel_type!=1 else (U, V)

        M, pms = M.b_reorder(return_pm=True)
        # check and prepair datas
        bm1, bm2 = M.labels[0].bm, M.labels[1].bm
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
            cell = M.get_block((c1, c2))
            data = csvd(cell)
            UL.append(data[0])
            VL.append(data[-1])
            if kernel_type != 1:
                SL.append(data[1])

        # get center BLabel and S
        nr = [vi.shape[0] for vi in VL]
        Nr = np.append([0], np.cumsum(nr))
        b0 = BLabel(cbond_str, BlockMarker(Nr=Nr, qns=common_qns_2d))
        if kernel_type!=1:
            S = np.concatenate(SL)

        # get U, V
        if isinstance(M, Tensor):
            # get correct shape of UL
            ptr = 0
            for i, lbi_1d in enumerate(qns1_1d):
                if lbi_1d != common_qns_1d[ptr]:
                    UL.insert(i, np.zeros(
                        [bm1.blocksize(i), 0], dtype=M.dtype))
                elif ptr != len(common_qns_1d) - 1:
                    ptr = ptr + 1

            # the same for VL
            ptr = 0
            for i, lbi_1d in enumerate(qns2_1d):
                if lbi_1d != common_qns_1d[ptr]:
                    VL.insert(i, np.zeros(
                        [0, bm2.blocksize(i)], dtype=M.dtype))
                elif ptr != len(common_qns_1d) - 1:
                    ptr = ptr + 1
            U, V = Tensor(block_diag(
                *UL), labels=[M.labels[0], b0]), Tensor(block_diag(*VL), labels=[b0, M.labels[1]])
        elif isinstance(M, BTensor):
            U = BTensor(dict(((b1, b2), data) for b2, (b1, data) in enumerate(
                zip(cqns1, UL))), labels=[M.labels[0], b0])
            V = BTensor(dict(((b1, b2), data) for b1, (b2, data) in enumerate(
                zip(cqns2, VL))), labels=[b0, M.labels[1]])

        # detect a shape error raised by the wrong ordering of block marker.
        if M.shape[0] != U.shape[0] or M.shape[1] != V.shape[1]:
            raise Exception('Error! 1. check block markers!')
        # U,V=U.take(np.argsort(pms[0]),axis=0).split_axis(axis=0,nlabels=self.labels[:cbond],dims=self.shape[:cbond]),V.take(np.argsort(pms[1]),axis=1).split_axis(axis=1,nlabels=self.labels[cbond:],dims=self.shape[cbond:])
        U, V = U.take(np.argsort(pms[0]), axis=0), V.take(
            np.argsort(pms[1]), axis=1)
        U, V = U.split_axis(axis=0, nlabels=self.labels[:cbond], dims=self.shape[:cbond]), V.split_axis(
            axis=1, nlabels=self.labels[cbond:], dims=self.shape[cbond:])
        return (U, S, V) if kernel_type!=1 else (U, V)

    def autoflow(self, axis, bmg, signs, check_conflicts=False):
        '''
        Determine the flow for one axis of a tensor by quantum number conservation rule.

        Args:
            axis (int): the direction with quantum number unspecified.
            bmg (<BlockMarkerGenerator>):
            signs (1d array): the flow directions of tensors.
            check_conflicts (bool, detect the conflicts in tensor): to filter out tensors without specific good quantum number.

        Returns:
            <Tensor>, the new tensor, with the remainning axis determined.
        '''
        if axis < 0:
            axis = axis + self.ndim
        # get the matrix of Quantum number
        QNS = np.zeros(
            self.shape + (bmg.qns1.shape[-1],), dtype=bmg.qns1.dtype)
        for i, (lb, sign) in enumerate(zip(self.labels, signs)):
            if i != axis:
                QNS = QNS + sign * \
                    lb.bm.inflate().qns[[slice(None)] +
                                        [np.newaxis] * (self.ndim - 1 - i)]
        QNS = bmg.trim_qns(-QNS * sign)

        # get quantum numbers using non-zero elements of tensors.
        mask = self != 0
        QNS[~mask] = 0
        raxes = tuple(list(range(axis)) + list(range(axis + 1, self.ndim)))
        qns = QNS.sum(axis=raxes, dtype=QNS.dtype)
        qns //= mask.sum(axis=raxes).astype(qns.dtype)[:, np.newaxis]

        # detect conflicts!
        if check_conflicts:
            for i in range(len(qns)):
                qni = QNS.take(i, axis=axis)[self.take(i, axis=axis) != 0]
                if any(qns[i] != qni):
                    raise ValueError()

        bmr = BlockMarker(Nr=np.arange(self.shape[axis] + 1), qns=qns)
        self.labels[axis] = BLabel(self.labels[axis], bmr)
        return self


class Tensor(np.ndarray, TensorBase):
    '''
    Tensor class subclassing ndarray, with each dimension labeled by a string(or BLabel).

    Construct:
        Tensor(shape,labels,**kwargs):
            Create a <Tensor> with random data with specified shape.
        Tensor(array,labels,**kwargs):
            Create a <Tensor> converted from array.

    Attributes:
        labels (list): the labels of axes.

        *refer numpy.ndarray for more details.*
    '''
    __array_priority__ = 0  # if it is >0, __str__ will not work.

    def __new__(subtype, param, labels, *args, **kwargs):
        if isinstance(param, tuple):
            dim = len(param)
            if dim != len(labels):
                raise ValueError(
                    'Inconsistant number of dimension(%s) and labels(%s).' % (dim, len(labels)))
            obj = np.ndarray.__new__(subtype, shape=param, *args, **kwargs)
        elif isinstance(param, subtype):
            obj = param
        else:
            dim = np.ndim(param)
            if dim != len(labels):
                raise ValueError(
                    'Inconsistant number of dimension(%s) and labels(%s).' % (dim, len(labels)))
            obj = np.asarray(param, *args, **kwargs).view(subtype)
        obj.labels = labels
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.labels = obj.labels[:] if hasattr(obj, 'labels') else None

    def __str__(self):
        return '<Tensor(%s)>\n%s' % (','.join([lb + '*' if isinstance(lb, BLabel) else lb for lb in self.labels]), super(Tensor, self).__str__())

    def __repr__(self):
        return '<Tensor(%s)>' % (','.join([lb + '*' if isinstance(lb, BLabel) else lb for lb in self.labels]),)

    def __mul__(self, target):
        if isinstance(target, Tensor):
            return tdot(self, target)
        else:
            res = super(Tensor, self).__mul__(target)
            return res

    def __rmul__(self, target):
        if isinstance(target, Tensor):
            return target.__mul__(self)
        else:
            return super(Tensor, self).__rmul__(target)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(Tensor, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.labels,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.labels = state[-1]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(Tensor, self).__setstate__(state[0:-1])

    def conj(self):
        res = Tensor(np.asarray(self).conj(), labels=self.labels[:])
        return res

    def make_copy(self, labels=None, copydata=True):
        if labels is not None:
            assert(len(labels) == len(self.shape))
        else:
            labels = self.labels[:]
        if copydata:
            res = copy.deepcopy(self)
            res.labels = labels
            return res
        else:
            res = self.view(Tensor)
            res.labels = labels
            return res

    def todense(self):
        return self.make_copy()

    def take(self, key, axis):
        if isinstance(axis, str):
            axis = self.labels.index(axis)
        if axis < 0:
            axis += self.ndim

        # regenerate the labels,
        labels = self.labels[:]
        if np.ndim(key) == 0:
            # 0d case, delete a dimension.
            lb = labels.pop(axis)
        elif np.ndim(key) == 1:
            # 1d case, shrink a dimension.
            key = np.asarray(key)
            if key.dtype == 'bool':
                key = np.where(key)[0]
            if hasattr(self.labels[axis], 'bm'):
                bm = self.labels[axis].bm
                # inflate and take the desired dimensions
                bm_infl = bm.inflate()
                qns = bm_infl.qns[key]
                bm = BlockMarker(qns=qns, Nr=np.arange(len(qns) + 1)).compact_form()
                labels[axis] = labels[axis].chbm(bm)
            else:
                # 1d case, shrink a dimension.
                return np.ndarray.take(self, key, axis)
        else:
            raise ValueError
        ts = super(Tensor, self).take(key, axis=axis)
        return Tensor(ts, labels=labels)

    def take_b(self, key, axis):
        if not hasattr(self.labels[axis], 'bm'):
            raise ValueError
        if isinstance(axis, str):
            axis = self.labels.index(axis)
        if axis < 0:
            axis += self.ndim

        # regenerate the labels, get keys
        labels = self.labels[:]
        if np.ndim(key) == 0:
            key = [key]
        elif np.ndim(key) > 1:
            raise ValueError
        if isinstance(key, np.ndarray) and key.dtype == 'bool':
            key = np.where(key)[0]
        # change block marker
        bm = self.labels[axis].bm
        qns, nr = bm.qns[key], bm.nr[key]
        nbm = BlockMarker(qns=qns, Nr=np.append([0], np.cumsum(nr)))
        labels[axis] = labels[axis].chbm(nbm)

        # take values
        ts = np.concatenate(
            [self[(slice(None),) * axis + (bm.get_slice(k),)] for k in key], axis=axis)
        return Tensor(ts, labels=labels)

    def chorder(self, order):
        assert(len(order)) == np.ndim(self)
        if isinstance(order[0], str):
            order = [self.labels.index(od) for od in order]
        t = Tensor(np.transpose(self, order), labels=[
                   self.labels[i] for i in order])
        return t

    def mul_axis(self, vec, axis):
        if isinstance(axis, str):
            axis = self.labels.index(axis)
        if axis < 0:
            axis += self.ndim
        if isinstance(axis, int):
            vec = np.asarray(vec).reshape([-1] + [1] * (self.ndim - axis - 1))
            res = vec * self
            return res
        else:
            raise TypeError('Wrong type for axis indicator: %s.' %
                            axis.__class__)

    def merge_axes(self, sls, nlabel=None, signs=None, bmg=None):
        start, stop = sls.start, sls.stop
        if stop is None:
            stop = self.ndim
        if start is None:
            start = 0
        if stop - start < 2:
            return self.make_copy()
        axes = np.mgrid[sls]
        labels = self.labels
        # get new shape
        shape = array(self.shape)
        newshape = list(shape[:start]) + \
            [np.prod(shape[sls])] + list(shape[stop:])
        ts = np.asarray(self.reshape(newshape))

        # get new labels
        if nlabel is None:
            nlabel = ''.join([labels[i] for i in axes])

        # get new block markers.
        if all([isinstance(labels[i], BLabel) for i in axes]):
            labels = self.labels
            if bmg is None:
                bm_mid = join_bms([labels[ax].bm for ax in axes], signs=signs)
            else:
                bm_mid = bmg.join_bms(
                    [labels[ax].bm for ax in axes], signs=signs)
            nlabel = BLabel(nlabel, bm_mid)
            # ts=ts.take(pm,axis=start)
        newlabels = labels[:start] + [nlabel] + labels[stop:]

        # generate the new tensor
        return Tensor(ts, labels=newlabels)

    def split_axis(self, axis, nlabels, dims=None):
        if isinstance(axis, str):
            axis = self.labels.index(axis)
        if axis < 0:
            axis += self.ndim
        if dims is None and not all(hasattr(lb, 'bm') for lb in nlabels):
            raise ValueError
        if dims is None:
            dims = [lb.bm.N for lb in nlabels]
        # get new shape
        shape = list(self.shape)
        newshape = shape[:axis] + list(dims) + shape[axis + 1:]
        # get new labels
        newlabels = self.labels[:axis] + nlabels + self.labels[axis + 1:]
        # generate the new tensor
        return Tensor(self.reshape(newshape), labels=newlabels)

    def split_axis_b(*args, **kwargs):
        return self.split_axis(*args, **kwargs)

    def get_block(self, block):
        if not isinstance(self.labels[0], BLabel):
            raise Exception('This tensor is not blocked!')
        return self[tuple([lb.bm.get_slice(b) for b, lb in zip(block, self.labels)])]

    def set_block(self, block, data):
        if not isinstance(self.labels[0], BLabel):
            raise Exception('This tensor is not blocked!')
        self[tuple([lb.bm.get_slice(b) for b, lb in zip(block, self.labels)])] = data

    def toarray(self):
        return np.asarray(self)

    def tobtensor(self, bms=None):
        '''
        Parse to <BTensor>.

        Returns:
            <BTensor>,
        '''
        data = {}
        if bms is None:
            bms = [l.bm for l in self.labels]
        # detect and extract datas.
        for blk in itertools.product(*[list(range(bm.nblock)) for bm in bms]):
            datai = self[tuple(bm.get_slice(i) for bm, i in zip(bms, blk))]
            if not np.allclose(datai, 0, atol=ZERO_REF):
                data[blk] = datai
        from .btensor import BTensor
        return BTensor(data, self.labels[:])

    def b_reorder(self, axes=None, return_pm=False):
        '''
        Reorder rows, columns to make tensor blocked.

        Args:
            axes (tuple): the target axes to be reorderd
            return_pm (bool): return the permutation series.

        Returns:
            <Tensor>.
        '''
        ts = self
        pms = []
        labels = self.labels[:]
        if axes is None:
            axes = list(range(np.ndim(self)))
        for i in axes:
            bm_new, info = labels[i].bm.sort(return_info=True)
            pm = info['pm']
            bm_new = bm_new.compact_form()
            labels[i] = labels[i].chbm(bm_new)
            if not np.allclose(pm, np.arange(len(pm))):
                ts = super(Tensor, ts).take(pm, axis=i)
            pms.append(pm)
        ts.labels = labels
        if return_pm:
            return ts, pms
        else:
            return ts

    def eliminate_zeros(self, tol=ZERO_REF):
        self[abs(self) < tol] = 0
        return self
