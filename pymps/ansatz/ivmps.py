from numpy import *
import pdb
import copy
from functools import reduce

from .tnet import TNet

__all__ = ['IVMPS', 'Link']


class Link(object):
    '''
    Link between tensors.

    Attributes:
        labels (list): labels.
        S (1d array): the Hamiltonian.
    '''

    def __init__(self, labels, S):
        self.labels = labels
        self.S = S

    def __str__(self):
        return '%s<-->%s(%s)' % (self.labels[0], self.labels[1], self.S.shape[0])


class IVMPS(TNet):
    '''
    The infinite Vidal-form Matrix Product State.
    e.g. 1d chain, GL[0]-LL[0]-GL[1]-LL[1]-GL[0]-LL[0] ...

    Attributes:
        tensors (list, Gamma Matrices): the first axis is the site_axis.
        LL (list): links.

    Read Only Attributes:
        npart (int): the number of parts(Readonly).
    '''

    def __init__(self, tensors, LL):
        super(IVMPS, self).__init__(tensors)
        self._check_input(LL)
        self.LL = LL

    def _check_input(self, LL):
        dims = self.dims
        legs = list(self.legs)
        # link labels are not duplicated
        linklbs = reduce(lambda x, y: x + y, [lk.labels for lk in LL])
        assert(len(unique(linklbs)) == len(linklbs))
        assert(all(diff([g.shape[0] for g in self.tensors]) == 0))
        for link in LL:
            li, lj = link.labels
            # uniqueness of labels
            assert(legs.count(li) == 1 and legs.count(lj) == 1)
            # same bond dimension
            iid = legs.index(li)
            jid = legs.index(lj)
            assert(dims[iid] == dims[jid])
            # connecting different tensors, can not be the first dimension
            tid_i, lid_i = self.lid2tid(iid)
            tid_j, lid_j = self.lid2tid(jid)
            assert(tid_i != tid_j and lid_i != 0 and lid_j != 0)

    def __copy__(self):
        return IVMPS([copy.copy(g) for g in self.tensors], self.LL[:])

    @property
    def nsite(self):
        '''Number of sites.'''
        return Inf

    @property
    def npart(self):
        '''The number of parts.'''
        return len(self.tensors)

    @property
    def hndim(self):
        return self.tensors[0].shape[0]

    def tobra(self):
        raise NotImplementedError()

    def toket(self):
        raise NotImplementedError()

    def attach_links(self, tensor, exception=[]):
        '''multiply links on tensor by matching labels.'''
        for lb in tensor.labels:
            for link in self.LL:
                if lb in link.labels and lb not in exception:
                    tensor = tensor.mul_axis(link.S, lb)
                    continue
        return tensor

    def detach_links(self, tensor, exception=[]):
        '''multiply links on tensor by matching labels.'''
        for lb in tensor.labels:
            for link in self.LL:
                if lb in link.labels and lb not in exception:
                    tensor = tensor.mul_axis(1. / link.S, lb)
                    continue
        return tensor
