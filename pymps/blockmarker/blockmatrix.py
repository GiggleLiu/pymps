#!/usr/bin/python
# Author: Leo
# Block Diagonalized Matrix with variable block size.

from numpy import *
from scipy.linalg import eigh
import scipy.sparse as sps

from .blocklib import block_diag

__all__ = ['BlockMatrix', 'tobdmatrix', 'random_bdmat']


class BlockMatrix(object):
    '''
    Block matrix class

    Attributes:
        block_size (1D array): a list of block size.
        block_size2 (1D array, a list of block size for column): it will be viewed as a square matrix if not provided.
        Nr (1D array): a list of block indexer.
        Nr2 (1D array): a list of block indexer.
        data (matrix): the data of matirx.
    '''

    def __init__(self, block_size, block_size2=None):
        self.block_size = block_size
        self.Nr = concatenate([[0], cumsum(block_size)])
        if block_size2 is None:
            self.block_size2 = block_size
            self.Nr2 = self.Nr
        else:
            self.block_size2 = block_size2
            self.Nr2 = concatenate([[0], cumsum(block_size2)])
        self.data = ndarray(shape=(self.nblock, self.nblock2), dtype='O')

    @property
    def nblock(self):
        '''get the number of blocks.'''
        return len(self.block_size)

    @property
    def nblock2(self):
        '''get the number of blocks.'''
        return len(self.block_size2)

    @property
    def ndim(self):
        '''
        get the matrix dimension.
        '''
        return self.Nr[-1]

    @property
    def ndim2(self):
        '''
        get the matrix dimension.
        '''
        return self.Nr2[-1]

    @property
    def shape(self):
        return (self.ndim, self.ndim2)

    @property
    def T(self):
        '''Get the transpose.'''
        newbm = BlockMatrix(self.block_size2, self.block_size)
        for i in range(self.nblock):
            for j in range(self.nblock2):
                cdata = self[i, j]
                if not cdata is None:
                    newbm[j, i] = cdata.T
        return newbm

    def __setitem__(self, key, data):
        '''set the block data, key is (m,n).'''
        if not (isinstance(key, tuple) and len(key) == 2):
            raise Exception(
                'Error', 'Non valid Key! It should be a tuple of size-2 but got %s' % key)
        if data.shape != (self.block_size[key[0]], self.block_size2[key[1]]):
            raise Exception('Error', 'Matrix Dimension Error!')
        self.data[key] = data

    def __getitem__(self, key):
        '''get the block data.'''
        return self.data[key]

    def __str__(self):
        return '''%s
    Blocks: %s x %s
    Dims: %s x %s
    ''' % (super(BlockMatrix, self).__str__(), self.nblock, self.nblock2, self.ndim, self.ndim2)

    def __add__(self, target):
        '''
        get conjugate.
        '''
        if isinstance(target, BlockMatrix):
            newbm = BlockMatrix(self.block_size, self.block_size2)
            for i in range(self.nblock):
                for j in range(self.nblock2):
                    cdata = self[i, j]
                    tdata = target[i, j]
                    if not cdata is None or not tdata is None:
                        if cdata is None:
                            ndata = tdata
                        elif tdata is None:
                            ndata = cdata
                        else:
                            ndata = cdata + tdata
                        newbm[i, j] = ndata
            return newbm
        else:
            return self.toarray() + target

    def __sub__(self, target):
        '''
        get conjugate.
        '''
        if isinstance(target, BlockMatrix):
            newbm = BlockMatrix(self.block_size, self.block_size2)
            for i in range(self.nblock):
                for j in range(self.nblock2):
                    cdata = self[i, j]
                    tdata = target[i, j]
                    if not cdata is None or not tdata is None:
                        if cdata is None:
                            ndata = -tdata
                        elif tdata is None:
                            ndata = cdata
                        else:
                            ndata = cdata - tdata
                        newbm[i, j] = ndata
            return newbm
        else:
            return self.toarray() - target

    def toarray(self):
        '''to array.'''
        arr = zeros(self.shape, dtype='complex128')
        for i in range(self.nblock):
            if self.block_size[i] == 0:
                continue
            for j in range(self.nblock2):
                if self.block_size2[j] == 0:
                    continue
                cdata = self.data[i, j]
                if cdata is None:
                    continue
                arr[self.get_slice(i), self.get_slice2(j)] = cdata
        return arr

    def tocoo(self):
        '''to coo_matrix.'''
        row = []
        col = []
        data = []
        for i in range(self.nblock):
            if self.block_size[i] == 0:
                continue
            for j in range(self.nblock2):
                if self.block_size2[j] == 0:
                    continue
                cdata = self.data[i, j]
                if cdata is None:
                    continue
                rind, cind = mgrid[self.get_slice(i), self.get_slice2(j)]
                row.append(rind.ravel())
                col.append(cind.ravel())
                data.append(cdata.ravel())
        if len(row) == 0:
            return sps.coo_matrix(self.shape, dtype='complex128')
        row = concatenate(row)
        col = concatenate(col)
        data = concatenate(data)
        return sps.coo_matrix((data, (row, col)), shape=self.shape)

    def tocsr(self):
        '''to csr_matrix.'''
        return self.tocoo().tocsr()

    def tocsc(self):
        '''to csc_matrix.'''
        return self.tocoo().tocsc()

    def get_slice(self, n):
        '''
        get the slicer for n-th block.
        '''
        return slice(self.Nr[n], self.Nr[n + 1])

    def get_slice2(self, n):
        '''
        get the slicer for n-th block.
        '''
        return slice(self.Nr2[n], self.Nr2[n + 1])

    def is_diag(self):
        '''check for diagonality.'''
        for i in range(self.nblock):
            for j in range(self.nblock2):
                if i != j and not self.data[i, j] is None:
                    return False
        return True

    def is_square(self):
        '''is a square matrix or not'''
        return all(self.Nr == self.Nr2)

    def eigh(self):
        '''
        block diagonalize and get the eigenvalues and eigenvectors.

        NOTE:
            this matrix must be square and diagonal in block.
        '''
        if not (self.is_diag() and self.is_square()):
            print('Not a diagonal matrix for eigh!')
            return eigh(self.toarray())
        evals = zeros(self.ndim)
        evecs = BlockMatrix(self.block_size, self.block_size)
        for i in range(self.nblock):
            if self.block_size[i] == 0:
                continue
            data = self.data[i, i]
            if data is None:
                evecs[i, i] = identity(self.block_size[i])
            else:
                evals[self.get_slice(i)], evecs[i, i] = eigh(data)
        return evals, evecs

    def dot(self, target):
        '''matrix dot product

        target:
            the target block matrix.
        '''
        # check for validity.
        if target.shape[0] == self.shape[1] and target.shape[1] == self.block_size[0]:
            return self.vdot(target)
        if not all(self.Nr2 == target.Nr):
            raise Exception(
                'Error', 'All Block matrix Dimension should match for block matrix dot production.')
        newbm = BlockMatrix(self.block_size, target.block_size2)
        for i in range(self.nblock):
            if self.block_size[i] == 0:
                continue
            for j in range(target.nblock2):
                if target.block_size2[j] == 0:
                    continue
                for k in range(self.nblock2):
                    m1 = self.data[i, k]
                    m2 = target.data[k, j]
                    if self.block_size2[k] == 0 or (m1 is None) or (m2 is None):
                        continue
                    newbm[i, j] = m1.dot(m2)
        return newbm

    def vdot(self, target):
        '''
        dot product between matrix and vector.

        target:
            target vector.
        '''
        # check for validity
        if not self.shape[1] == target.shape[0]:
            raise Exception('Error', 'Dimension mismatch!')
        res = []
        for i in range(self.nblock):
            if self.block_size[i] == 0:
                continue
            for j in range(self.nblock2):
                if self.block_size2[j] == 0:
                    continue
                m1 = self.data[i, j]
                if m1 is None:
                    continue
                m2 = target[self.get_slice2(j)]
                res.append(m1.dot(m2))
        return concatenate(res, axis=0)

    def set_diagonal(self, datalist):
        '''
        set the diagonal part of matrix.

        Args:
            datalist (list): a list of diagonal datas.
        '''
        for i in range(self.nblock):
            if self.block_size[i] != len(datalist[i]) or self.block_size2[i] != datalist[i].shape[1]:
                raise Exception('Error', 'Submatrix dimension mismatch!')
            self.data[i, i] = datalist[i]

    def set_data(self, array, tol=1e-15):
        '''
        set block data.

        Args:
            array (matrix(dtype 'O')): the data matrix.
            tol (float): data below which to view as 0.
        '''
        if array.shape != self.shape:
            raise Exception('Matrix dimension mismatch @set_data')
        if sps.issparse(array):
            is_sparse = True
            rarr = array.tocsr()
        for i in range(self.nblock):
            for j in range(self.nblock2):
                if is_sparse:
                    cdata = rarr[self.get_slice(i)].tocsc()[
                        :, self.get_slice2(j)]
                    if cdata.nnz > 0:
                        self.data[i, j] = cdata.todense()
                else:
                    cdata = array[self.get_slice(i), self.get_slice2(j)]
                    if any(abs(cdata) > tol):
                        self.data[i, j] = cdata

    def conj(self):
        '''
        get conjugate.
        '''
        newbm = BlockMatrix(self.block_size, self.block_size2)
        for i in range(self.nblock):
            for j in range(self.nblock2):
                cdata = self[i, j]
                if not cdata is None:
                    newbm[i, j] = cdata.conj()
        return newbm


def tobdmatrix(A, block_marker):
    '''
    transform block-diagonal matrix A to <BlockMatrix>.
    '''
    nblock = block_marker.nblock
    bmatrix = BlockMatrix(block_marker.nr)
    data = []
    for i in range(nblock):
        data.append(block_marker.extract_block(A, (i, i)).toarray())
    bmatrix.set_diagonal(data)
    return bmatrix


def random_bdmat(bm, dense=True):
    '''
    Generate a random Block Diagonal matrix.

    Args:
        bm (<BlockMarker>): the block marker.
        dense (bool): return dense matrix if True.

    Returns:
        Block diagonal matrix.
    '''
    bl = []
    for p in bm.nr:
        bl.append(random.random([p, p]))
    if dense:
        return block_diag(*bl)
    else:
        return sps.block_diag(bl)
