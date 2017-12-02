import numpy as np
from scipy.linalg import eigh, norm, svd
from scipy.sparse.linalg import LinearOperator
import scipy.sparse as sps

from . import fkron

__all__ = ['tuple_prod', 'typed_randn', 'typed_unitform',
           'icgs', 'fast_svd', 'eigen_cholesky', 'kron_csr', 'kron_coo']

ZERO_REF = 1e-12


def tuple_prod(tp):
    '''
    product over a tuple of numbers.

    Args:
        tp (tuple): the target tuple to product over.

    Returns:
        number: product of tuple.
    '''
    res = 1
    for item in tp:
        res *= item
    return res


def typed_randn(dtype, shape):
    '''
    generate a normal distributed random numbers with specific data type.

    Args:
        dtype (str): data type.
        shape (tuple): shape of desired array.

    Returns:
        ndarray: random array in 'F' order.
    '''
    # fix shape with dummy index.
    shp = [si if si >= 0 else np.random.randint(1, 21) for si in shape]

    if dtype == 'complex128':
        return np.transpose(np.random.randn(
            *shape[::-1]) + 1j * np.random.randn(*shape[::-1]))
    elif dtype == 'complex64':
        return np.complex64(typed_randn('complex128', shape))
    else:
        return np.transpose(np.random.randn(
            *shape[::-1])).astype(np.dtype(dtype))


def typed_uniform(dtype, shape, low=-1., high=1.):
    '''
    generate a uniformly distributed random numbers with specific data type.

    Args:
        dtype (str): data type.
        shape (tuple): shape of desired array.

    Returns:
        ndarray: random array in 'F' order.
    '''
    # fix shape with dummy index.
    shp = [si if si >= 0 else np.random.randint(1, 21) for si in shape]

    if dtype == 'complex128':
        return np.transpose(np.random.uniform(
            low, high, shape[::-1]) + 1j * np.random.uniform(
                low, high, shape[::-1]))
    elif dtype == 'complex64':
        return np.complex64(typed_uniform('complex128', shape, low, high))
    else:
        return np.transpose(np.random.uniform(
            low, high, shape[::-1])).astype(np.dtype(dtype))


def icgs(u, Q, M=None, colwise=True, return_norm=False, max_iter=3, alpha=0.5):
    '''
    Iterative Classical M-orthogonal Gram-Schmidt orthogonalization.

    Args:
        u (1darray): the vector to be orthogonalized.
        Q (2darray): the search space.
        M (2darray|None, default=None): the matrix, if provided, perform M-orthogonal.
        colwise (bool, default=True): column wise orthogonalization.
        return_norm (bool, default=False): return the norm of u.
        max_iter (int, default=3): maximum iteration for re-orthogonalization process.
        alpha (float, default=0.5): re-orthongonalization quanlity factor.

    Return:
        1darray: orthogonalized vector u.
    '''
    assert(np.ndim(u) == 2)
    assert(M is None or colwise)
    uH, QH = u.T.conj(), Q.T.conj()
    Mu = M.dot(u) if M is not None else u
    r_pre = np.sqrt(abs(uH.dot(Mu))) if colwise else np.sqrt(abs(Mu.dot(uH)))
    for it in range(max_iter):
        if colwise:
            u = u - Q.dot(QH.dot(Mu))
            Mu = M.dot(u) if M is not None else u
            r1 = np.sqrt(abs(uH.dot(Mu)))
        else:
            u = u - u.dot(QH).dot(Q)
            r1 = np.sqrt(abs(u.dot(uH)))
        if r1 > alpha * r_pre:
            break
        r_pre = r1
    if r1 <= alpha * r_pre:
        warnings.warn('loss of orthogonality @icgs.')
    return (u, r1) if return_norm else u


def fast_svd(A, d):
    '''
    Fast SVD decomposition algorithm for matrices with d-dominating singular values, the complexity is d*m^3.

    Args:
        A (2darray): the input matrix.
        d (int): the number of singular values.

    Return:
        tuple: (U,S,V), A=U*diag(S)*V
    '''
    N, M = A.shape
    if not (N > 0 and M > 0 and d <= min(N, M)):
        raise
    # get V
    x = np.random.random([d, N])
    # Allows a special kind of LinearOperator with function rdot.
    y = A.rdot(x) if isinstance(A, LinearOperator) else x.dot(A)
    # schmidt orthogonalization of y, using icgs to guarantee orthogonality.
    for i in range(d):
        yi = icgs(y[i:i + 1].T, Q=y[:i].T).T
        y[i:i + 1] = yi / norm(yi)

    # z=M*y.H, so that M=z*y
    z = A.dot(y.T.conj())
    U, S, V = svd(z, full_matrices=False)
    V = V.dot(y)
    return U, S, V


def eigen_cholesky(A):
    '''
    Perform decomposition A=X^H*X, discarding the rank defficient part.

    Args:
        A (1darray): square matrix as input.

    Return:
        2darray: X that X.T.conj().dot(X) = A
    '''
    E, V = eigh(A)
    if any(E < -ZERO_REF):
        raise ValueError('Negative Eigenvalue Found! %s' % E)
    kpmask = E > ZERO_REF
    X = (V[:, kpmask] * np.sqrt(E[kpmask])).T.conj()
    return X


def kron_coo(A, B):
    '''
    sparse kronecker product, the version eliminate zeros.

    Parameters:
        :A,B: matrix, the two sparse matrices.

    Return:
        coo_matrix, the kronecker product of A and B, without zeros.
    '''
    A = A.asformat('coo')
    B = B.asformat('coo')
    if len(A.data) == 0 or len(B.data) == 0:
        return sps.coo_matrix((A.shape[0] * B.shape[0], A.shape[1] * B.shape[1]))
    rown, coln, datn = fkron.fkron_coo(col1=A.col, row1=A.row, dat1=A.data,
                                       col2=B.col, row2=B.row, dat2=B.data, ncol2=B.shape[1], nrow2=B.shape[0])
    mat = sps.coo_matrix((datn, (rown, coln)), shape=(
        A.shape[0] * B.shape[0], A.shape[1] * B.shape[1]))
    return mat


def kron_csr(A, B, takerows=None):
    '''
    sparse kronecker product, the csr version.

    Parameters:
        :A,B: matrix, the two sparse matrices.
        :takerows: 1darray, the row desired.

    Return:
        csr_matrix, the kronecker product of A and B.
    '''
    A = A.asformat('csr')
    B = B.asformat('csr')
    rowdim = len(takerows) if takerows is not None else A.shape[0] * B.shape[0]
    if len(A.data) == 0 or len(B.data) == 0:
        return sps.csr_matrix((rowdim, A.shape[1] * B.shape[1]))
    if takerows is None:
        indptr, indices, data = fkron.fkron_csr(
            indptr1=A.indptr, indices1=A.indices, dat1=A.data, indptr2=B.indptr, indices2=B.indices, dat2=B.data, ncol2=B.shape[1])
    else:
        # calculate non-zero elements desired
        nrow2 = B.shape[0]
        i1s = asarray(takerows) / nrow2
        i2s = takerows - nrow2 * i1s
        nnz = sum(diff(A.indptr)[i1s] * diff(B.indptr)[i2s])
        if nnz == 0:
            return sps.csr_matrix((rowdim, A.shape[1] * B.shape[1]))

        # calculate
        indptr, indices, data = fkron.fkron_csr_takerow(indptr1=A.indptr, indices1=A.indices, dat1=A.data,
                                                        indptr2=B.indptr, indices2=B.indices, dat2=B.data, ncol2=B.shape[1], takerows=takerows, nnz=nnz)
    mat = sps.csr_matrix((data, indices, indptr),
                         shape=(rowdim, A.shape[1] * B.shape[1]))
    return mat
