'''
Utilities for commute and anti-commute, and broadcastable dot.
'''

from numpy import *
from scipy import sparse as sps
from scipy.linalg import norm,svd
from scipy.sparse.linalg import LinearOperator
import cPickle as pickle
import pdb

__all__=['icgs','fast_svd','eigen_cholesky','inherit_docstring_from','quicksave','quickload']

def icgs(u,Q,M=None,colwise=True,return_norm=False):
    '''
    Iterative Classical M-orthogonal Gram-Schmidt orthogonalization.

    Parameters:
        :u: vector, the vector to be orthogonalized.
        :Q: matrix, the search space.
        :M: matrix/None, the matrix, if provided, perform M-orthogonal.
        :colwise: bool, column wise orthogonalization.
        :return_norm: bool, return the norm of u.

    Return:
        vector, orthogonalized vector u.
    '''
    assert(ndim(u)==2)
    assert(M is None or colwise)
    uH,QH=u.T.conj(),Q.T.conj()
    alpha=0.5
    itmax=3
    it=1
    Mu=M.dot(u) if M is not None else u
    r_pre=sqrt(abs(uH.dot(Mu))) if colwise else sqrt(abs(Mu.dot(uH)))
    for it in xrange(itmax):
        if colwise:
            u=u-Q.dot(QH.dot(Mu))
            Mu=M.dot(u) if M is not None else u
            r1=sqrt(abs(uH.dot(Mu)))
        else:
            u=u-u.dot(QH).dot(Q)
            r1=sqrt(abs(u.dot(uH)))
        if r1>alpha*r_pre:
            break
        r_pre=r1
    if r1<=alpha*r_pre:
        warnings.warn('loss of orthogonality @icgs.')
    return (u,r1) if return_norm else u


def fast_svd(A,d):
    '''
    Fast SVD decomposition algorithm for matrices with d-dominating singular values, the complexity is d*m^3.

    Parameters:
        :A: matrix, the input matrix.
        :d: int, the number of singular values.

    Return:
        U,S,V: tuple, A=U*diag(S)*V
    '''
    N,M=A.shape
    assert(N>0 and M>0 and d<=min(N,M))
    #get V
    x=random.random([d,N])
    #Allows a special kind of LinearOperator with function rdot.
    y=A.rdot(x) if isinstance(A,LinearOperator) else x.dot(A)
    #schmidt orthogonalization of y, using icgs to guarantee orthogonality.
    for i in xrange(d):
        yi=icgs(y[i:i+1].T,Q=y[:i].T).T
        y[i:i+1]=yi/norm(yi)

    #z=M*y.H, so that M=z*y
    z=A.dot(y.T.conj())
    U,S,V=svd(z,full_matrices=False)
    V=V.dot(y)
    return U,S,V

def eigen_cholesky(A,tol=1e-10):
    '''
    Perform decomposition A=X^H*X, discarding the rank defficient part.
    '''
    E,V=eigh(A)
    if any(E<-tol):
        raise ValueError('Negative Eigenvalue Found! %s'%E)
    kpmask=E>tol
    X=(V[:,kpmask]*sqrt(E[kpmask])).T.conj()
    return X

def inherit_docstring_from(cls):
    def docstring_inheriting_decorator(fn):
        fn.__doc__ = getattr(cls, fn.__name__).__doc__
        return fn
    return docstring_inheriting_decorator

def quicksave(filename,obj):
    '''Save an instance.'''
    f=open(filename,'wb')
    pickle.dump(obj,f,2)
    f.close()

def quickload(filename):
    '''Load an instance.'''
    f=open(filename,'rb')
    obj=pickle.load(f)
    f.close()
    return obj
