#!/usr/bin/python
'''
Tests for MPS and MPO
'''
from numpy import *
from numpy.linalg import norm, svd
from copy import deepcopy
from numpy.testing import dec, assert_, assert_raises, assert_almost_equal, assert_allclose
from matplotlib.pyplot import *
import pdb
import time

from ..basic import ldu, dpl
from ...toolbox.linear import fast_svd, icgs


def test_fastsvd():
    print('Testing for fast svd decomposition!')
    N, M = 1000, 800
    d = 5
    sl = random.random(d)
    # construct random matrix.
    U = random.random([N, d])
    V = random.random([d, M])
    for i in range(d):
        ui = icgs(U[:, i:i + 1], Q=U[:, :i])
        vi = icgs(V[i:i + 1].T, Q=V[:i].T).T
        U[:, i:i + 1] = ui / norm(ui)
        V[i:i + 1] = vi / norm(vi)
    U = U / norm(U, axis=0)
    V = V / norm(V, axis=1)[:, newaxis]
    A = U.dot(sl[:, newaxis] * V)
    t0 = time.time()
    Uf, Sf, Vf = fast_svd(A, d)
    t1 = time.time()
    S0 = svd(A, full_matrices=False)[1][:d]
    t2 = time.time()
    assert_allclose(sort(sl)[::-1], Sf, atol=1e-6)
    assert_allclose(A, (Uf * Sf).dot(Vf), atol=1e-6)
    print('Elapse -> %s(old->%s) tol->%s' % (t1 - t0, t2 - t1, norm(Sf - S0)))


def test_ldu():
    print('Test LDU')
    A = random.random([10, 10])
    L, D, U = ldu(A)
    assert_allclose((L * D).dot(U), A, atol=1e-8)
    assert_(sum(abs(L > 1e-8)) <= 55)
    assert_(sum(abs(U > 1e-8)) <= 55)


def test_dpl():
    print('Test DPL, col wise')
    A = array([[1, 2, 3], [4, 5, 6], [0, 0, 0], [2, 4, 6]])
    M, T = dpl(A.T, 1)
    assert_(M.shape[1] == 2)
    assert_allclose(M.dot(T), A.T)
    print('Test DPL, row wise')
    T, M = dpl(A, 0)
    assert_(M.shape[0] == 2)
    assert_allclose(T.dot(M), A)


if __name__ == '__main__':
    test_dpl()
    test_ldu()
    test_fastsvd()
