#!/usr/bin/python
'''
Tests for MPS and MPO
'''
from numpy import *
from numpy.linalg import norm,svd
from copy import deepcopy
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig
from mpo import *
from mpolib import *
from mpslib import *
from contraction import *
from mps import MPS
from contraction import Contractor
from pydavidson import gs
from utils import fast_svd
from blockmatrix import SimpleBMG

def test_fastsvd():
    print 'Testing for fast svd decomposition!'
    N,M=1000,800
    d=5
    sl=random.random(d)
    #construct random matrix.
    U=random.random([N,d])
    V=random.random([d,M])
    for i in xrange(d):
        ui=gs(U[:,i:i+1],Q=U[:,:i])
        vi=gs(V[i:i+1].T,Q=V[:i].T).T
        U[:,i:i+1]=ui/norm(ui)
        V[i:i+1]=vi/norm(vi)
    U=U/norm(U,axis=0)
    V=V/norm(V,axis=1)[:,newaxis]
    A=U.dot(sl[:,newaxis]*V)
    t0=time.time()
    Uf,Sf,Vf=fast_svd(A,d)
    t1=time.time()
    S0=svd(A,full_matrices=False)[1][:d]
    t2=time.time()
    assert_allclose(sort(sl)[::-1],Sf,atol=1e-6)
    assert_allclose(A,(Uf*Sf).dot(Vf),atol=1e-6)
    print 'Elapse -> %s(old->%s) tol->%s'%(t1-t0,t2-t1,norm(Sf-S0))

if __name__=='__main__':
    test_fastsvd()
