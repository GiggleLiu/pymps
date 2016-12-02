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

class VMPSTest(object):
    '''
    Test function for Vidal matrix product state

    hndim:
        The number of states on each site.
    '''
    def __init__(self):
        hndim=3
        nsite=10   #number of sites
        self.vec=random.random(hndim**nsite)/sqrt(hndim**nsite/2.)  #a random state in form of 1D array.
        t0=time.time()
        self.tol=4e-2
        self.vmps=state2VMPS(self.vec,sitedim=hndim,tol=self.tol)     #parse the state into a <MPS> instance.

    def test_state(self):
        t0=time.time()
        nstate=self.vmps.state            #recover the 1D array state representation.
        t1=time.time()
        print 'State tolerence %s, Elapse -> %s'%(float(sum(abs(nstate-self.vec))),t1-t0)
        assert_allclose(nstate,self.vec,atol=self.tol)

    def test_canonical(self):
        assert_(check_canonical(self.vmps))

    def test_show(self):
        ion()
        self.vmps.show()

    def test_transform(self):
        print '\nChanging to canonical form!'
        for l in [0,2,5,10]:
            t0=time.time()
            mps=self.vmps.tocanonical(l)
            t1=time.time()
            vmps=mps.tovidal()
            t2=time.time()
            nstate=mps.state            #recover the 1D array state representation.
            nstate2=vmps.state
            print 'State tolerence %s, %s, Elapse -> %s, %s'%(sum(abs(nstate-self.vec)),sum(abs(nstate2-self.vec)),t1-t0,t2-t1)
            assert_(check_canonical(vmps))
            assert_allclose(nstate,self.vec,atol=self.tol)
            assert_allclose(nstate,nstate2,atol=self.tol)
            t1=time.time()
            print 'Get MPS: %s, Elapse -> %s'%(mps,t1-t0)

    def test_rho(self):
        vmps=self.vmps
        for i,j in [(1,2),(3,6),(0,2),(vmps.nsite-2,vmps.nsite)]:
            rho=rho_on_block(vmps,i,j)
            assert_(rho.shape==tuple([vmps.hndim]*(2*(j-i))))
            assert_(all([l[0]=='s' for l in rho.labels]))
            if i==0 or j==vmps.nsite:
                link=j-1 if i==0 else i-1
                entropyl=entropy_on_link(vmps,link)
                entropyb=entropy_on_block(vmps,i,j)
                assert_almost_equal(entropyl,entropyb)

    def test_all(self):
        self.test_rho()
        self.test_state()
        self.test_canonical()
        self.test_transform()
        self.test_show()

if __name__=='__main__':
    VMPSTest().test_all()
