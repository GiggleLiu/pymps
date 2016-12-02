'''
Tests for MPS and MPO
'''
from numpy import *
from numpy.linalg import norm,svd
from copy import deepcopy
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
import sys,pdb,time,numbers
from os import path
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig
from mpo import *
from mpolib import *
from mpslib import *
from contraction import *
from pydavidson import gs
from utils import fast_svd
from blockmatrix import SimpleBMG

class MPSTest(object):
    '''Tests for MPS.'''
    def __init__(self):
        nsite=10
        bmg=SimpleBMG(spaceconfig=SpinSpaceConfig([2,1]),qstring='M')
        mps=random_bmps(bmg=bmg,nsite=nsite,maxN=10)
        print 'Checking the validity!'
        assert_(check_validity_mps(mps))
        print 'Checking the conservation of flow!'
        assert_(check_flow_mpx(mps))
        self.mps=mps

    def test_cano(self):
        print 'Test canonical move and inner product.'
        mps=self.mps
        nsite=mps.nsite
        pro1=mps.tobra()*mps
        mps=float(1./sqrt(abs(pro1)))*mps
        err=mps.canomove(-nsite,tol=1e-20)
        pro2=mps.tobra()*mps
        assert_almost_equal(1,pro2)
        print 'Error -> %s'%err
        err=mps.canomove(nsite,tol=1e-20)
        print 'Error -> %s'%err
        pro2=mps.tobra()*mps
        assert_almost_equal(1,pro2)
        pro3=mps.tobra()*mps
        self.mps=mps

    def test_addcomp(self):
        print 'Testing the addition of mpses.'
        mps=self.mps
        pro1=mps.tobra()*mps
        mps2=mps+mps
        assert_(check_validity_mps(mps2))
        assert_(check_flow_mpx(mps2))
        print 'Testing for compression!'
        err=mps2.compress(tol=1e-8,maxN=50)
        print 'Error -> %s'%err
        pro2=mps2.tobra()*mps2
        assert_almost_equal(pro2,4*pro1,decimal=2)

    def test_all(self):
        self.test_addcomp()
        self.test_cano()

if __name__=='__main__':
    MPSTest().test_all()
