from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
import copy,pdb,sys
sys.path.insert(0,'../')

from tba.hgen.spaceconfig import *
from tba.hgen import Bilinear,Qlinear,Xlinear,op_c,op_cdag
from mpo import *
from mpolib import *
from mpslib import check_flow_mpx,random_mps
from copy import deepcopy
from blockmatrix import SimpleBMG
from plotlib import show_mpo

random.seed(10)

class TestMPO():
    def __init__(self):
        spaceconfig=SpinSpaceConfig([1,2])
        #generator WL
        hndim=2
        nsite=8
        j1,j2=0.5,0.2
        ion()
        I=OpUnitI(hndim=hndim)
        Sz=opunit_Sz(spaceconfig=spaceconfig)
        Sp=opunit_Sp(spaceconfig=spaceconfig)
        Sm=opunit_Sm(spaceconfig=spaceconfig)
        wi=zeros((4,4),dtype='O')
        wi[0,0],wi[1,0],wi[2,1],wi[3,1:]=I,Sz,I,(j1*Sz,j2*Sz,I)
        WL=[deepcopy(wi) for i in xrange(nsite)]
        WL[0]=WL[0][3:4]
        WL[-1]=WL[-1][:,:1]
        self.opc=WL2OPC(WL)
        self.mpo=WL2MPO(WL)

    def test_construction(self):
        H0=self.mpo.H
        opc=self.opc
        mpo1=opc.toMPO(method='direct')
        H1=mpo1.H
        mpo2=opc.toMPO(method='addition')
        H2=mpo2.H
        assert_allclose(H0,H1,atol=1e-8)
        assert_allclose(H0,H2,atol=1e-8)

    def test_product(self):
        '''Test for the product between mpo and mps.'''
        mps=random_mps(hndim=2,nsite=self.mpo.nsite)
        pdb.set_trace()

    def test_insert(self):
        print 'Test insert and remove.'
        import copy
        H0=self.mpo.H
        seg=self.mpo.OL[3:5]
        mpo1=copy.copy(self.mpo)
        mpo1.remove(3,5)
        assert_(check_validity_mpo(mpo1))
        mpo1.insert(3,seg)
        assert_allclose(H0,mpo1.H)
        assert_(check_validity_mpo(mpo1))

    def test_show(self):
        '''
        Test for MPO-MPC convertion.
        '''
        mpo=self.mpo
        print 'Testing for displaying!'
        show_mpo(mpo)

    def test_Hcompress(self):
        mpo=self.mpo
        H1=mpo.H
        mpo.compress()
        H2=mpo.H
        assert_allclose(H1,H2,atol=1e-8)

    def test_hermicity(self):
        print 'Checking Hermicity!'
        H=self.mpo.H
        assert_allclose(H,H.T.conj(),atol=1e-8)

    def test_all(self):
        self.test_insert()
        self.test_construction()
        self.test_Hcompress()
        self.test_hermicity()
        self.test_show()

class TestBMPO():
    def __init__(self):
        spaceconfig=SpinSpaceConfig([1,2])
        bmg=SimpleBMG(spaceconfig=spaceconfig,qstring='M')
        print 'Testing for random <BMPO>'
        self.mpo=random_bmpo(nsite=8,bmg=bmg)
        assert_(check_validity_mpo(self.mpo))
        assert_(check_flow_mpx(self.mpo))

    def test_Hcompress(self):
        print 'Checking compressing! using svd'
        mpo=deepcopy(self.mpo)
        nnz0=mpo.nnz
        H1=mpo.H
        mpo.compress(kernel='svd')
        H3=mpo.H
        print 'Rate = %s'%(1.*mpo.nnz/nnz0)
        assert_allclose(H1,H3,atol=1e-8)
        print 'Checking compressing! using dpl'
        mpo=deepcopy(self.mpo)
        mpo.compress(kernel='dpl')
        H2=mpo.H
        print 'Rate = %s'%(1.*mpo.nnz/nnz0)
        assert_allclose(H1,H2,atol=1e-8)
        print 'Checking compressing! using ldu'
        mpo=deepcopy(self.mpo)
        mpo.compress(kernel='ldu')
        H2=mpo.H
        print 'Rate = %s'%(1.*mpo.nnz/nnz0)
        assert_allclose(H1,H2,atol=1e-8)

    def test_insert(self):
        print 'Test insert and remove.'
        import copy
        H0=self.mpo.H
        seg=self.mpo.OL[3:5]
        mpo1=deepcopy(self.mpo)
        mpo1.remove(3,5)
        assert_(check_validity_mpo(mpo1))
        mpo1.insert(3,seg)
        assert_allclose(H0,mpo1.H)
        assert_(check_validity_mpo(mpo1))

    def test_addsum(self):
        mpo=self.mpo
        H1=mpo.H
        print 'Testing for addition.'
        mpo2=(mpo+mpo)
        mpo2.compress()
        H2=mpo2.H
        assert_allclose(2*H1,H2,atol=1e-8)
        print 'Testing for summation.'
        mpo3=mpo_sum([mpo,mpo,mpo])
        mpo3.compress()
        H3=mpo3.H
        assert_allclose(3*H1,H3,atol=1e-8)

    def test_mpoket(self):
        pass

    def test_all(self):
        self.test_insert()
        self.test_Hcompress()
        self.test_addsum()

if __name__=='__main__':
    TestMPO().test_all()
    TestBMPO().test_all()
