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
from opstring import *
from mps import BMPS
from mpolib import *
from mpslib import *
from contraction import *
from mps import MPS
from pydavidson import gs
from utils import fast_svd
from tensor import Tensor
from blockmatrix import SimpleBMG

class TestCon(object):
    '''
    Contract for addition of two <MPS> instances.
    '''
    def __init__(self):
        nsite=6   #number of sites
        hndim=2
        l=0
        vec=random.random(hndim**nsite)   #a random state in form of 1D array.
        vec2=random.random(hndim**nsite)  #a random state in form of 1D array.

        mps=state2MPS(vec,sitedim=hndim,l=l,method='svd')     #parse the state into a <MPS> instance.
        mps2=state2MPS(vec2,sitedim=hndim,l=l,method='svd')     #parse the state into a <MPS> instance.

        j1,j2=0.5,0.2
        scfg=SpinSpaceConfig([1,2])
        I=OpUnitI(hndim=hndim)
        Sz=opunit_Sz(spaceconfig=scfg)
        Sp=opunit_Sp(spaceconfig=scfg)
        Sm=opunit_Sm(spaceconfig=scfg)
        wi=zeros((4,4),dtype='O')
        wi[0,0],wi[1,0],wi[2,1],wi[3,1:]=I,Sz,I,(j1*Sz,j2*Sz,I)
        WL=[deepcopy(wi) for i in xrange(nsite)]
        WL[0]=WL[0][3:4]
        WL[-1]=WL[-1][:,:1]
        mpo=WL2MPO(WL)

        self.mps,self.mps2=mps,mps2
        self.mpo=mpo
        self.vec,self.vec2=vec,vec2
        self.spaceconfig=scfg
     
    def test_braOket(self):
        print 'Testing contraction of mpses.'
        from contraction import Contractor
        con=Contractor(self.mpo,self.mps,bra_bond_str='c')
        con.contract2l()
        S2=con.RPART[-1]*self.mps.S**2
        l0=3
        self.mps>>l0
        S0=con.evaluate()
        H=self.mpo.H
        v=self.mps.state
        S1=v.conj().dot(H.dot(v))
        self.mps>>self.mps.nsite-l0
        for i in xrange(self.mps.nsite):
            con.lupdate(i+1)
        S3=con.LPART[self.mps.nsite]*self.mps.S**2
        assert_almost_equal(S0,S1)
        assert_almost_equal(S0,S2)
        assert_almost_equal(S0,S3)

    def test_getexpect(self):
        print 'Tesing Geting expectation values of OpUnit, OpString and OpCollection'
        opu=opunit_Sx(self.spaceconfig).as_site(0)
        ops=opunit_Sx(self.spaceconfig).as_site(2)*opunit_Sx(self.spaceconfig).as_site(3)
        opc=opunit_Sy(self.spaceconfig).as_site(3)*opunit_Sy(self.spaceconfig).as_site(1)+ops
        nsite=self.mps.nsite
        for op in [opu,ops,opc]:
            exp1=get_expect(op,self.mps)
            v=self.mps.state
            exp2=v.dot(op.H(nsite).dot(v))
            assert_almost_equal(exp1,exp2)

    def test_usv(self):
        print 'Test for USV representation.'
        l=1
        mps,mps2=self.mps,self.mps2
        mps2b=mps.tobra(labels=["s","a'"])
        mpo=self.mpo
        U,S,V=mps2b.get(l),mpo.get(l),mps2.get(l)
        #S=S.take(0,axis=2).take(0,axis=2)
        S=random.random(S.shape[1])
        M=(U.mul_axis(S,axis=-2)*V).reshape([-1,V.shape[0]*V.shape[2]])
        v=random.random(M.shape[1])
        w=random.random(M.shape[0])
        usv=USVobj(U,S,V)

        t0=time.time()
        v1=usv.dot(v)
        w1=usv.rmatvec(w)
        t1=time.time()
        w0=w.dot(M)
        v0=M.dot(v)
        t2=time.time()
        #test for rdot.
        assert_allclose(v1.ravel(),v0)
        #test for dot.
        assert_allclose(w1.ravel(),w0)
        #test for toarray.
        assert_allclose(usv.toarray(),M)
        #test for __str__
        print usv
        print 'Elapse -> %s(old %s)'%(t1-t0,t2-t1)

    def test_usvs(self):
        print 'Test for XUSV representation.'
        mps,mps2=self.mps,self.mps2
        mps2b=mps.tobra(labels=["s","a'"])
        mpo=self.mpo
        usvs=[]
        M=None
        for l in [1,2]:
            U,S,V=mps2b.get(l),mpo.get(l),mps2.get(l)
            S=random.random(S.shape[0])
            M=U*S[:,newaxis]*V if M is None else U*M*(S[:,newaxis]*V)
            usv=USVobj(U,S,V)
            usvs.append(usv)
        M=M.chorder([1,0,2,3])
        print [M]
        #M=M.chorder([1,0,2,3])
        M=M.reshape([usvs[0].U.shape[0]*usvs[-1].U.shape[2],\
                usvs[0].V.shape[0]*usvs[-1].V.shape[2]])
        usv=XUSVobj(*usvs)
        v=random.random(M.shape[1])
        w=random.random(M.shape[0])

        v1=usv.dot(v)
        v0=M.dot(v)
        #w1=w.dot(usv)
        w1=usv.rmatvec(w)
        w0=w.dot(M)
        #test for rdot
        assert_allclose(w1.ravel(),w0)
        #test for dot
        assert_allclose(v1.ravel(),v0)
        #test for toarray.
        assert_allclose(usv.toarray(),M)
        usvc=usv.compress(min(M.shape))
        #test for toarray.
        assert_allclose(usvc.toarray(),M)
        #test for __str__
        print usvc
        pdb.set_trace()

    def test_expect_onsite(self):
        '''
        test for getting expectation values for on-site term.
        '''
        print 'Testing for get on site expectation.'
        mps=deepcopy(self.mps)
        mps.S=mps.S/norm(mps.S)
        opu=OpUnitI(hndim=mps.hndim,siteindex=3)
        assert_almost_equal(get_expect(opu,mps),1)
        assert_almost_equal(get_expect(opu,mps,bra=mps.tobra(labels=mps.labels)),1)

    def test_all(self):
        self.test_expect_onsite()
        #self.test_braOket()
        self.test_getexpect()
        #self.test_usv()
        #self.test_usvs()

def test_G_Gong():
    bmg=SimpleBMG(spaceconfig=SpinSpaceConfig([1,2]),qstring='QM')
    ket=random_bmps(bmg=bmg,nsite=10,maxN=50)
    bra=ket.tobra(labels=[ket.labels[0],ket.labels[1]+'_'])
    print 'Test Gong graph.'
    for sls in [slice(4,5),slice(None)]:
        t0=time.time()
        mat1=G_Gong(ket,bra,sls).todense()
        t1=time.time()
        mat2=G_Gong(ket,bra,sls,dense=True).chorder([0,2,1,3]).reshape(mat1.shape)
        t2=time.time()
        print 'Elapse %s vs %s'%(t1-t0,t2-t1)
        assert_allclose(mat1,mat2)

if __name__=='__main__':
    test_G_Gong()
    TestCon().test_all()
