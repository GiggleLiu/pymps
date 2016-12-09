from numpy import *
from numpy.linalg import norm,svd
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig,SuperSpaceConfig,quicksave,quickload
from tensor import *
from btensor import BTensor
from utils import quicksave,quickload
from tensorlib import *
from blockmatrix import SimpleBMG,BlockMarker

class TestBTensor2D(object):
    '''Test for Tensor.'''
    def __init__(self):
        #make data that sparse enough
        self.bmg=SimpleBMG(spaceconfig=SpinSpaceConfig([2,1]),qstring='M')
        mat=array([[3,3,0,0,0],
                   [0,0,1,2,2],
                   [4,4,0,0,0],
                   [4,4,0,0,0]])
        bm1=BlockMarker([0,1,2,4],[[-1],[0],[1]])
        bm2=BlockMarker([0,2,3,5],[[-1],[0],[1]])
        labels=[BLabel('a',bm1),BLabel('b',bm2)]
        self.dtensor=Tensor(mat,labels=labels)
        data={(0,0):mat[:1,:2],(1,1):mat[1:2,2:3],(1,2):mat[1:2,3:],(2,0):mat[2:,:2]}
        self.btensor=BTensor(data,labels)
        assert_(check_validity_tensor(self.btensor))
        assert_(check_validity_tensor(self.dtensor))

    def test_data_parsing(self):
        print 'test for data parsing'
        assert_allclose(self.dtensor,self.dtensor.tobtensor().todense())
        assert_allclose(self.dtensor,self.btensor.todense())
        print self.btensor

    def test_mul_axis(self):
        print 'Testing multiplying 1D array to specific axis.'
        axis=1
        arr=random.random(self.dtensor.shape[axis])
        assert_allclose(self.btensor.mul_axis(arr,axis=axis).todense(),self.dtensor.mul_axis(arr,axis=axis))

    def test_take(self):
        print 'Testing taking axes!'
        axis=0
        i=2
        axis_label=self.btensor.labels[axis]
        t1=self.dtensor.take(i,axis=axis)
        #t2=self.btensor.take(i,axis=axis)
        #assert_allclose(t1,t2.todense())
        t11=self.dtensor.take(i,axis=axis_label)
        t22=self.btensor.take(i,axis=axis_label)
        assert_allclose(t11,t22.todense())
        assert_allclose(t1,t11)

    def test_chorder(self):
        print 'Testing chorder!'
        norder=[1,0]
        labels=[self.dtensor.labels[i] for i in norder]
        t1=self.dtensor.chorder(norder)
        t2=self.btensor.chorder(norder)
        assert_allclose(t1,t2.todense())
        t11=self.dtensor.chorder(labels)
        #t22=self.btensor.chorder(labels)
        #assert_allclose(t11,t22.todense())
        assert_allclose(t11,t1)

    def test_merge_split(self):
        print 'Testing merge and split!'
        axes=slice(0,2)
        t1=self.btensor.merge_axes(axes)
        t2=self.dtensor.merge_axes(axes)
        tval=asarray(self.dtensor).reshape([prod(self.dtensor.shape[:2])]+list(self.dtensor.shape[2:]))
        assert_allclose(t2,tval)
        assert_allclose(t2,t1.todense())

    def test_all(self):
        self.test_data_parsing()
        self.test_mul_axis()
        self.test_chorder()
        self.test_take()
        self.test_merge_split()

class TestBTensor3D(object):
    '''Test for Tensor.'''
    def __init__(self):
        #make data that sparse enough
        self.bmg=SimpleBMG(spaceconfig=SpinSpaceConfig([2,1]),qstring='M')
        mat=array([[[3,3,0,0],
                    [0,0,1,2]],
                   [[4,4,0,0],
                    [4,4,0,0]]])
        bm1=BlockMarker([0,1,2],[[0],[1]])
        bm2=BlockMarker([0,1,2],[[1],[-1]])
        bm3=BlockMarker([0,2,4],[[-1],[0]])
        labels=[BLabel('a',bm1),BLabel('b',bm2),BLabel('c',bm3)]
        self.dtensor=Tensor(mat,labels=labels)
        data={(0,0,0):mat[:1,:1,:2],(0,1,1):mat[:1,1:2,2:],(1,0,0):mat[1:,:1,:2],(1,1,0):mat[1:,1:,:2]}
        self.btensor=BTensor(data,labels)
        assert_(check_validity_tensor(self.btensor))
        assert_(check_validity_tensor(self.dtensor))

    def test_data_parsing(self):
        print 'test for data parsing'
        assert_allclose(self.dtensor,self.dtensor.tobtensor().todense())
        assert_allclose(self.dtensor,self.btensor.todense())
        print self.btensor

    def test_mul_axis(self):
        print 'Testing multiplying 1D array to specific axis.'
        axis=1
        arr=random.random(self.dtensor.shape[axis])
        assert_allclose(self.btensor.mul_axis(arr,axis=axis).todense(),self.dtensor.mul_axis(arr,axis=axis))

    def test_take(self):
        print 'Testing taking axes!'
        axis=2
        i=1
        axis_label=self.btensor.labels[axis]
        t1=self.dtensor.take(i,axis=axis)
        #t2=self.btensor.take(i,axis=axis)
        #assert_allclose(t1,t2.todense())
        t11=self.dtensor.take(i,axis=axis_label)
        t22=self.btensor.take(i,axis=axis_label)
        assert_allclose(t11,t22.todense())
        assert_allclose(t1,t11)

    def test_chorder(self):
        print 'Testing chorder!'
        norder=[2,1,0]
        labels=[self.dtensor.labels[i] for i in norder]
        t1=self.dtensor.chorder(norder)
        t2=self.btensor.chorder(norder)
        assert_allclose(t1,t2.todense())
        t11=self.dtensor.chorder(labels)
        #t22=self.btensor.chorder(labels)
        #assert_allclose(t11,t22.todense())
        assert_allclose(t11,t1)

    def test_merge_split(self):
        print 'Testing merge and split!'
        axes=slice(1,3)
        t1=self.btensor.merge_axes(axes)
        t2=self.dtensor.merge_axes(axes)
        assert_allclose(t2,t1.todense())

    def test_all(self):
        self.test_data_parsing()
        self.test_mul_axis()
        self.test_chorder()
        self.test_take()
        self.test_merge_split()


def test_tensor():
    a=np.random.random((210,30,70,10))
    ts1=Tensor(a,labels=['s1','s5','a2','bb'])
    ts2=Tensor((210,10,30),['s1','a1','s5'])
    ts2[:]=random.random((210,10,30))

    print 'testing for tensor dot'
    t0=time.time()
    res1=contract(ts1,ts2)
    t1=time.time()
    res2=tdot(ts1,ts2)
    t2=time.time()
    assert_allclose(res1,res2)
    assert_(res1.labels==res2.labels)
    print 'Elapse -> Einsum %s, Tensordot %s'%(t1-t0,t2-t1)

def test_btensor():
    print 'Testing Blocked dense Tensor.'
    print 'test for generation.'
    spaceconfig=SuperSpaceConfig([2,1,1])
    bmg=SimpleBMG(spaceconfig=spaceconfig,qstring='QM')
    bms1=[bmg.random_bm(nsite=i) for i in [3,1,3]]
    bms2=[bmg.random_bm(nsite=i) for i in [1,3,4]]
    ts1=random_btensor(bms=bms1,label_strs=['B','A','K'])
    ts2=random_btensor(bms=bms2,label_strs=['A','B','C'])
    print 'test for display'
    print ts1,[ts1,ts2]

    print 'test for __mul__'
    print 'Test for multiplication.'
    res=ts1*ts2
    assert_(res.labels==['K','C'])
    print 'test for make_copy'
    ts11=ts1.make_copy(copydata=False)
    ts111=ts11.make_copy(copydata=True,labels=['a','b','c'])
    ts11.labels[0]='xxoo'
    assert_(ts1.labels[0]!='xxoo')
    ts1[0,0,0]=15
    assert_(ts11[0,0,0]==15 and ts111[0,0,0]!=15)
    print 'Test conjugate.'
    ts1111=ts1.conj()
    ts1111.labels[2]='love'
    assert_(ts1.labels[2]!='love')
    print 'test for chorder'
    tsc=ts1.chorder([2,1,0])
    assert_(all([lb1==lb2 for lb1,lb2 in zip(tsc.labels,ts1.labels[::-1])]))
    print 'test for query_block'
    bs=(2,0,7)
    blk=ts1.get_block(bs)
    assert_allclose(blk.shape,[lb.bm.blocksize(ib) for ib,lb in zip(bs,ts1.labels)])
    #test merge axes
    spaceconfig=SuperSpaceConfig([2,1,1])
    bmg=SimpleBMG(spaceconfig=spaceconfig,qstring='QM')
    tsm=ts1.merge_axes(sls=slice(1,3),bmg=bmg)
    assert_((tsm.labels[0].bm.N,tsm.labels[1].bm.N)==tsm.shape)
    assert_allclose(tsm,ts1.reshape([ts1.shape[0],-1]))

def test_svdbd():
    '''Test for function svdbd'''
    spaceconfig=SuperSpaceConfig([2,1,1])
    bmg=SimpleBMG(spaceconfig=spaceconfig,qstring='QM')
    bms=[bmg.random_bm(nsite=i) for i in [3,4]]
    ts=random_btensor(bms=bms,label_strs=['B','A'])
    #make it block diagonal
    ts[...]=0
    bm1,bm2=ts.labels[0].bm,ts.labels[1].bm
    qns1=ts.labels[0].bm.qns
    qns2=ts.labels[1].bm.qns
    for qn in qns1:
        if any(all(qns2==qn,axis=1)):
            cell=ts[tuple([bm1.get_slice(bm1.index_qn(qn).item()),bm2.get_slice(bm2.index_qn(qn).item())])]
            cell[...]=random.random(cell.shape)
    U,S,V=svdbd(ts)
    assert_allclose(ts,(U*S).dot(V))

def test_blabel():
    print 'test BLabel. chbm,chstr'
    bl=BLabel('x',BlockMarker(Nr=[0,1,3]))
    bl1=bl.chstr('y')
    bl2=bl.chbm(BlockMarker(Nr=[0,1,4]))
    assert_(bl1=='y' and bl2=='x')
    assert_(bl1.bm is bl.bm and not (bl2.bm is bl1.bm))

    print 'test pickle BLabel'
    fn='test_save_blabel.dat'
    quicksave(fn,bl)
    bl3=quickload(fn)
    assert_(bl==bl3 and bl.bm==bl3.bm)

    print 'test copy BLabel'
    import copy
    bl3=copy.copy(bl)
    assert_(bl==bl3 and bl.bm==bl3.bm)
    bl3=copy.deepcopy(bl)
    assert_(bl==bl3 and bl.bm==bl3.bm)

if __name__=='__main__':
    TestBTensor2D().test_all()
    TestBTensor3D().test_all()
    test_btensor()
    test_blabel()
    test_svdbd()
    test_tensor()
