from numpy import *
from numpy.linalg import norm,svd
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from tensor import *
from tensorlib import *
from tnet import *
from tnetlib import *
from tba.lattice import Structure
from plotlib import show_tnet,show_contract_route

def get_tnet():
    '''Generate a random tensor network.'''
    tensors=[Tensor(random.random([10,20,3]),labels=['a','b','c']),
            Tensor(random.random([20,3,4]),labels=['b','c','d']),
            Tensor(random.random([4,5,6,7]),labels=['d','e','f','g']),
            Tensor(random.random([6,10]),['f','a']),
            Tensor(random.random([5]),['e']),
            Tensor(random.random([8,9,7]),labels=['xx','yy','g'])] #disconnected tensor
    st=Structure([[0,0],[1,0.],[2,1],[1,2],[0,2],[-1,1]])  #'honeycomb'
    tnet=TNet(tensors)
    return tnet,st


def test_complexity():
    print 'Test complexity calculation.'
    tnet,st=get_tnet()
    cap=TNCSAP(tnet)
    order=[random.randint(N*(N-1)/2) for N in xrange(tnet.ntensor,1,-1)]  #upto N=ntensor,...,2
    c1=tnet.complexity(cap._decode(order))
    for x in xrange(20):
        t0=time.time()
        proposal=cap.propose(order,c1)
        order=cap.accept(proposal,order)
        t1=time.time()
        c2=tnet.complexity(cap._decode(order))
        t2=time.time()
        assert_(c1+proposal[1]==c2)
        c1=c2
        print 'Elapse %s vs %s'%(t1-t0,t2-t1)

def test_findcontract():
    print 'Test finding the optimal contraction route.'
    tnet,st=get_tnet()
    flops,order=find_optcontract(tnet)
    assert_(flops==4019 and flops==tnet.complexity(order))
    print 'Test Show construct Route.'
    ion()
    show_tnet(tnet,st.sites)
    show_contract_route(tnet,st.sites,order)
    pause(1)
    print 'Test contraction.'
    sap=TNCSAP(tnet)
    rand_order=sap._decode(sap.get_random_state())
    t0=time.time()
    res1=tnet.contract(order).ravel()
    t1=time.time()
    res2=tnet.contract(rand_order).ravel()
    t2=time.time()
    assert_allclose(res1,res2)
    print 'Elapse %s(%s flops) vs %s(%s flops)'%(t1-t0,flops,t2-t1,tnet.complexity(rand_order))
    assert_(t2-t1>0.8*(t1-t0))
    pdb.set_trace()

def test_parse():
    print 'Test lid tid conversion.'
    tnet,st=get_tnet()
    for i in xrange(20):
        lid=random.randint(tnet.nleg)
        assert_(lid==tnet.tid2lid(tnet.lid2tid(lid)))
    connection=tnet.get_connection()
    for i in xrange(connection.nlink):
        assert_(i==connection.whichlink(connection.whichlegs(i)[0]))
        assert_(i==connection.whichlink(connection.whichlegs(i)[1]))


if __name__=='__main__':
    test_parse()
    test_complexity()
    test_findcontract()
