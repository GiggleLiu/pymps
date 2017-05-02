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

from sweep import *

def test_iterator():
    start=(1,'->',2)
    stop=(3,'<-',1)
    print 'Testing iterator start = %s, stop= %s'%(start,stop)
    iterator=get_sweeper(start=start,stop=stop,nsite=4-2,iprint=2)
    order=[(1,'->',2),(1,'<-',1),(1,'<-',0),
            (2,'->',1),(2,'->',2),(2,'<-',1),(2,'<-',0),
            (3,'->',1),(3,'->',2),(3,'<-',1),
            ]
    ion()
    visualize_sweeper(iterator,nsite=3)
    for od,it in zip(order,iterator):
        assert_(od==it)

    print 'Testing 2-site iterator.'
    start=(1,'->',0)
    stop=(3,'->',0)
    order=[(1,'->',0),(2,'->',0),(3,'->',0)]
    iterator=get_sweeper(start=start,stop=stop,nsite=2-2)
    for od,it in zip(order,iterator):
        assert_(od==it)

    print 'Testing periodic case.'
    iterator=get_psweeper(start=(1,2),stop=(3,1),nsite=4,iprint=2)
    order=[(1,2),(1,3),(2,0),(2,1),(2,2),(2,3),(3,0),(3,1)]
    for od,it in zip(order,iterator):
        assert_(od==it)
    iterator=get_psweeper(start=(1,0),stop=(3,0),nsite=2,iprint=2)
    order=[(1,0),(1,1),(2,0)]
    for od,it in zip(order,iterator):
        assert_(od==it)

if __name__=='__main__':
    test_iterator()
