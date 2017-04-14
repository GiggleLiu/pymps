from numpy import *
from numpy.linalg import norm,svd
from copy import deepcopy
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig,SuperSpaceConfig,quicksave,quickload
from mpslib import random_mps,random_bmps
from blockmatrix import SimpleBMG
from ssf import *

def getmps():
    nsite=10
    mps=random_mps(hndim=2,nsite=10,maxN=20)
    mps<<mps.nsite
    mps=mps/mps.S.item()
    return mps

def getbmps():
    nsite=10
    bmg=SimpleBMG(spaceconfig=SpinSpaceConfig([1,2]),qstring='M')
    mps=random_bmps(bmg=bmg,nsite=nsite,maxN=10)
    return mps


def test_segment():
    '''test for getting a segment.'''
    mps=getmps()
    #first, stop=3
    #mix2=get_segment(mps,stop=3,reverse=True)
    mix2=get_segment(mps,start=3)
    mix1=get_segment(mps,stop=3)
    #s1=seg_overlap(mix1,mix1.tobra(labels=[mix1.labels[0],mix1.labels[1]+'\'']),exceptions=[0])
    s2=seg_overlap(mix2,mix2.tobra(labels=[mix2.labels[0],mix2.labels[1]+'\'']))
    s1=seg_overlap(mix1,mix1.tobra(labels=[mix1.labels[0],mix1.labels[1]+'\'']))
    pdb.set_trace()
    assert_almost_equal(s1*s2,mps.tobra()*mps)
    #second, start=3
    mix1=get_segment(mps,start=3)
    mix2=get_segment(mps,start=3,reverse=True)
    s1=seg_overlap(mix1,mix1.tobra(labels=[mix1.labels[0],mix1.labels[1]+'\'']))
    s2=seg_overlap(mix2,mix2.tobra(labels=[mix1.labels[0],mix1.labels[1]+'\'']))
    print s1
    assert_almost_equal(s1*s2,mps.tobra()*mps)
    #second, the center piece.
    mix1=get_segment(mps,start=2,stop=6,reverse=False)
    mix2=get_segment(mps,start=2,stop=6,reverse=True)
    s1=seg_overlap(mix1,mix1.tobra(labels=[mix1.labels[0],mix1.labels[1]+'\'']))
    s2=seg_overlap(mix2,mix2.tobra(labels=[mix1.labels[0],mix1.labels[1]+'\'']))
    print s1
    assert_almost_equal(s1*s2,mps.tobra()*mps)
    pdb.set_trace()

if __name__=='__main__':
    test_segment()
