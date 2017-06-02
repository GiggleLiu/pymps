'''
Library for MPOs
'''

import numpy as np
from mpo import BMPO,MPO
from mpo import _mpo_sum
from tensorlib import random_bdmatrix

__all__=['random_bmpo','random_mpo','check_validity_mpo','mpo_sum']

def random_bmpo(bmg,nsite=10,maxN=6):
    '''
    Random <BMPS>.

    Parameters:
        :nsite: int, number of sites.
        :bmg: <BlockMarkerGenerator>,
        :maxN: int, the maximum bond dimension.

    Return:
        <BMPO>,
    '''
    hndim=len(bmg.qns1)
    #first generate block markers.
    bmi=bmg.bm0
    bm1=bmg.bm1_
    bms=[bmi]
    OL=[]
    for i in xrange(nsite):
        bmi=bmg.join_bms([bmi,bm1,bm1],signs=[1,1,-1])
        bmi,info=bmi.sort(return_info=True); pm=info['pm']
        bmi=bmi.compact_form()
        #create a random block diagonal matrix
        ts=random_bdmatrix(bmi)
        dim=min(maxN,(hndim**2)**(nsite-i-1))
        if bmi.N>dim:
            #do random truncation!
            kpmask=np.zeros(bmi.N,dtype='bool')
            randarr=np.arange(bmi.N)
            np.random.shuffle(randarr)
            kpmask[randarr[:dim]]=True
            ts=ts.take(kpmask,axis=1)
        else:
            kpmask=None
        #unsort left labels, truncate right labels
        ts=ts.take(np.argsort(pm),axis=0)
        bmi=ts.labels[1].bm
        ts=np.reshape(ts,[-1,hndim,hndim,ts.shape[-1]])
        bms.append(bmi)
        OL.append(ts)
    mpo=BMPO(OL=OL,bmg=bmg)
    return mpo

def random_mpo(hndim=2,nsite=10,maxN=6,hermitian=True):
    '''
    Random <MPO>.

    Parameters:
        :hndim: int, the single site hilbert space dimension.
        :nsite: int, number of sites.
        :maxN: int, the maximum bond dimension.
        :hermitian: bool, get a hermitian MPO if True.

    Return:
        <MPO>,
    '''
    OL=[]
    rdim=1
    for i in xrange(nsite):
        ldim=rdim
        rdim*=hndim**2
        rdim=min(maxN,(hndim**2)**(nsite-i-1),rdim)
        ts=(random.random([ldim,hndim,hndim,rdim])+0j)/sqrt(ldim*rdim*hndim**2)
        if hermitian:
            ts=ts+transpose(ts,(0,2,1,3)).conj()  #make it hermitian
        OL.append(ts)
    mpo=MPO(OL=OL)
    return mpo

def check_validity_mpo(mpo):
    '''
    check the validity of mpo.

    Parameters:
        :mpo: <MPO>,

    Return:
        bool,
    '''
    valid=True
    #1. the link dimension check
    nsite=mpo.nsite
    llink_axis,s1_axis,s2_axis,rlink_axis=0,1,2,3
    hndim=mpo.hndim
    for i in xrange(nsite-1):
        valid=valid and mpo.get(i+1).shape[llink_axis]==mpo.get(i).shape[rlink_axis]
        if hasattr(mpo,'bmg'):
            valid=valid and mpo.get(i+1).labels[llink_axis].bm==mpo.get(i).labels[rlink_axis].bm

    for i in xrange(nsite):
        cell=mpo.get(i)
        assert(np.ndim(cell)==4)
        valid=valid and cell.shape[s1_axis]==hndim
        valid=valid and cell.shape[s2_axis]==hndim
        #2. the block marker check
        for i in xrange(3):
            if hasattr(cell.labels[i],'bm'):
                valid=valid and cell.shape[i]==cell.labels[i].bm.N
    return valid

def mpo_sum(mpos,labels=['m','s','b']):
    '''
    Parameters:
        :mpos: list, list of <MPO>s.
        :labels: list, list of string as new labels(site-up,site-down,link).

    Return:
        <MPO>,
    '''
    return _mpo_sum(mpos,labels=labels)

