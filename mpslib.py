#!/usr/bin/python
'''
Matrix Product State.
'''
from numpy import *
from matplotlib.pyplot import *
from matplotlib import patches
from matplotlib.collections import LineCollection
from scipy import sparse as sps
from scipy.linalg import svd,qr,rq,block_diag,eigvalsh,eigh,sqrtm,cholesky
from numpy.linalg import norm
import pdb,time,warnings

from tensor import Tensor,tdot,BLabel
from tensorlib import contract,random_bdmatrix
from mps import MPS,_mps_sum,MPSBase,BMPS
from vidalmps import VidalMPS
from blockmatrix import BlockMarker,trunc_bm

__all__=['state2MPS','state2VMPS','mps_sum','rho_on_link','rho_on_block',\
        'entropy_on_link','entropy_on_block',\
        'product_state','random_product_state','check_validity_mps',\
        'check_flow_mpx','random_bmps','random_mps','check_canonical','swap_forder']

def _auto_label(ts,labels,in_place=False,offset=0):
    '''
    Automatically label tensors.
    '''
    dim=len(labels)
    res=[]
    if ndim(offset)==0: offset=offset*ones(dim)
    for l,t in enumerate(ts):
        assert(ndim(t)==dim)
        lbis=['%s_%s'%(labels[i],offset[i]+l) for i in xrange(dim)]
        if in_place:
            t.labels=lbis
        else:
            t=t.make_copy(copydata=False,labels=lbis)
        res.append(t)
    return res

def state2VMPS(state,sitedim,tol=1e-8,labels=['s','a']):
    '''
    Parse a normal state into a Vidal Matrix produdct state.

    Parameters
    --------------
    state:
        The target state, 1D array.
    sitedim:
        The dimension of a single site, integer.
    tol:
        The tolerence of singular value, float.
    labels:
        (label_site,label_link), The labels for degree of freedom on site and intersite links.

    *return*:
        A <VidalMPS> instance.

    Note
    ---------------
    `svd` method is used in decomposition.
    '''
    nsite=int(round(log(len(state))/log(sitedim)))
    GL,LL=[],[]
    ri=1

    factor=norm(state)
    state=state/factor
    for i in xrange(nsite):
        state=state.reshape([sitedim*ri,-1])
        U,S,V=svd(state,full_matrices=False)
        #remove zeros from v
        kpmask=abs(S)>tol
        ri=kpmask.sum()
        S=S[kpmask]
        state=S[:,newaxis]*V[kpmask]
        U=U[:,kpmask]
        ai=U.reshape([-1,sitedim,ri])
        #ai=swapaxes(ai,0,1)
        if i==0:
            gi=ai
        else:
            gi=ai/LL[-1][:,newaxis,newaxis]
        if i==nsite-1:
            GL.append(gi*V)
        else:
            LL.append(S)
            GL.append(gi)
    LL[-1]=LL[-1]*V[0]
    return VidalMPS(GL,LL,labels=labels,factor=factor)

def state2MPS(state,sitedim,l,method='qr',tol=1e-8,labels=('s','a')):
    '''
    Parse a normal state into a Matrix produdct state.

    state:
        The target state, 1D array.
    sitedim:
        The dimension of a single site, integer.
    l:
        The division point of left and right canonical scanning, integer between 0 and number of site.
    method:
        The method to extract A,B matrices.
        * 'qr'  -> get A,B matrices by the method of QR decomposition, faster, rank revealing in a non-straight-forward way.
        * 'svd'  -> get A,B matrices by the method of SVD decomposition, slow, rank revealing.
    tol:
        The tolerence of singular value, float.
    labels:
        (label_site,label_link), The labels for degree of freedom on site and intersite links.

    *return*:
        A <MPS> instance.
    '''
    nsite=int(round(log(len(state))/log(sitedim)))
    ML=[]
    ri=1
    assert(method=='svd' or method=='qr')
    assert(l>=0 and l<=nsite)

    for i in xrange(l):
        state=state.reshape([sitedim*ri,-1])
        if method=='svd':
            U,S,V=svd(state,full_matrices=False)
            #remove zeros from v
            kpmask=abs(S)>tol
            ri=kpmask.sum()
            state=S[kpmask,newaxis]*V[kpmask]
            U=U[:,kpmask]
        else:
            U,state=qr(state,mode='economic')
            kpmask=sum(abs(state),axis=1)>tol
            ri=kpmask.sum()
            state=state[kpmask]
            U=U[:,kpmask]
        ai=U.reshape([-1,sitedim,ri])
        ML.append(ai)

    ri=1
    ML2=[]
    for i in xrange(nsite-l):
        state=state.reshape([-1,sitedim*ri])
        if method=='svd':
            U,S,V=svd(state,full_matrices=False)
            #remove zeros from v
            kpmask=abs(S)>tol
            ri=kpmask.sum()
            state=S[kpmask]*U[:,kpmask]
            V=V[kpmask,:]
        else:
            state,V=rq(state,mode='economic')
            kpmask=sum(abs(state),axis=0)>tol
            ri=kpmask.sum()
            state=state[:,kpmask]
            V=V[kpmask]
        bi=V.reshape([ri,sitedim,-1])
        ML2.append(bi)
    ML.extend(ML2[::-1])
    S=state.diagonal()
    return MPS(ML,l,S=S,labels=labels)

def rho_on_link(mps,l=None):
    '''
    Get the density matrix at the link between A|B with A,B specified by divison point l.

    Parameter:
        :mps: <MPSBase>, the matrix product state.
        :l: int, the index of division point.

    Return:
        1D array, the diagonal part of density matrix.
    '''
    if l is None:
        if isinstance(mps,MPS):
            ri=mps.S**2
        else:
            raise TypeError('Please set division point or use correct <MPS> instance.')
    else:
        if isinstance(mps,VidalMPS):
            ri=mps.LL[l]**2
        elif isinstance(mps,MPS) and l==mps.l:
            ri=mps.S**2
        else:
            raise TypeError('Please set division point or use correct <MPS> instance.')
    return ri

def entropy_on_link(mps,l=None):
    '''
    The von-Neumann entropy at the link between A|B with A,B specified by division point l.

    Parameter:
        :mps: <MPSBase>, the matrix product state.
        :l: int, the index of division point.

    Return:
        float, the von-Neumann entropy.
    '''
    ri=rho_on_link(mps,l)
    entropy=-sum((ri*log(ri)/log(2.)))
    return entropy

def rho_on_block(vmps,i,j):
    '''
    The density matrix on block.

    Parameters:
        :vmps: <VidalMPS>, MPS of Vidal form.
        :i,j: int, the border of desired region.

    Return:
        matrix, the density matrix.
    '''
    assert(j>i)
    llink_axis,rlink_axis=vmps.llink_axis,vmps.rlink_axis
    psi=(ones(1) if i==0 else vmps.LL[i-1]).reshape([-1]+[1]*(2-llink_axis))
    for k in xrange(i,j):
        rk=vmps.GL[k]*(ones(1) if k==vmps.nsite-1 else vmps.LL[k]).reshape([-1]+[1]*(2-rlink_axis))
        psi=psi*rk
    aaxes=(llink_axis,rlink_axis+j-i-1)
    psid=psi.make_copy(labels=[l if k in aaxes else l+'\'' for k,l in enumerate(psi.labels)]).conj()
    r=psi*psid
    return r

def entropy_on_block(vmps,i,j):
    '''
    The von-Neumann entropy at the block specified by i,j.

    Parameter:
        :vmps: <VidalMPS>, MPS of Vidal form.
        :i,j: int, the border of desired region.

    Return:
        float, the von-Neumann entropy.
    '''
    ri=rho_on_block(vmps,i,j)
    N=sqrt(prod(ri.shape))
    ri=eigvalsh(ri.reshape([N,N]))
    entropy=-sum((ri*log(ri)/log(2.)))
    return entropy

def mps_sum(mpses,labels=('s','a'),maxN=None):
    '''
    Summation over <MPS>es.

    Parameters:
        :mpses: list of <MPS>, instances to be added.
        :labels: list of str, the new labels for added state.

    Return:
        <MPS>, the added MPS.
    '''
    Nmps=len(mpses)
    if maxN is None:
        res=_mps_sum(mpses,labels)
        res.recanonicalize()
        return res
    else:
        res=mpses[0]
        for i in xrange(1,Nmps):
            res=_mps_sum([res,mpses[i]],labels)
            info=res.compress(maxN=maxN)
            if info==1:
                res.recanonicalize()
        return res

def product_state(config,hndim=None,bmg=None):
    '''
    Generate a product state

    Parameters:
        :config: 1Darray/2Darray, for 1D array, it is the occupied single site index, for 2D array, it's the wave functions for each site.
        :hndim: int, the site dimension(needed if config is 1D).
        :bmg: <BlockMarkerGenerator>,

    Return:
        <MPS>, right canonical.
    '''
    nsite=len(config)
    if ndim(config)==1:
        if hndim is None: raise ValueError('The hndim is needed!')
        BL=zeros([nsite,hndim],dtype='float64')
        BL[arange(nsite),config]=1
    else:
        BL=config
    BL=reshape(BL,[nsite,1,len(BL[0]),1])

    #set up block markers
    if bmg is not None:
        if ndim(config)!=1:
            raise Exception('Fail to using bms to this state.')
        res=BMPS(list(BL),0,S=ones(1),bmg=bmg)
    else:
        res=MPS(list(BL),0,S=ones(1))
    return res

def random_product_state(nsite,hndim):
    '''
    Generate a random product state.

    Parameters:
        :nsite/hndim: int, the number of site, the dimension of single site.

    Return:
        <MPS>, right canonical.
    '''
    config=random.random([nsite,hndim])
    config=config/norm(config,axis=1)[:,newaxis]
    return product_state(config)

def mps_overlap(ket,bra,start=0,end=-1,attach_S='B'):
    '''
    Get the overlap of two TensorTrains.

    Parameters:
        :ket/bra: <MPS>s,
        :start/end: int, the segment.
        :attach_S: str, 'A' or 'B'.

    Return:
        <Tensor>, left dimensions from bra and right dimensions from ket.
    '''
    if end==-1:
        end=ket.nsite
    site_axis=ket.site_axis
    rlink_axis=ket.rlink_axis
    llink_axis=ket.llink_axis
    tt1=bra.get_all(attach_S=attach_S)[start:end]
    tt2=ket.get_all(attach_S=attach_S)[start:end]
    return tt_overlap(tt1,tt2,data_format=(llink_axis,site_axis,rlink_axis),adjust_labels=True)

def random_bmps(bmg,nsite,maxN=50):
    '''
    Random <BMPS>.

    Parameters:
        :bmg: <BlockMarkerGenerator>,
        :nsite: int, the number of sites.
        :maxN: int, the maximum bond dimension.

    Return:
        <BMPS>,
    '''
    hndim=len(bmg.qns1)
    #first generate block markers.
    bmi=bmg.bm0
    bms=[bmi]
    ML=[]
    for i in xrange(nsite):
        bmi,pm=bmg.update1(bmi)
        #create a random block diagonal matrix
        ts=random_bdmatrix(bmi,dtype='complex128')
        dim=min(maxN,hndim**(nsite-i-1))
        if bmi.N>dim:
            #do random truncation!
            kpmask=zeros(bmi.N,dtype='bool')
            randarr=arange(bmi.N)
            random.shuffle(randarr)
            kpmask[randarr[:dim]]=True
            ts=ts.take(kpmask,axis=1)
        else:
            kpmask=None
        #unsort left labels, truncate right labels
        ts=ts.take(argsort(pm),axis=0)
        bmi=ts.labels[1].bm
        ts=reshape(ts,[-1,hndim,ts.shape[-1]])
        bms.append(bmi)
        ML.append(ts)
    mps=BMPS(ML=ML,l=nsite,S=ones(1),bmg=bmg)
    return mps

def random_mps(hndim=2,nsite=10,maxN=50):
    '''
    Random <MPS>.

    Parameters:
        :hndim: int, the single site Hilbert space dimension.
        :nsite: int, the number of sites.
        :maxN: int, the maximum bond dimension.

    Return:
        <MPS>,
    '''
    ML=[]
    rdim=1
    for i in xrange(nsite):
        ldim=rdim
        rdim*=hndim
        rdim=min(maxN,hndim**(nsite-i-1),rdim)
        ts=(random.random([ldim,hndim,rdim])+1j*random.random([ldim,hndim,rdim]))/sqrt(ldim*rdim)  #the factor to make it well normed.
        ML.append(ts)
    mps=MPS(ML=ML,l=nsite,S=ones(1))
    return mps

def check_validity_mps(mps):
    '''
    Check the validity of mps, mainly the bond dimension check.

    Parameters:
        <MPS>,

    Return:
        bool, True if it is a valid <MPS>.
    '''
    valid=True
    #1. the link dimension check
    nsite=mps.nsite
    llink_axis,site_axis,rlink_axis=mps.llink_axis,mps.site_axis,mps.rlink_axis
    hndim=mps.hndim
    for i in xrange(nsite-1):
        valid=valid and mps.get(i+1).shape[llink_axis]==mps.get(i).shape[rlink_axis]
        if hasattr(mps,'bmg'):
            valid=valid and mps.get(i+1).labels[llink_axis].bm==mps.get(i).labels[rlink_axis].bm

    for i in xrange(nsite):
        cell=mps.get(i)
        assert(ndim(cell)==3)
        valid=valid and cell.shape[site_axis]==hndim
        #2. the block marker check
        for i in xrange(3):
            if hasattr(cell.labels[i],'bm'):
                valid=valid and cell.shape[i]==cell.labels[i].bm.N
    return valid

def check_canonical(ket,tol=1e-8):
    '''
    Check if a MPS meets some canonical condition.

    Parameters:
        :ket: <MPSBase>,
        :tol: float, the tolerence.

    Return:
        bool, true if this MPS is canonical.
    '''
    res=[]
    hndim=ket.hndim
    site_axis=ket.site_axis

    if isinstance(ket,MPS):
        l=ket.l
        for i in xrange(l):
            mi=ket.ML[i]
            res.append(all(abs(sum([mi.take(j,axis=site_axis).T.conj().dot(mi.take(j,axis=site_axis)) for j in xrange(hndim)],axis=0)-identity(mi.shape[ket.rlink_axis]))<tol))
        for i in xrange(l,ket.nsite):
            mi=ket.ML[i]
            res.append(all(abs(sum([mi.take(j,axis=site_axis).dot(mi.take(j,axis=site_axis).T.conj()) for j in xrange(hndim)],axis=0)-identity(mi.shape[ket.llink_axis]))<tol))
        return all(res)
    elif isinstance(ket,VidalMPS):
        rl=ket.get_rho()
        #check for left canonical
        i_l=[sum([gi.take(i,axis=site_axis).T.conj().dot(ri).dot(gi.take(i,axis=site_axis)) for i in xrange(hndim)],axis=0) for ri,gi in zip([1]+rl,ket.GL)]
        diff_l=array([sum(abs(ii-identity(ii.shape[-1]))) for ii in i_l])
        #check for right canonical
        i_r=[sum([gi.take(i,axis=site_axis).dot(ri).dot(gi.take(i,axis=site_axis).T.conj()) for i in xrange(hndim)],axis=0) for ri,gi in zip(rl+[1],ket.GL)]
        diff_r=array([sum(abs(ii-identity(ii.shape[-1]))) for ii in i_r])
        cl=diff_l<tol
        cr=diff_r<tol
        return all(cl) and all(cr)
    else:
        raise TypeError()

def check_flow_mpx(mpx):
    '''
    Check the quantum number flow of mpx.

    Parameters:
        :mpx: <BMPS>/<BMPO>,

    Return:
        bool, true if the flow is quantum number conserving.
    '''
    nsite=mpx.nsite
    is_mps=isinstance(mpx,MPSBase)
    valid=True
    bmg=mpx.bmg
    for i in xrange(nsite):
        #cell is a tensor tensor(al,sup,ar)
        cell=mpx.get(i)
        if is_mps:
            cell=cell.merge_axes(slice(0,3),bmg=bmg,signs=[1,1,-1])
        else:
            cell=cell.merge_axes(slice(0,4),bmg=bmg,signs=[1,1,-1,-1])
        kpmask=(cell>1e-10)
        cbm=trunc_bm(cell.labels[0].bm,kpmask)
        valid=valid and all(cbm.qns==0)
    return valid

def swap_forder_advanced(mps,n,forder,spaceconfig,n_start=0,n_end=None,maxN=200):
    '''
    Swap ordering at n-th(ordering index) site.
    |res> = 0.5*(1+ZL+ZR-ZL*ZR)|mps>

    Parameters:
        :mps: <MPS>,
        :n_start,n,n_end: int, the start point of left segment, the division point and the end of right segment.
        :spaceconfig: <SpaceConfig>
        :ne: int, the number of electrons.
        :maxN: int, the maximum number of state if compression is needed.

    Return:
        <MPS>, the state after reordering.
    '''
    nsite=mps.nsite
    from mpo import opunit_Z
    if n_end is None: n_end=nsite
    assert(n_start>=0 and n>=n_start and n_end>=n and nsite>=n_end)
    if n_start==n or n_end==n:  #no reordering is needed!
        return mps
    #get new ordering
    dnl=n-n_start
    dnr=n_end-n
    norder=[o if (o>=n_end or o<n_start) else (o+dnr if o<n else o-dnl) for o in forder]
    #get fermionic sign operators.
    ZS=[opunit_Z(spaceconfig,siteindex=j) for j in xrange(nsite) if forder[j]>=n_start and forder[j]<n_end]
    ZL=[opunit_Z(spaceconfig,siteindex=j) for j in xrange(nsite) if forder[j]<n and forder[j]>=n_start]
    ZR=[opunit_Z(spaceconfig,siteindex=j) for j in xrange(nsite) if forder[j]>=n and forder[j]<n_end]
    ZS,ZL,ZR=prod(ZS),prod(ZL),prod(ZR)

    #add and compress
    zls=op_mul_mps(ZL,mps)
    zrs=op_mul_mps(ZR,mps)
    zlzrs=op_mul_mps(ZS,mps)
    res=0.5*mps_sum([mps,zls,zrs,-zlzrs],maxN=maxN)
    res.forder=norder
    return res

def swap_forder(mps,n,forder,ne_tot,Z1):
    '''
    Swap ordering at n-th(ordering index) site.
        |res> = 0.5*(1+ZL+ZR-ZL*ZR)|mps>
        |res> = (1-2*NL*NR)|mps>

    Parameters:
        :mps: <MPS>,
        :n: int, the division point to swap orders.
        :forder: 1d array, the fermionic ordering.
        :ne_tot: int, the number of electrons.
        :Z1: Z for single site.

    Return:
        <MPS>, the state after reordering.
    '''
    #get new ordering
    norder=[(o+mps.nsite-n if o<n else o-n) for o in forder]
    Z1=asarray(Z1)

    if ne_tot%2==0:  #nl*nr=nl
        res=mps.toket()
        for i,(fi,M) in enumerate(zip(forder,res.ML)):
            if fi<n:
                res.set(i,M*Z1[:,newaxis])
    else:  #nl*nr=0
        res=mps
    return res,norder
