'''
Utilities for sub-system fidelity.
'''

from numpy import *
from scipy.linalg import eigvalsh,norm,svd,svdvals,inv,eigh
import pdb,time,os

import tensor
from contraction import USVobj,G_Gong
from mps import MPSBase
from tnet import TNet,find_optcontract
from tba.hgen import inherit_docstring_from
from utils import eigen_cholesky

__all__=['ReducedMPS','get_segment','seg_overlap','SSFC2E','SSFLR','SSFE2C','SSFC2E_F']

class ReducedMPS(MPSBase):
    '''
    ReducedMPS, MPS containing super sites.

    Attributes:
        :ML: list of tensors,
        :SN: list of int, the site division points.
        :labels: list, ('site','link')
        :is_ket: is ket or not(bra).
    '''
    def __init__(self,ML,SN,labels=['s','a'],is_ket=True):
        assert(len(ML)==len(SN)-1)
        self.ML=ML
        self.SN=asarray(SN)
        self.is_ket=is_ket
        self.chlabel(labels)

    def __str__(self):
        string='<ReducedMPS,%s>\n'%(self.nsite)
        string+='\n'.join(['  M[s=%s] (%s x %s) (%s,%s,%s)'%(\
                a.shape[self.site_axis],a.shape[self.llink_axis],a.shape[self.rlink_axis],\
                a.labels[self.llink_axis],a.labels[self.site_axis],a.labels[self.rlink_axis]\
                ) for a in self.ML])
        return string

    @property
    def nsite(self):
        return self.SN[-1]

    @inherit_docstring_from(MPSBase)
    def tobra(self,labels=None):
        if labels is None:
            labels=self.labels[:]
        if not self.is_ket:
            return ReducedMPS(ML=[mi.make_copy(copydata=False) for mi in self.ML],SN=self.SN[:],labels=labels,is_ket=False)
        else:
            return ReducedMPS(ML=[mi.conj() for mi in self.ML],SN=self.SN[:],labels=labels,is_ket=False)

    @inherit_docstring_from(MPSBase)
    def toket(self,labels=None):
        if labels is None:
            labels=self.labels[:]
        if self.is_ket:
            return ReducedMPS(ML=[mi.make_copy(copydata=False) for mi in self.ML],SN=self.SN[:],labels=labels,is_ket=True)
        else:
            return ReducedMPS(ML=[mi.conj() for mi in self.ML],SN=self.SN[:],labels=labels,is_ket=True)

    def chlabel(self,nlabel):
        '''
        Change the overall labels.
        '''
        self.labels=nlabel
        for i,(l_left,l_right) in enumerate(zip(self.SN[:-1],self.SN[1:])):
            l_all=''.join(str(i) for i in xrange(l_left,l_right))
            self.ML[i]=tensor.Tensor(self.ML[i],labels=['%s_%s'%(nlabel[1],l_left),
                    '%s_%s'%(nlabel[0],l_all),
                    '%s_%s'%(nlabel[1],l_right)])

def get_segment(mps,start=None,stop=None,reverse=False):
    '''
    Get the segment of MPS from start to stop, keeping the evironment orthogonal.

    Parameters:
        :mps: <MPS>,
        :start/stop: int/None, the start and stop point of slice.
        :reverse: bool, the edges instead of bulk are returned if True.

    Return:
        <ReducedMPS>
    '''
    if start is None and stop is None:
        raise ValueError('Please set at least one of start and stop position!')
    elif start==stop:
        if reverse:
            ML,SN=mps.get_all(),arange(mps.nsite+1)
        else:
            ML,SN=[mps.S[newaxis,:,newaxis]],array([0,mps.nsite])  #create a fake site to replace deprecated sites.
    elif start is None:
        #the left block.
        if reverse:
            return get_segment(mps,start=stop,stop=None,reverse=False)
        #orthogonalize right block.
        if mps.l>stop: mps<<mps.l-stop
        ML=mps.get_all(attach_S='A')[:stop]
        ML.append(identity(ML[-1].shape[2])[:,:,newaxis])  #create a fake site to replace deprecated sites.
        SN=append(arange(mps.l+1),[mps.nsite])
    elif stop is None:
        #the right block.
        if reverse:
            return get_segment(mps,start=None,stop=start,reverse=False)
        if mps.l<start: mps>>start-mps.l
        ML=mps.get_all(attach_S='B')[start:]
        ML.insert(0,identity(ML[0].shape[0])[newaxis,:,:])  #create a fake site to replace deprecated sites.
        SN=append([0],arange(start,mps.nsite+1))
    else:
        if not reverse:
            #the center block.
            if mps.l<start or mps.l>stop:
                mps>>start-mps.l
            ML=mps.get_all(attach_S='B')[start-mps.l:stop-mps.l]
            ML.insert(0,identity(ML[0].shape[0])[newaxis,:,:])  #create a fake site to replace left sites.
            ML.append((diag(mps.S) if mps.l==stop else identity(ML[-1].shape[2]))[:,:,newaxis])  #create a fake site to replace right sites.
            SN=concatenate([[0],arange(start,stop),[mps.nsite]])
        else:
            #the edge block.
            #orthogonalize center blocks.
            #first, get the overlap matrix of center block M(al,al',ar,ar').
            M=G_Gong(mps,mps.tobra(labels=[mps.labels[0],mps.labels[1]+'\'']),slice(start,stop),attach_S='')
            #then get X that M=X^H*X
            nshape=M.shape
            dim=nshape[0]*nshape[1]
            mat=M.reshape([dim,-1])
            X=eigen_cholesky(mat)   #mat=X^H*X, cholesky is replaced by eigh here for it is rank deficient.
            #assert(allclose(X.T.conj().dot(X),mat))
            X=X.reshape([X.shape[0],nshape[2],nshape[3]])
            X=Tensor(X,labels=['X']+M.labels[2:])
            X=X.chorder([1,0,2])
            ML=mps.get_all()[:start]+[X]+mps.get_all()[stop:]
            SN=append(arange(start+1),arange(stop,mps.nsite+1))
    return ReducedMPS(ML,SN,labels=mps.labels,is_ket=mps.is_ket)

def tt_overlap(tt1,tt2):
    '''
    Get the overlap between two tensor trains.

    Parameters:
        :tt1,tt2: list, tensor trains, datas in the form (llink, data, rlink).

    Return:
        <Tensor>, the contraction.
    '''
    #decide which direction to start contraction.
    res=1
    for MH,M in zip(tt1,tt2):
        res=MH*res*M
    return res

def seg_overlap(ket,bra):
    '''
    Get the overlap of segments.

    Parameters:
        :ket/bra: <ReducedMPS>, ket and bra.
    '''
    nseg=ket.nseg
    if not (ket.labels[0]==bra.labels[1] and len(set(ket.labels[1:],bra.labels[1:]))==4): raise ValueError()
    ket_segments=[seg[:] for seg in ket.segments]
    bra_segments=[seg[:] for seg in bra.segments]
    #absorb edge Xs, it will not cause additional computaion in general.
    if nseg!=0:
        if ket.Xs[0] is not None: ket_segments[0][0]=contract(ket.Xs[0],ket_segment[0][0])
        if ket.Xs[-1] is not None: ket_segments[-1][-1]=contract(ket_segment[-1][-1],ket.Xs[-1])
        if bra.Xs[0] is not None: bra_segments[0][0]=contract(bra.Xs[0],bra_segment[0][0])
        if bra.Xs[-1] is not None: bra_segments[-1][-1]=contract(bra_segment[-1][-1],bra.Xs[-1])
    #perform overlap for each segment.
    if nseg>2:
        raise NotImplementedError()
    elif nseg==2:
        #perform overlap for left part.
        seg_ket,seg_bra=ket_segments[0],bra_segments[0]
        if len(seg_ket)>0:
            L=tt_overlap(seg_bra,seg_ket) #use the default data format.
        else:
            L=None
        seg_ket,seg_bra=ket_segments[1],bra_segments[1]
        if len(seg_bra)>0:
            R=tt_overlap(seg_bra,seg_ket) #use the default data format.
        else:
            R=None
        #taking central nodes into consideration.
        items=[it for it in [L,ket.Xs[1],R,bra.Xs[1]] if it is not None]
        res=items[0]
        for item in items[1:]:
            res=contract(res,item)
        return res
    elif nseg==1:
        seg_ket,seg_bra=ket_segments[0],bra_segments[0]
        return tt_overlap(seg_bra,seg_ket)
    else:   # zero segments
        res=contract(bra.Xs[0],ket.Xs[0])
        return res

def SSFLR(kets,direction):
    '''
    Sweep fidelity for left to right or right to left.
    '''
    bra=kets[0].tobra(labels=[kets[0].labels[0],kets[0].labels[1]+'\''])
    ket=kets[1]
    if direction=='->':
        [keti<<keti.l-1 for keti in [bra,ket]]
        step=1
        clink_axis=kets[0].llink_axis
        attach_S='A'
        edge_labels=[bra.ML[0].labels[clink_axis],ket.ML[0].labels[clink_axis]]
    else:
        step=-1
        clink_axis=kets[0].rlink_axis
        attach_S='B'
        [keti>>keti.nsite-1-keti.l for keti in [bra,ket]]
        edge_labels=[bra.BL[-1].labels[clink_axis],ket.BL[-1].labels[clink_axis]]
    Ri=tensor.Tensor(identity(1),labels=edge_labels)
    fs=[1]
    for i in xrange(ket.nsite):
        sitei=i if direction=='->' else ket.nsite-i-1
        Ri=(bra.get(sitei,attach_S=attach_S)*Ri*ket.get(sitei,attach_S=attach_S))
        S=svdvals(Ri)
        fs.append(sum(S))
        print i,sum(S)
    return fs

def SSFC2E(kets,maxN=30,usvmode=False):
    '''
    Sweep fidelity from center to edge.

    Parameters:
        :kets: len-2 list, the kets to sweep fidelity.
        :maxN: int, the maximum retained singular value for usv mode, and the maximum retained states for direct mode.
        :usvmode: bool, use usv mode if True.
    '''
    nsite=kets[0].nsite
    bra=kets[0].tobra(labels=[kets[0].labels[0],kets[0].labels[1]+'\''])
    ket=kets[1]
    dl=maxN*ones(nsite+1,dtype='int32')
    if not usvmode:
        #compress datas
        ket.compress(maxN=maxN)
        bra.compress(maxN=maxN)
    ket>>(nsite/2-ket.l,1e-8,Inf)
    bra>>(nsite/2-bra.l,1e-8,Inf)

    rlink_axis=kets[0].rlink_axis
    edge_labels_l=[bra.ML[bra.l-1].labels[rlink_axis],ket.ML[l-1].labels[rlink_axis]]
    llink_axis=kets[0].llink_axis
    bra.BL[0].labels[llink_axis]+='@'
    ket.BL[0].labels[llink_axis]+='@'
    edge_labels_r=[bra.BL[0].labels[llink_axis],ket.BL[0].labels[llink_axis]]
    if usvmode:
        Ci=USVobj(U=tensor.Tensor(diag(bra.S)[:,newaxis,:],labels=[edge_labels_l[0],'null',edge_labels_r[0]]),S=None,\
                V=tensor.Tensor(diag(ket.S)[:,newaxis,:],labels=[edge_labels_l[1],'null',edge_labels_r[1]]))
    else:
        Ci=tensor.Tensor(diag(bra.S),labels=[edge_labels_l[0],edge_labels_r[0]])*tensor.Tensor(diag(ket.S),labels=[edge_labels_l[1],edge_labels_r[1]])
    fs=[1]
    SL=[]
    for i in xrange(ket.nsite/2):
        t0=time.time()
        site_l=nsite/2-i-1
        site_r=nsite/2+i
        if usvmode:
            t0=time.time()
            Li=USVobj(U=bra.get(site_l,attach_S='B'),S=None,V=ket.get(site_l,attach_S='B'))
            Ci=Li.join(Ci)
            print Ci
            Ci=Ci.compress(min(dl[2*i+1],min(Ci.shape)))
            Ri=USVobj(U=bra.get(site_r,attach_S='A'),S=None,V=ket.get(site_r,attach_S='A'))
            Ci=Ci.join(Ri)
            print Ci
            t1=time.time()
            Ci=Ci.compress(min(dl[2*i+2],min(Ci.shape)))
            t2=time.time()
            print 'Elapse -> %s, %s'%(t1-t0,t2-t1)
            fi=sum(Ci.S)
            print Ci,Ci.S.min()
            SL.append(Ci.S)
        else:
            t0=time.time()
            Ci=bra.get(site_l,attach_S='B')*(ket.get(site_l,attach_S='B')*Ci)*bra.get(site_r,attach_S='A')*ket.get(site_r,attach_S='A')
            #Ci=tensor.contract(mpses)
            t1=time.time()
            Ci=Ci.chorder(array([0,2,1,3]))
            S=svdvals(Ci.reshape([Ci.shape[0]*Ci.shape[1],-1]))
            t2=time.time()
            print 'Elapse -> %s, %s'%(t1-t0,t2-t1)
            fi=sum(S)
            SL.append(S)
        fs.append(fi)
        t1=time.time()
        print '%s F->%s, Elapse->%s'%(i,fi,t1-t0)
    return fs

def SSFC2E_F1(kets,spaceconfig,maxN=55):
    '''
    Sweep fidelity from center to edge, the single version taking fermionic sign into consideration.

    Parameters:
        :kets: len-2 list, the kets to sweep fidelity.
        :spaceconfig: <SuperSpaceConfig>,
        :maxN: int, the maximum retained singular value for usv mode, and the maximum retained states for direct mode.
    '''
    nsite=kets[0].nsite
    #prepair kets.
    bra=kets[0].tobra(labels=[kets[0].labels[0],kets[0].labels[1]+'\''])
    ket=kets[1]
    ket>>(nsite/2-ket.l,1e-8,Inf)
    bra>>(nsite/2-bra.l,1e-8,Inf)
    l=kets[0].forder.index(0)-nsite/2 #bulk size/2.

    rlink_axis=kets[0].rlink_axis
    edge_labels_l=[bra.ML[bra.l-1].labels[rlink_axis],ket.ML[ket.l-1].labels[rlink_axis]]
    llink_axis=kets[0].llink_axis
    bra.BL[0].labels[llink_axis]+='@'
    ket.BL[0].labels[llink_axis]+='@'
    edge_labels_r=[bra.BL[0].labels[llink_axis],ket.BL[0].labels[llink_axis]]
    Ci=tensor.Tensor(diag(bra.S),labels=[edge_labels_l[0],edge_labels_r[0]])*tensor.Tensor(diag(ket.S),labels=[edge_labels_l[1],edge_labels_r[1]])
    fs=[1]
    #get the bulk overlap matrix.
    for i in xrange(l):
        t0=time.time()
        site_l=nsite/2-i-1
        site_r=nsite/2+i
        Ci=bra.get(site_l,attach_S='B')*(ket.get(site_l,attach_S='B')*Ci)
        Ci=Ci*bra.get(site_r,attach_S='A')*ket.get(site_r,attach_S='A')
        Ci=Ci.chorder(array([0,2,1,3]))
        t1=time.time()
        print 'Update %s, Elapse->%s'%(i,t1-t0)
    S=svdvals(Ci.reshape([Ci.shape[0]*Ci.shape[1],-1]))
    f=sum(S)
    print 'Get Fidlity for l = %s: %s.'%(l,f)
    return f


def SSFC2E_F(kets,spaceconfig,maxN=None,usvmode=False):
    '''
    Sweep fidelity from center to edge, the version with fermionic sign.

    Parameters:
        :kets: len-2 list, the kets to sweep fidelity.
        :maxN: int, the maximum retained singular value for usv mode, and the maximum retained states for direct mode.
        :usvmode: bool, use usv mode if True.
    '''
    nsite=kets[0].nsite
    bra=kets[0].tobra(labels=[kets[0].labels[0],kets[0].labels[1]+'\''])
    ket=kets[1]
    ordering=kets[0].forder
    assert(allclose(ordering,arange(nsite)[::-1]))

    if not usvmode and maxN is not None:
        #compress datas
        ket.compress(maxN=maxN)
        bra.compress(maxN=maxN)
    ket>>(nsite/2-ket.l,1e-8,Inf)
    bra>>(nsite/2-bra.l,1e-8,Inf)
    ket_bra=bra.toket()
    bra_ket=ket.tobra()

    #get SRLS(sign for right part) from right to left
    site_axis=ket.site_axis
    rlink_axis=ket.rlink_axis
    llink_axis=ket.llink_axis
    op=prod([opunit_Z(spaceconfig,siteindex=j) for j in xrange(nsite/2,nsite)])
    #op=prod([OpUnitI(hndim=spaceconfig.hndim,siteindex=j) for j in xrange(nsite/2,nsite)])
    SRLS=[]
    for kt,br in [(ket,bra_ket),(ket_bra,bra)]:
        if br is bra:
            kt.chlabel('site',br.labels[0]+'2')
            kt.chlabel('link',br.labels[1]+'2')
        else:
            br.chlabel('site',kt.labels[0]+'2')
            br.chlabel('link',kt.labels[1]+'2')
        SRL=[]
        res=None
        for i in xrange(nsite/2):
            sitei=nsite-i-1
            M=kt.get(sitei,attach_S='A')
            MH=br.get(sitei,attach_S='A')
            if i==0:
                MH.labels[rlink_axis]=M.labels[rlink_axis]

            opunit=op if isinstance(op,OpUnit) else op.query(sitei)[0]
            MH.labels[site_axis]=M.labels[site_axis]+'\''
            O=tensor.Tensor(opunit.get_data(),labels=[MH.labels[site_axis],M.labels[site_axis]])
            items=[MH,O,M]
            for item in items:
                res=tensor.contract(res,item) if res is not None else item
            SRL.append(res.diagonal())
            assert(sum(abs(res))-sum(abs(res.diagonal()))<1e-5)
        SRLS.append(SRL)
    SRLS=zip(SRLS)

    #calculation
    edge_labels_l=[bra.ML[bra.l-1].labels[rlink_axis],ket.ML[ket.l-1].labels[rlink_axis]]
    bra.BL[0].labels[llink_axis]+='@'
    ket.BL[0].labels[llink_axis]+='@'
    edge_labels_r=[bra.BL[0].labels[llink_axis],ket.BL[0].labels[llink_axis]]
    if usvmode:
        Ci=USVobj(U=tensor.Tensor(diag(bra.S)[:,newaxis,:],labels=[edge_labels_l[0],'null',edge_labels_r[0]]),S=None,\
                V=tensor.Tensor(diag(ket.S)[:,newaxis,:],labels=[edge_labels_l[1],'null',edge_labels_r[1]]))
    else:
        Ci=tensor.Tensor(diag(bra.S),labels=[edge_labels_l[0],edge_labels_r[0]])*tensor.Tensor(diag(ket.S),labels=[edge_labels_l[1],edge_labels_r[1]])
    fs=[1]
    SL=[]
    for i in xrange(ket.nsite/2):
        t0=time.time()
        site_l=nsite/2-i-1
        site_r=nsite/2+i
        if usvmode:
            t0=time.time()
            Li=USVobj(U=bra.get(site_l,attach_S='B'),S=None,V=ket.get(site_l,attach_S='B'))
            Ci=Li.join(Ci)
            print Ci
            Ci=Ci.compress(min(maxN,min(Ci.shape)))
            Ri=USVobj(U=bra.get(site_r,attach_S='A'),S=None,V=ket.get(site_r,attach_S='A'))
            Ci=Ci.join(Ri)
            print Ci
            t1=time.time()
            Ci=Ci.compress(min(maxN,min(Ci.shape)))
            t2=time.time()
            print 'Elapse -> %s, %s'%(t1-t0,t2-t1)
            fi=sum(Ci.S)
            print Ci,Ci.S.min()
            SL.append(Ci.S)
        else:
            #get the sign
            SK,SB=SRLS[nsite/2-i-1]
            pdb.set_trace()
            Ci=bra.get(site_l,attach_S='B')*(ket.get(site_l,attach_S='B')*Ci)*bra.get(site_r,attach_S='A')*ket.get(site_r,attach_S='A')
            pdb.set_trace()
            #Ci=tensor.contract(mpses)
            Ci=Ci.chorder(array([0,2,1,3]))
            S=svdvals(Ci.reshape([Ci.shape[0]*Ci.shape[1],-1]))
            fi=sum(S)
            SL.append(S)
        fs.append(fi)
        t1=time.time()
        print '%s F->%s, Elapse->%s'%(i,fi,t1-t0)
    return fs

def SSFE2C(kets):
    ''' 
    Sweep fidelity from edge to center.

    Parameters:
        :kets: len-2 list, the kets to sweep fidelity.
    '''
    #strategy:
        #1. get edge segment of size ()
        #2. get overlap of edge segment F.
        #3. calculate fidelity.
    ket10=kets[0]
    if hasattr(ket10,'__call__'):
        ket10=ket10(0)
    nsite=ket10.nsite
    def solve1(l):
        ket1,ket2=kets
        if hasattr(ket1,'__call__'):
            ket1,ket2=ket1(l),ket2(l)
        nsite=ket1.nsite
        t0=time.time()
        seg1=get_segment(ket1,start=nsite/2-l,end=nsite/2+l,reverse=True)
        seg2=get_segment(ket2,start=nsite/2-l,end=nsite/2+l,reverse=True)
        t1=time.time()
        R=set_overlap(seg1,seg2.tobra())
        t2=time.time()
        R=R.reshape([prod(R.shape[:3]),-1])
        S=svdvals(R)
        fi=sum(S)
        t3=time.time()
        print '%s F->%s, Elapse->%s'%(l,fi,t1-t0),t2-t1,t3-t2
        return fi
    res=mpido(solve1,inputlist=arange(nsite/2+1))
    return res

def seg_overlap(ket,bra,exceptions=[],dense=True):
    '''
    Get the overlap of segments.

    Parameters:
        :ket/bra: <ReducedMPS>, ket and bra.
        :exception: list, sites do not contract.
    '''
    if not (ket.labels[0]==bra.labels[0] and ket.labels[1]!=bra.labels[1:]): raise ValueError()
    MHL,ML=[m if i not in exceptions else m.make_copy(labels=[m.labels[0],m.labels[1]+'\'',m.labels[2]]) for i,m in enumerate(bra.ML)],ket.ML
    tnet=TNet(MHL+ML)
    flops,order=find_optcontract(tnet,nrun=4,nswap=100)
    res=tnet.contract(order)
    labels=[MHL[0].labels[0]]+[MHL[i].labels[1] for i in exceptions]+[MHL[-1].labels[-1]]+\
            [ML[0].labels[0]]+[ML[i].labels[1] for i in exceptions]+[ML[-1].labels[-1]]
    res=res.chorder([res.labels.index(lbi) for lbi in labels])  #two ends and environments
    return res
