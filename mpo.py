'''
Matrix Product State.
'''

from numpy import *
from scipy.linalg import svd,qr,rq,norm
from scipy import sparse as sps
from abc import ABCMeta, abstractmethod
import copy
import pdb,time,warnings,numbers

from tensor import Tensor
from tba.hgen.op import _format_factor
from blockmatrix import block_diag
from mps import _autoset_bms,_auto_label,_replace_cells
from flib import fmerge_mpo
from sweep import get_sweeper
from opstring import OpUnitI,OpUnit

__all__=['MPO','MPOConstructor','BMPO','WL2MPO','WL2OPC','OPC2MPO']

def _mpo_sum(mpos,labels=('m','s','b')):
    '''
    Summation over <MPO>es.

    Parameters:
        :mpos: list of <MPO>, instances to be added.
        :labels: list of str, the new labels for added state.

    Return:
        <MPO>, the added MPO.
    '''
    if len(mpos)==1:
        return mpos[0]
    elif len(mpos)==0:
        raise ValueError('At least 1 <MPS> instances are required.')
    mpo0=mpos[0]
    MLs=[mpo.OL for mpo in mpos]
    hndim=mpo0.hndim
    nsite=mpo0.nsite

    #get datas
    ML,bms=[],[]
    for i in xrange(nsite):
        ai=transpose([[block_diag(*[mi[i][:,j1,j2,:] for mi in MLs]) for j2 in xrange(hndim)] for j1 in xrange(hndim)],axes=[2,0,1,3])
        ML.append(ai)
    #fix the ends
    ML[0]=ML[0].sum(axis=0)[newaxis,...]
    ML[-1]=ML[-1].sum(axis=3)[...,newaxis]

    #cope with block markers
    if all([hasattr(mpo,'bmg') for mpo in mpos]):
        mpo=mpos[0].__class__(OL=ML,labels=labels,bmg=mpo0.bmg)
    else:
        mpo=mpos[0].__class__(OL=ML,labels=labels)
    return mpo

class MPOConstructor(object):
    '''
    Constructor class for MPO.

    Attributes:
        :WL: list, the operator string.
        :link: list, storing the link information.
        :hndim: integer, the Hilbert space dimension of a single site.
    '''
    INDEX_CEIL='n'
    def __init__(self,nsite,hndim):
        self.WL=[{} for i in xrange(nsite)]
        self.links=[set([0]) for i in xrange(nsite+1)]
        self.hndim=hndim

    def asign(self,i):
        '''
        Asign an new index to connection (i,i+1).

        i:
            The index.
        '''
        li=self.links[i]
        limax=max(li)
        if len(li)==limax+1:
            li.add(limax+1)
            return limax+1
        else:
            for i in xrange(limax):
                if i not in li:
                    li.add(i)
                    return li

    def read1(self,op):
        '''
        Read 1 OpString/OpUnit.
        '''
        opunits=op.opunits if hasattr(op,'opunits') else [op]
        sites=list(op.siteindices)
        minsite=sites[0]
        maxsite=sites[-1]
        pre=0
        for i in xrange(minsite,maxsite+1):
            if not i in sites:
                op=OpUnitI(hndim=self.hndim,siteindex=i)
            else:
                op=opunits[sites.index(i)]
            if i==maxsite:
                nex=self.INDEX_CEIL
            else:
                nex=self.asign(i)
            self.WL[i][pre,nex]=op
            pre=nex

    def compile(self):
        '''
        Compile data to MPO.
        '''
        WL=[]
        for sitei,wi in enumerate(self.WL):
            maxi=max(self.links[sitei-1])
            maxj=max(self.links[sitei])
            b=zeros((maxi+2,maxj+2),dtype='O')
            for i,j in wi:
                op=wi[i,j]
                if i=='n': i=maxi+1
                if j=='n': j=maxj+1
                b[i,j]=op
                b[0,0]=OpUnitI(hndim=self.hndim,siteindex=sitei)
                b[maxi+1,maxj+1]=OpUnitI(hndim=self.hndim,siteindex=sitei)
            WL.append(b)
        WL[0]=WL[0][:1]
        WL[-1]=WL[-1][:,-1:]
        return WL2MPO(WL)

class MPOBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get(self,i):
        '''
        Get the operator tensor at specific site.

        Parameters:
            :i: int, the site index.

        Return:
            4-leg <Tensor>,
        '''
        pass

    @abstractmethod
    def eliminate_zeros(self):
        '''Eliminate zero elements.'''
        pass

    @abstractmethod
    def get_all(self):
        '''
        Get all tensors in a train.
        '''
        pass

    @abstractmethod
    def check_link(self):
        '''
        The bond dimension for l-th link.

        Parameters:
            :l: int, the bond index.

        Return:
            int, the bond dimension.
        '''
        pass

    @abstractmethod
    def chlabel(self):
        '''
        Change the label of specific axis.
        
        Parameters:
            :labels: list, the new labels.
                * 'site1' -> The on-site freedom, the first dimension.
                * 'site2' -> The on-site freedom, the second dimension.
                * 'link' -> The label of the links.
        '''
        pass

    @abstractmethod
    def compress(self,niter=2,tol=1e-8,maxN=Inf,kernel='svd'):
        '''
        Move l-index by one with specific direction.
        
        Parameters:
            :niter: int, number of iteractions.
            :tol: float, the tolerence for compression.
            :maxN: int, the maximum dimension.
            :kernel: 'svd'/'ldu'/'dpl', the compressing kernel.

        Return:
            float, approximate truncation error.
        '''
        pass

class MPO(MPOBase):
    '''
    Matrix product operator.

    Attributes:
        :labels: len-3 list, the labels for legs, in the order site-site-link.
        :OL: list, the Tensor form of MPO datas.
        :hndim: integer, the Hilbert space dimension of single site.
        :nsite: integer, the number of sites.
    '''
    def __init__(self,OL,labels=["m","s",'b']):
        self.labels=labels
        self.OL=[]
        for isite,oi in enumerate(OL):
            tlabel=['%s_%s'%(labels[2],isite),'%s_%s'%(labels[0],isite),'%s_%s'%(labels[1],isite),'%s_%s'%(labels[2],isite+1)]
            self.OL.append(Tensor(oi,labels=tlabel))

    def __str__(self):
        llink_axis,s1_axis,s2_axis,rlink_axis=0,1,2,3
        string='<MPO,%s>\n'%(self.nsite)
        string+='\n'.join(['  O[s=%s;%s] (%s x %s) (%s,%s,%s,%s)'%(\
                a.shape[s1_axis],a.shape[s2_axis],a.shape[llink_axis],a.shape[rlink_axis],\
                a.labels[llink_axis],a.labels[s1_axis],a.labels[s2_axis],a.labels[rlink_axis]\
                ) for a in self.OL])
        return string

    def __add__(self,target):
        if isinstance(target,MPO):
            return _mpo_sum([self,target])
        else:
            raise TypeError()

    def __radd__(self,target):
        if isinstance(target,MPO):
            return _mpo_sum([target,self])
        else:
            raise TypeError()

    def __copy__(self):
        return MPO(self.OL[:],self.labels[:])

    @property
    def hndim(self):
        '''The number of state in a single site.'''
        return self.OL[0].shape[1]

    @property
    def nsite(self):
        '''Number of sites.'''
        return len(self.OL)

    @property
    def H(self):
        '''Get the Hamiltonian.'''
        MAXDIM=100000
        dim=self.hndim**self.nsite
        if dim>MAXDIM:
            raise Exception('dimension of Hamiltonian too large!')
        H=reduce(lambda x,y:x*y,self.OL)
        H=H.chorder(range(1,self.nsite*2+2,2)+range(0,self.nsite*2+2,2))
        return asarray(H.reshape([dim,dim]))

    @property
    def nnz(self):
        return sum([sum(o!=0) for o in self.OL])

    def use_bm(self,bmg):
        '''
        Use <BlockMarker> to indicate block structure.
        
        Parameters:
            :bmg: <BlockMarkerGenerator>, the generator of block markers.

        Return:
            <BMPS>,
        '''
        return BMPO(self.OL[:],labels=self.labels[:],bmg=bmg)

    def eliminate_zeros(self,tol=1e-8):
        for o in self.OL:
            o[abs(o)<tol]=0
        return self

    def get(self,i,*args,**kwargs):
        return self.OL[i]

    def set(self,i,A,*args,**kwargs):
        self.OL[i]=A

    def insert(self,pos,cells):
        '''
        Insert cells into MPO.

        Parameters:
            :cells: list, tensors.
        '''
        _replace_cells(self,slice(pos,pos),cells)

    def remove(self,start,stop):
        '''
        Remove a segment from MPS.

        Parameters:
            :start: int,
            :stop: int,
        '''
        _replace_cells(self,slice(start,stop),[])

    def get_all(self):
        return self.OL[:]
 
    def check_link(self,l):
        sites=self.OL
        if l==self.nsite:
            return sites[-1].shape[self.rlink_axis]
        elif l>=0 and l<self.nsite:
            return sites[l].shape[self.llink_axis]
        else:
            raise ValueError('Link index out of range!')

    def chlabel(self,labels):
        self.labels=labels
        _auto_label(self.OL,labels)

    def compress(self,niter=2,tol=1e-8,maxN=Inf,kernel='svd'):
        nsite=self.nsite
        hndim=self.hndim
        len2k=kernel=='dpl'
        use_bm=hasattr(self,'bmg')
        llink_axis,rlink_axis,s1_axis,s2_axis=0,3,1,2
        acc=1.
        S=ones(1)
        sweeper=get_sweeper(start=(0,'->',0),stop=(niter,'<-',0),nsite=nsite-2)
        for itervar in sweeper:
            iit,direction,l=itervar
            right=direction=='->'  #right moving
            l=l+1
            #prepair the tensor, Get A,B matrix
            if right:
                A=self.get(l-1).mul_axis(S,llink_axis)
                B=self.get(l)
            else:
                B=self.get(l).mul_axis(S,rlink_axis)
                A=self.get(l-1)
            cbond_str=B.labels[llink_axis]
            #contract AB,
            AB=A*B
            ckernel=kernel
            if kernel=='dpl': ckernel=ckernel+('_c' if right else '_r')
            data=AB.svd(cbond=3,cbond_str=cbond_str,bmg=self.bmg if hasattr(self,'bmg') else None,signs=[1,1,-1,-1,1,1],kernel=ckernel)
            U,V=data[0],data[-1]

            if not len2k:
                S=data[1]
                #truncation
                if maxN<S.shape[0]:
                    tol=max(S[maxN],tol)
                kpmask=S>tol
                acc*=(1-sum(S[~kpmask]**2))

                #set data
                U,S,V=U.take(kpmask,axis=-1),S[kpmask],V.take(kpmask,axis=0)
                nS=norm(S); S/=nS
                if right:
                    U*=nS
                else:
                    V*=nS

            #set datas
            self.set(l-1,U)
            self.set(l,V)
        self.set(l-1,U*S)
        return self,1-acc

class BMPO(MPO):
    '''
    Matrix product operator.

    Attributes:
        :labels: len-3 list, the labels for legs, in the order site-site-link.
        :WL: list, the Matrix product operator datas.
        :OL: list, the matrix(Tensor) form of MPO datas.
        :hndim: integer, the Hilbert space dimension of single site.
        :nsite: integer, the number of sites.
    '''
    def __init__(self,OL,bmg,labels=["m","s",'b']):
        super(BMPO,self).__init__(OL,labels=labels)
        self.bmg=bmg
        _autoset_bms(self.OL,bmg)

    def unuse_bm(self):
        '''Get the non-block version <MPO>.'''
        return MPO(self.OL[:],labels=self.labels[:])

    def chlabel(self,labels):
        self.labels=labels
        _auto_label(self.OL,labels)

class PMPO(MPOBase):
    '''
    Periodic MPO with block structure.

    Attributes:
        :OP: Periodic structure of MPO
        :OH: Headers of MPO
    '''
    def __init__(self,OP,nsite,OH=None,labels=["m","s",'b']):
        self.OP=OP
        self.OH=OH
        self.labels=labels
        self.nsite=nsite

    @property
    def hndim(self):
        return self.OP.shape[1]

    @property
    def nnz(self):
        return sum(self.OP!=0)

    @property
    def H(self):
        raise NotImplementedError()

    def get(self,l):
        l=l%self.nsite
        if l==0 and self.OH is not None:
            T=self.OH[0]
        elif l==self.nsite-1 and self.OH is not None:
            T=self.OH[-1]
        else:
            T=self.OP

        slabel1,slabel2,llabel=self.labels
        labels=['%s_%s'%(llabel,l),
            '%s_%s'%(slabel1,l),
            '%s_%s'%(slabel2,l),
            '%s_%s'%(llabel,l+1)]
        return Tensor(T,labels=labels)

    def eliminate_zeros(self,tol=1e-8):
        for O in [self.OP]+self.OH:
            O[abs(O)<tol]=0
        return self

    def set(self,l,A):
        self.get(l)[...]=A

    def check_link(self,l):
        l=l%self.nsite
        if l==0: return self.OH[0].shape[0]
        elif l==self.nsite-1: return self.OH[-1].shape[-1]
        else: return self.OP.shape[0]

    def chlabel(self,labels):
        self.labels=labels

def WL2MPO(WL,labels=['m','s','b'],bmg=None):
    '''Construct MPO from WL.'''
    w0=WL[0]
    hndim=w0[w0!=0][0].hndim
    OL=[]
    for isite,wi in enumerate(WL):
        ni,nj=wi.shape[0],wi.shape[1]
        wm=zeros((ni,hndim,hndim,nj),dtype='complex128')
        for i in xrange(ni):
            for j in xrange(nj):
                if wi[i,j]!=0:
                    op=wi[i,j]
                    op.siteindex=isite
                    wm[i,...,j]=op.get_data()
        OL.append(wm)
    if bmg is None:
        return MPO(OL,labels=labels)
    else:
        return BMPO(OL,labels=labels,bmg=bmg)

def WL2OPC(WL,fix_sites=True):
    '''
    Return The serialized form of operator.
    '''
    #fix site indcies.
    if fix_sites:
        for isite,wi in enumerate(WL):
            ni,nj=wi.shape[0],wi.shape[1]
            for i in xrange(ni):
                for j in xrange(nj):
                    if wi[i,j]!=0:
                        wi[i,j].siteindex=isite
    opc=reduce(dot,WL)[0,0]
    for op in opc.ops:
        if hasattr(op,'opunits'):
            op.trim_I()
    return opc


def OPC2MPO(op,method='direct',nsite=None,bmg=None):
    '''
    Construct Matrix product Operator corresponding to this <OpCollection>/<OpString>/<OpUnit> instance.

    Parameters:
        :method: str, 'direct' or 'addition'.
            
            * direct: construct directly using complicated manipulation.
            * additive: construct by <MPO> addition operations term by term.
        :nsite: int, number of sites.
        :bmg: <BlockMarkerGenerator>,

    Return:
        <MPO>(<BMPO> if bmg is not None).
    '''
    if nsite is None:nsite=op.maxsite+1
    hndim=op.hndim
    OL=[identity(hndim).reshape([1,hndim,hndim,1])]*nsite
    if hasattr(op,'opunits'):
        for ou in op.opunits:
            OL[ou.siteindex]=ou.get_data().reshape([1,hndim,hndim,1])
    elif hasattr(op,'ops'):
        if method=='direct':
            nop=op.nop
            opmatrix=zeros((nop,nsite),dtype='O')
            for i,opi in enumerate(op.ops):
                if isinstance(opi,OpUnit):
                    opmatrix[i,opi.siteindex]=opi
                else:
                    opmatrix[i,asarray(opi.siteindices)]=opi.opunits

            constr=MPOConstructor(nsite=nsite,hndim=hndim)
            for opi in op.ops:
                constr.read1(opi)
            mpo=constr.compile()
            if bmg is not None: mpo=mpo.use_bm(bmg)
            return mpo
        else:
            mpo_collection=[OPC2MPO(op,nsite=nsite) for op in op.ops]
            OL=[]
            for i in xrange(nsite):
                if i==0:
                    oi=concatenate([mpo.get(i) for mpo in mpo_collection],axis=3)
                elif i==nsite-1:
                    oi=concatenate([mpo.get(i) for mpo in mpo_collection],axis=0)
                else:
                    oi=fmerge_mpo(concatenate([mpo.get(i) for mpo in mpo_collection],axis=0))
                OL.append(oi)
    else:
        OL[op.siteindex]=op.get_data().reshape([1,hndim,hndim,1])

    mpo=MPO(OL) if bmg is None else BMPO(OL,bmg=bmg)
    return mpo
