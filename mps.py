'''
Matrix Product State.
'''

from numpy import *
from scipy.linalg import svd,norm,block_diag
import pdb,copy,numbers
from abc import ABCMeta, abstractmethod

from utils import inherit_docstring_from
from blockmatrix import BlockMarker
from tensor import BLabel,Tensor
from btensor import BTensor
from tensorlib import svdbd,tensor_block_diag,check_validity_tensor

__all__=['MPSBase','MPS','BMPS','mPS']

ZERO_REF=1e-12

def _mps_sum(mpses,labels=('s','a')):
    '''
    Summation over <MPS>es.

    Parameters:
        :mpses: list of <MPS>, instances to be added.
        :labels: list of str, the new labels for added state.

    Return:
        <MPS>, the added MPS.
    '''
    if len(mpses)==1:
        return mpses[0]
    elif len(mpses)==0:
        raise ValueError('At least 1 <MPS> instances are required.')
    assert(all(diff([mps.l for mps in mpses])==0))  #the l should be same
    mps0=mpses[0]
    l=mps0.l
    MLs=[mps.get_all(attach_S='') for mps in mpses]
    hndim=mps0.hndim
    nsite=mps0.nsite
    site_axis=mps0.site_axis

    #get datas
    #ML,bms=[],[]
    #for i in xrange(nsite):
        #ai=transpose([block_diag(*[mi[i].take(j,axis=site_axis) for mi in MLs]) for j in xrange(hndim)],axes=[1,0,2])
        #ML.append(ai)
    ML=[tensor_block_diag(mis,axes=(2,) if i==0 else ((0,) if i==nsite-1 else (0,2))) for i,mis in enumerate(zip(*MLs))]
    ##fix the ends
    #ML[0]=ML[0].sum(axis=0)[newaxis,...]
    #ML[-1]=ML[-1].sum(axis=2)[...,newaxis]

    #get S matrix
    if l==0 or l==nsite:
        S=ones(1,dtype='complex128')
    else:
        S=ones(sum([len(mps.S) for mps in mpses]))
    return mPS(ML,l,S=S,labels=labels,bmg=mps0.bmg if hasattr(mps0,'bmg') else None)

def _autoset_bms(TL,bmg,check_conflicts=False):
    '''
    Auto setup blockmarkers for <MPS> and <MPO>.

    Parameters:
        :TL: list, a list of tensors.
        :bmg: <BlockMarkerGenerator>,
        :check_conflicts: bool,

    Return:
        list, tensors with block marker set.
    '''
    nsite=len(TL)
    bm1=bmg.bm1_
    is_mps=ndim(TL[0])==3
    pm=slice(None)
    for i in xrange(nsite):
        #cell is a tensor tensor(al,sup,ar), first permute left axes to match the transformation of block marker
        cell=TL[i][pm]
        #get bml
        bml=bmr if i!=0 else bmg.bm0
        #setup left, site labels.
        cell.labels[:2]=[
                BLabel(cell.labels[0],bml),
                BLabel(cell.labels[1],bm1)]
        if not is_mps:
            cell.labels[2]=BLabel(cell.labels[2],bm1)
            #merge left&site1&site2 labels, in order to get the good quantum number flow to right.
            cell_flat=cell.merge_axes(slice(0,3),signs=[1,1,-1],bmg=bmg)
        else:
            #merge left&site labels, in order to get the good quantum number flow to right.
            cell_flat=cell.merge_axes(slice(0,2),signs=[1,1],bmg=bmg)
        x,y=where(cell_flat)
        bmlc=cell_flat.labels[0].bm  #the block label for each dimension of row
        qns=zeros([cell_flat.shape[1],bml.qns.shape[-1]],dtype=bm1.qns.dtype)
        #get bmr
        #detect conflicts!
        if check_conflicts:
            for xi,yi in zip(x,y):
                if any(qns[yi]!=0) and any(qns[yi]!=bmlc.qns[xi]):
                    print 'Conflict check failed!'
                    pdb.set_trace()
                else:
                    qns[yi]=bmlc.qns[xi]
        else:
            qns[y]=bmlc.qns[x]
        bmr,info=BlockMarker(Nr=arange(cell_flat.shape[1]+1),qns=qns).sort(return_info=True); pm=info['pm']
        bmr=bmr.compact_form()
        cell=cell[...,pm]
        cell.labels[-1]=BLabel(cell.labels[-1],bmr)
        TL[i]=cell
    return TL

class MPSBase(object):
    '''
    The Base class of Matrix Product state.

    Attributes:
        :hndim: The number of channels on each site.
        :nsite: The number of sites.
        :site_axis/llink_axis/rlink_axis: The specific axes for site, left link, right link.
        :state: The state in the normal representation.
    '''

    __metaclass__ = ABCMeta

    @property
    def site_axis(self):
        '''int, axis of site index.'''
        return 1

    @property
    def llink_axis(self):
        '''int, axis of left link index.'''
        return 0

    @property
    def rlink_axis(self):
        '''int, axis of right link index.'''
        return 2

    @property
    @abstractmethod
    def hndim(self):
        '''int, number of state in a single site.'''
        pass

    @property
    @abstractmethod
    def nsite(self):
        '''int, number of sites.'''
        pass

    @property
    @abstractmethod
    def state(self):
        '''1d array, vector representation of this MPS'''
        pass

    @abstractmethod
    def tobra(self,labels):
        '''
        Get the bra conterpart.
        
        Parameters:
            :labels: list, label strings for site and bond.

        Return:
            <MPS>,
        '''
        pass

    @abstractmethod
    def toket(self,labels):
        '''
        Get the ket conterpart.

        Parameters:
            :labels: list, label strings for site and bond.

        Return:
            <MPS>,
        '''
        pass

class MPS(MPSBase):
    '''
    Matrix product states.

    Attributes:
        :ML: list of 3D array, the sequence of A/B-matrices.
        :l/S: int, 1D array, the division point of left and right scan, and the singular value matrix at the division point.
            Also, it is the index of the non-unitary M-matrix(for the special case of l==N, S is attached to the right side of N-1-th matrix)
        :is_ket: bool, It is a ket if True else bra.
        :labels: len-2 list of str, the labels for auto-labeling in MPS [site, link].
    '''
    def __init__(self,ML,l,S,is_ket=True,labels=['s','a']):
        assert(ndim(S)==1)
        assert(len(labels)==2)
        self.ML=ML
        self.labels=list(labels)
        self.l=l
        self.S=S
        self.is_ket=is_ket

    def __str__(self):
        string='<MPS,%s>\n'%(self.nsite)
        string+='\n'.join(['  A[s=%s] (%s x %s) (%s,%s,%s)'%(\
                a.shape[self.site_axis],a.shape[self.llink_axis],a.shape[self.rlink_axis],\
                a.labels[self.llink_axis],a.labels[self.site_axis],a.labels[self.rlink_axis]\
                ) for a in self.ML[:self.l]])+'\n'
        string+='  S      %s\n'%(self.S.shape,)
        string+='\n'.join(['  B[s=%s] (%s x %s) (%s,%s,%s)'%(\
                a.shape[self.site_axis],a.shape[self.llink_axis],a.shape[self.rlink_axis],\
                a.labels[self.llink_axis],a.labels[self.site_axis],a.labels[self.rlink_axis]\
                ) for a in self.ML[self.l:]])
        return string

    def __add__(self,target):
        if isinstance(target,MPS):
            return _mps_sum([self,target],labels=self.labels[:])
        else:
            raise TypeError('Can not add <MPS> with %s'%target.__class__)

    def __radd__(self,target):
        if isinstance(target,MPS):
            return target.__add__(self)
        else:
            raise TypeError('Can not add %s with <MPS>'%target.__class__)

    def __sub__(self,target):
        if isinstance(target,MPS):
            return self.__add__(-target)
        else:
            raise TypeError('Can not subtract <MPS> with %s'%target.__class__)

    def __rsub__(self,target):
        if isinstance(target,MPS):
            return target.__sub__(self)
        else:
            raise TypeError('Can not subtract %s with <MPS>'%target.__class__)

    def __mul__(self,target):
        hndim=self.hndim
        site_axis=self.site_axis
        if isinstance(target,numbers.Number):
            mps=self.toket(labels=self.labels[:]) if self.is_ket else self.tobra(labels=self.labels[:])
            mps.S=self.S*target
            return mps
        elif isinstance(target,MPS):
            if self.is_ket or not target.is_ket:
                raise Exception('Not implemented for multipling ket on the left side.')
            S=identity(1)
            for mi,tmi in zip(self.get_all(attach_S=''),target.get_all(attach_S='')):
                #need some check!
                S=sum([mi.take(j,axis=site_axis).toarray().T.dot(S).dot(tmi.take(j,axis=site_axis).toarray()) for j in xrange(hndim)],axis=0)
            return S
        else:
            raise TypeError('Can not multiply <MPS> with %s'%target.__class__)

    def __rmul__(self,target):
        if isinstance(target,numbers.Number):
            return self.__mul__(target)
        elif isinstance(target,MPS):
            return target.__mul__(self)
        else:
            raise TypeError('Can not multiply %s with <MPS>'%target.__class__)

    def __imul__(self,target):
        if isinstance(target,numbers.Number):
            self.S=self.S*target
            return self
        else:
            raise TypeError('Can not i-multiply by %s'%target.__class__)

    def __neg__(self):
        return -1*self

    def __div__(self,target):
        hndim=self.hndim
        if isinstance(target,numbers.Number):
            mps=self.toket(labels=self.labels[:]) if self.is_ket else self.tobra(labels=self.labels[:])
            mps.S=self.S/target
            return mps
        else:
            raise TypeError('Can not divide <MPS> with %s'%target.__class__)

    def __rdiv__(self,target):
        if isinstance(target,numbers.Number):
            return self.__div__(target)
        else:
            raise TypeError('Can not divide %s with <MPS>'%target.__class__)

    def __idiv__(self,target):
        if isinstance(target,numbers.Number):
            self.S=self.S/target
            return self
        else:
            raise TypeError('Can not i-divide by %s'%target.__class__)

    def __lshift__(self,k):
        '''Left move l-index by k.'''
        tol=ZERO_REF
        maxN=Inf
        if isinstance(k,tuple):
            k,tol,maxN=k
        return self.canomove(-k,tol=tol,maxN=maxN)

    def __rshift__(self,k):
        '''Right move l-index by k.'''
        tol=ZERO_REF
        maxN=Inf
        if isinstance(k,tuple):
            k,tol,maxN=k
        return self.canomove(k,tol=tol,maxN=maxN)

    @property
    def hndim(self):
        '''The number of state in a single site.'''
        return self.ML[0].shape[self.site_axis]

    @property
    @inherit_docstring_from(MPSBase)
    def nsite(self):
        return len(self.ML)

    @property
    @inherit_docstring_from(MPSBase)
    def state(self):
        ML=self.get_all()
        res=reduce(lambda x,y:x*y,ML)
        return asarray(res.ravel())

    def get(self,siteindex,attach_S='',*args,**kwargs):
        '''
        Get the tensor element for specific site.

        Parameters:
            :siteindex: int, the index of site.
            :attach_S: bool, attach the S-matrix to

                * ''  -> don't attach to any block.
                * 'A' -> right most A block.
                * 'B' -> left most B block.

        Return:
            <Tensor>,
        '''
        assert(attach_S in ['A','B',''])
        if siteindex>self.nsite or siteindex<0:
            raise ValueError('l=%s out of bound!'%siteindex)
        res=self.ML[siteindex]
        if  attach_S=='A' and siteindex==self.l-1:
            res=res.mul_axis(self.S,self.rlink_axis)
        elif attach_S=='B' and siteindex==self.l:
            res=res.mul_axis(self.S,self.llink_axis)
        return res

    def set(self,siteindex,A,*args,**kwargs):
        '''
        Get the matrix for specific site.

        Parameters:
            :siteindex: int, the index of site.
            :A: <Tensor>, the data
        '''
        if siteindex>self.nsite or siteindex<0:
            raise ValueError('l=%s out of bound!'%siteindex)
        self.ML[siteindex]=A

    def check_link(self,l):
        '''
        The bond dimension for l-th link.

        Parameters:
            :l: int, the bond index.

        Return:
            int, the bond dimension.
        '''
        if l==self.nsite:
            return self.ML[-1].shape[self.rlink_axis]
        elif l>=0 and l<self.nsite:
            return self.ML[l].shape[self.llink_axis]
        else:
            raise ValueError('Link index out of range!')

    def get_all(self,attach_S=''):
        '''
        Get the concatenation of A and B sectors.

        Parameters:
            :attach_S: bool, attach the S-matrix to

                * ''  -> don't attach to any block.
                * 'A' -> right most A block.
                * 'B' -> left most B block.

        Return:
            list,
        '''
        assert(attach_S in ['A','B',''])
        ML=self.ML[:]
        l=self.l
        nsite=self.nsite
        if attach_S=='': attach_S='A' if l!=0 else 'B'
        #fix no S cases
        if l==0 and attach_S=='A':
            attach_S='B'
        if l==nsite and attach_S=='B':
            attach_S='A'

        if attach_S=='A':
            ML[l-1]=ML[l-1].mul_axis(self.S,self.rlink_axis)
        elif attach_S=='B':
            ML[l]=ML[l].mul_axis(self.S,axis=self.llink_axis)
        return ML

    def canomove(self,nstep,tol=ZERO_REF,maxN=Inf):
        '''
        Move l-index by one with specific direction.
        
        Parameters:
            :nstep: int, move l nstep towards right.
            :tol: float, the tolerence for compression.
            :maxN: int, the maximum dimension.

        Return:
            float, approximate truncation error.
        '''
        use_bm=hasattr(self,'bmg')
        nsite=self.nsite
        hndim=self.hndim
        llink_axis,rlink_axis,site_axis=self.llink_axis,self.rlink_axis,self.site_axis
        #check and prepair data
        if self.l+nstep>nsite or self.l+nstep<0:
            raise ValueError('Illegal Move!')
        right=nstep>0
        acc=1.
        for i in xrange(abs(nstep)):
            #prepair the tensor, Get A,B matrix
            self.l=(self.l+1) if right else (self.l-1)
            if right:
                A=self.ML[self.l-1].mul_axis(self.S,llink_axis)
                if self.l==nsite:
                    S=sqrt((A**2).sum())
                    self.S=array([S])
                    self.ML[-1]=A/S
                    return 1-acc
                B=self.ML[self.l]
            else:
                B=self.ML[self.l].mul_axis(self.S,rlink_axis)
                if self.l==0:
                    S=sqrt((B**2).sum())
                    self.S=array([S])
                    self.ML[0]=B/S
                    return 1-acc
                A=self.ML[self.l-1]
            cbond_str=B.labels[llink_axis]
            #contract AB,
            AB=A*B
            #transform it into matrix form and do svd decomposition.
            if use_bm:
                AB=AB.merge_axes(bmg=self.bmg,sls=slice(0,2),signs=[1,1]).merge_axes(bmg=self.bmg,sls=slice(1,3),signs=[-1,1])
                AB,pms=AB.b_reorder(return_pm=True)
                U,S,V=svdbd(AB,cbond_str=cbond_str)
            else:
                AB=AB.reshape([-1,prod(AB.shape[2:])])
                U,S,V=svd(AB,full_matrices=False)

            #truncation
            if maxN<S.shape[0]:
                tol=max(S[maxN],tol)
            kpmask=S>tol
            acc*=(1-sum(S[~kpmask]**2))

            #unpermute blocked U,V and get c label
            if use_bm:
                U,S,V=U.take(kpmask,axis=1).take(argsort(pms[0]),axis=0),S[kpmask],V.take(kpmask,axis=0).take(argsort(pms[1]),axis=1)
                clabel=U.labels[1]
            else:
                U,S,V=U[:,kpmask],S[kpmask],V[kpmask]
                clabel=cbond_str

            #set datas
            self.S=S
            self.ML[self.l-1]=U.split_axis(0,nlabels=A.labels[:2])
            self.ML[self.l]=V.split_axis(1,nlabels=B.labels[1:])
        return 1-acc

    def use_bm(self,bmg,sharedata=True):
        '''
        Use <BlockMarker> to indicate block structure.
        
        Parameters:
            :bmg: <BlockMarkerGenerator>, the generator of block markers.
            :sharedata: bool, the new <BMPS> will share the data with current one if True.

        Return:
            <BMPS>,
        '''
        mps=BMPS([ai.make_copy(copydata=not sharedata) for ai in self.ML],self.l,\
                self.S if sharedata else self.S[...],is_ket=self.is_ket,labels=self.labels[:],bmg=bmg)
        return mps

    @inherit_docstring_from(MPSBase)
    def toket(self,labels=None):
        if labels is None:
            labels=self.labels[:]

        mps=MPS([ai.make_copy(copydata=False) if self.is_ket else ai.conj() for ai in self.ML],self.l,\
                self.S if self.is_ket else self.S.conj(),is_ket=True,labels=labels)
        return mps

    @inherit_docstring_from(MPSBase)
    def tobra(self,labels=None):
        #get new labels,
        if labels is None:
            labels=self.labels[:]

        mps=MPS([ai.make_copy(copydata=False) if not self.is_ket else ai.conj() for ai in self.ML],self.l,\
                self.S if not self.is_ket else self.S.conj(),is_ket=False,labels=labels)
        return mps

    def tovidal(self):
        '''
        Transform to the Vidal form.
        '''
        llink_axis,rlink_axis=self.llink_axis,self.rlink_axis
        nsite=self.nsite
        hndim=self.hndim
        LL=[]
        GL=[]
        factor=norm(self.S)
        S=self.S/factor
        LL.append(S)
        for i in xrange(self.l-1,-1,-1):
            UL=U0*LL[-1] if i!=self.l-1 else diag(S)
            A=self.ML[i]
            A=(A.reshape([-1,A.shape[2]]).dot(UL)).reshape([-1,A.shape[2]*hndim])
            U0,L,V=svd(A,full_matrices=False)
            LL.append(L)
            GL.append(V.reshape(self.ML[i].shape)/LL[-2].reshape([-1]+[1]*(2-rlink_axis)))
        LL,GL=LL[::-1],GL[::-1]
        for i in xrange(nsite-self.l):
            LV=LL[-1][:,newaxis]*V0 if i!=0 else diag(S)
            B=self.ML[self.l+i]
            B=LV.dot(B.reshape([B.shape[0],-1])).reshape([B.shape[0]*hndim,-1])
            U,L,V0=svd(B,full_matrices=False)
            LL.append(L)
            GL.append(U.reshape(self.ML[self.l+i].shape)/LL[-2].reshape([-1]+[1]*(2-llink_axis)))
        if self.l>0: factor=factor*U0
        if self.l<nsite: factor=factor*V0
        vmps=VidalMPS(GL,LL[1:-1],labels=self.labels[:],factor=factor.item())
        return vmps

    def chlabel(self,labels):
        '''
        Change the label of specific axis.
        
        Parametrs:
            :labels: list, the new labels.
        '''
        nsite=self.nsite
        self.labels=labels
        slabel,llabel=labels
        for l,ai in enumerate(self.ML):
            if hasattr(ai.labels[0],'bm'):
                ai.labels[self.site_axis]=ai.labels[self.site_axis].chstr('%s_%s'%(slabel,l))
                ai.labels[self.llink_axis]=ai.labels[self.llink_axis].chstr('%s_%s'%(llabel,l))
                ai.labels[self.rlink_axis]=ai.labels[self.rlink_axis].chstr('%s_%s'%(llabel,l+1))
            else:
                ai.labels[self.site_axis]='%s_%s'%(slabel,l)
                ai.labels[self.llink_axis]='%s_%s'%(llabel,l)
                ai.labels[self.rlink_axis]='%s_%s'%(llabel,l+1)

    def query(self,serie):
        '''
        Query the magnitude of a state.

        Parameters:
            :serie: 1d array, sgimas(indices of states on sites).

        Return:
            number, the amplitude.
        '''
        state=identity(1)
        site_axis=self.site_axis
        for si,Mi in zip(serie,self.get_all(attach_S='B'))[::-1]:
            state=Mi.take(si,axis=site_axis).dot(state)
        return state.item()

    def compress(self,tol=1e-8,maxN=200,niter=3):
        '''
        Compress this state.

        Parameters:
            :tol: float, the tolerence used for compressing.
            :maxN: int, the maximum retained states.
            :niter: int, number of iterations.

        Return:
            float, approximate truncation error.
        '''
        nsite,l=self.nsite,self.l
        M=self.check_link(nsite/2)
        dM=max(M-maxN,0)
        acc=1.
        for i in xrange(niter):
            m1=maxN+int(dM*((niter-i-0.5)/niter))
            m2=maxN+int(dM*((niter-i-1.)/niter))
            acc*=1-(self>>(nsite-l,tol,m1))
            acc*=1-(self<<(nsite-l,tol,m2))
            acc*=1-(self<<(l,tol,m1))
            acc*=1-(self>>(l,tol,m2))
        print 'Compression of MPS Done!'
        return 1-acc

    def recanonicalize(self,left=True,tol=1e-12,maxN=Inf):
        '''
        Trun this MPS into canonical form.

        Parameters:
            :left: bool, use left canonical if True, else right canonical
            :tol: float, the tolerence.
            :maxN: int, the maximum retained states.
        '''
        nsite,l=self.nsite,self.l
        if not left:
            self>>(nsite-l,tol,maxN)
            self<<(nsite,tol,maxN)
            self>>(l,tol,maxN)
        else:
            self<<(l,tol,maxN)
            self>>(nsite,tol,maxN)
            self<<(nsite-l,tol,maxN)
        return self

class BMPS(MPS):
    '''
    MPS with block structure.

    Attributes:
        :bmg: <BlockMarkerGenerator>, the block infomation manager.

        *see <MPS> for more.*
    '''
    def __init__(self,ML,l,S,bmg,**kwargs):
        super(BMPS,self).__init__(ML,l,S,**kwargs)
        self.bmg=bmg

    def unuse_bm(self,sharedata=True):
        '''
        Get the non-block version of current ket.
        
        Parameters:
            :sharedata: bool, the new <MPS> will share the data with current one if True.

        Return:
            <MPS>,
        '''
        mps=MPS([ai.make_copy(copydata=not sharedata) for ai in self.ML],self.l,\
                self.S if sharedata else self.S[...],is_ket=self.is_ket,labels=self.labels[:])
        return mps

    @inherit_docstring_from(MPS)
    def toket(self,labels=None):
        if labels is None:
            labels=self.labels[:]

        mps=BMPS([ai.make_copy(copydata=False) if self.is_ket else ai.conj() for ai in self.ML],self.l,\
                self.S if self.is_ket else self.S.conj(),is_ket=True,labels=labels,bmg=self.bmg)
        return mps

    @inherit_docstring_from(MPS)
    def tobra(self,labels=None):
        if labels is None:
            labels=self.labels[:]

        mps=BMPS([ai.make_copy(copydata=False) if not self.is_ket else ai.conj() for ai in self.ML],self.l,\
                self.S if not self.is_ket else self.S.conj(),is_ket=False,labels=labels,bmg=self.bmg)
        return mps

def mPS(ML,l,S,is_ket=True,labels=['s','a'],bmg=None,bms=None):
    '''
    Construct MPS.
    '''
    nML=[]
    s,a=labels
    for i,M in enumerate(ML):
        lbs=['%s_%s'%(a,i),'%s_%s'%(s,i),'%s_%s'%(a,i+1)]
        if isinstance(M,ndarray):
            mi=Tensor(M,labels=lbs)
        elif isinstance(M,BTensor):
            mi=M.make_copy(labels=[bl.chstr(lb) for bl,lb in zip(M.labels,lbs)],copydata=False)
        else:
            raise TypeError
        #set up block markers manually.
        if bms is not None:
            mi.labels=[BLabel(lbs[0],bms[i]),BLabel(lbs[1],bmg.bm1_),BLabel(lbs[2],bms[i+1])]
        nML.append(mi)
    if bmg is None:  #a normal MPS
        return MPS(nML,l,S,is_ket=is_ket,labels=labels)
    else:
        if isinstance(nML[0],Tensor) and bms is None: _autoset_bms(nML,bmg)
        return BMPS(nML,l,S,is_ket=is_ket,labels=labels,bmg=bmg)
