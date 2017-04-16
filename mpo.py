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

__all__=['OpUnit','OpString','OpCollection','MPO','MPOConstructor','OpUnitI','WL2MPO','WL2OPC','BMPO']
UNSETTLED='-'
ZERO_REF=1e-12

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

class OpUnit(object):
    '''
    Site-wise Operator, the basic element in constructing Operators.

    Attributes:
        :label: string, the label.
        :data: matrix, the data of this Operator unit.
        :siteindex: integer, the site index, or leave '-' to be un specified.
        :factor: number, the factor acting on data.
        :math_str: string/None, the string for mathematical display.
        :fermionic: bool, this is a fermionic operator(sign problem arises) or not.

    Readonly Attributes:
        :hndim: int, single site Hilbert space dimenion.
        :siteindices: list, siteindices interface to be compatible with opstring.
        :opunits: list, opunits interface to be compatible with opstring.
    '''
    def __init__(self,label,data,siteindex=UNSETTLED,factor=1.,math_str=None,fermionic=False):
        self.label=label
        self.data=data
        self.siteindex=siteindex
        self.factor=factor
        self.fermionic=fermionic
        if math_str is None:
            self.__math_str__=self.label
        else:
            self.__math_str__=math_str

    def __eq__(self,target):
        '''Note: site index is not considered as internal attribute!'''
        tol=1e-8
        if isinstance(target,OpUnit):
            if target.siteindex!=self.siteindex:
                return False
            return allclose(target.get_data(),self.get_data())
        elif isinstance(target,(OpString,OpCollection)):
            return False
        elif isinstance(target,numbers.Number):
            return allclose(target/self.factor*identity(self.hndim),self.data)
        else:
            raise TypeError('Wrong type for comparison -> %s'%target.__class__)

    def __str__(self):
        return _format_factor(self.factor)+'%s[%s] = %s'%(self.label,self.siteindex,self.get_data())

    def __repr__(self):
        return _format_factor(self.factor)+self.label+('' if self.siteindex=='-' else '[%s]'%self.siteindex)

    def __mul__(self,target):
        if isinstance(target,numbers.Number):
            if target==0:
                return 0
            else:
                res=OpUnit(self.label,self.data,self.siteindex,self.factor,math_str=self.__math_str__,fermionic=self.fermionic)
                res.factor*=target
                return res
        elif isinstance(target,OpUnitI):
            return copy.copy(self)
        elif isinstance(target,OpUnit):
            if target.siteindex!=self.siteindex:
                res=OpString([self])
                res*=target
                return res
            else:
                return OpUnit(label=self.label+'*'+target.label,data=self.data.dot(target.data),factor=self.factor*target.factor,\
                        siteindex=self.siteindex,math_str=self.__math_str__+target.__math_str__,fermionic=self.fermionic^target.fermionic)
        elif isinstance(target,OpString):
            return target.__rmul__(self)
        else:
            raise TypeError('Can not multiply %s with %s.'%(self.__class__,target.__class__))

    def __rmul__(self,target):
        if isinstance(target,numbers.Number):
            return self.__mul__(target)
        elif isinstance(target,OpUnit):
            return target.__mul__(self)
        elif isinstance(target,OpString):
            return target.__mul__(self)
        else:
            raise TypeError('Can not multiply %s with %s.'%(target.__class__,self.__class__))

    def __imul__(self,target):
        if isinstance(target,numbers.Number):
            if target==0:
                return 0
            self.factor*=target
            return self
        else:
            raise TypeError('Can not multiply %s with %s.'%(self.__class__,target.__class__))

    def __add__(self,target):
        if isinstance(target,OpCollection):
            target.__radd__(self)
        if isinstance(target,numbers.Number):
            if target==0:
                #return deepcopy(self)
                return copy.copy(self)
            else:
                raise TypeError('OpUnit not allowed to add with nonzeros.')
        elif isinstance(target,OpUnit):
            if target.siteindex==self.siteindex:
                if target.fermionic!=self.fermionic:
                    raise Exception('Can not add opunits with difference parity!')
                return OpUnit(label=self.label+'+'+target.label,data=self.factor*self.data+target.factor*target.data,siteindex=self.siteindex,math_str=self.__math_str__+'+'+target.__math_str__,fermionic=self.fermionic)
            else:
                return OpCollection([self,target])
        elif isinstance(target,OpString):
            return OpCollection([self,target])
        else:
            raise TypeError('Can not add %s with %s.'%(self.__class__,target.__class__))

    def __radd__(self,target):
        if isinstance(target,OpCollection):
            target.__add__(self)
        elif isinstance(target,(OpUnit,OpString)):
            return target.__add__(self)
        elif isinstance(target,numbers.Number):
            if target==0:
                #return deepcopy(self)
                return copy.copy(self)
            else:
                raise TypeError('OpUnit not allowed to add with nonzeros.')
        else:
            raise TypeError('Can not add %s with %s.'%(self.__class__,target.__class__))

    def __sub__(self,target):
        return self.__add__(-target)

    def __rsub__(self,target):
        return target.__add__(-self)

    def __isub__(self,target):
        return self.__iadd__(-target)

    def __div__(self,target):
        if isinstance(target,numbers.Number):
            res=copy.copy(self)
            res.factor/=target
            return res
        else:
            raise TypeError('Can not divide %s with %s.'%(self.__class__,target.__class__))

    def __idiv__(self,target):
        if isinstance(target,numbers.Number):
            self.factor/=target
            return self
        else:
            raise TypeError('Can not divide %s with %s.'%(self.__class__,target.__class__))

    def __neg__(self):
        return -1*self

    @property
    def hndim(self):
        '''The dimension of Hilbertspace at single site.'''
        return self.data.shape[-1]

    @property
    def siteindices(self):
        return [self.siteindex]

    @property
    def opunits(self):
        return [self]

    @property
    def maxsite(self):
        '''
        int, the maximum refered site.
        '''
        return self.siteindex

    def H(self,nsite):
        '''Get the Hamiltonian matrix.

        Parameters:
            :nsite: int, number of sites.

        Return:
            matrix, the operator.
        '''
        hndim=self.hndim
        nl=self.siteindex
        nr=nsite-self.siteindex-1
        return kron(kron(identity(hndim**nl),self.get_data()),identity(hndim**nr))

    def as_site(self,i):
        '''Get a copy of this operator at site i.'''
        res=copy.copy(self)
        res.siteindex=i
        return res

    def get_mathstr(self,factor=1.):
        '''Get the math string for display.'''
        factor=factor*self.factor
        if abs(factor-1)<1e-15:
            factor_str=''
        else:
            factor_str=format(factor,'.3f').rstrip('0').rstrip('.')
        return r'$%s{%s}(%s)}$'%(factor_str,self.__math_str__,self.siteindex)

    def get_data(self,dense=True):
        '''
        Get the data(taking factor into consideration).

        Parameters:
            :dense: bool, dense or not.

        Return:
            matrix, the data.
        '''
        if dense:
            return self.factor*self.data
        else:
            return self.factor*sps.csr_matrix(self.data)

    def toMPO(self,nsite):
        '''
        Turn to MPO format with bond dimension 1.

        Parameters:
            :nsite: int, number of sites.

        Return:
            <MPO>,
        '''
        hndim=self.hndim
        OL=[identity(hndim).reshape([1,hndim,hndim,1])]*nsite
        OL[self.siteindex]=self.get_data().reshape([1,hndim,hndim,1])
        return MPO(OL)

class OpString():
    '''
    Multiplication serie of operator units.
    e.g. Sx(i)*Sx(j).

    Attributes:
        :opunits: list, a list of <OpUnit> instances.

    Readonly Attributes:
        :hndim: int, single site Hilbert space dimenion.
        :nunit: number of constructing <OpUnit>s
        :siteindices: 1d array, the site indices,
        :fermionic: bool, this is a fermionic operator or not.
    '''
    def __init__(self,opunits=None):
        if opunits is None:
            self.opunits=[]
        elif isinstance(opunits,OpString):
            self.opunits=opunits.opunits[:]
        else:
            self.opunits=list(opunits)
            self._tonormalorder(care_sign=True)

    def __len__(self):
        return len(self.opunits)

    def __str__(self):
        return '<OpString> '+'*'.join([op.__repr__() for op in self.opunits])

    def __repr__(self):
        return '*'.join([op.__repr__() for op in self.opunits])

    def __copy__(self):
        newone=OpString(opunits=list(self.opunits))
        return newone

    def __add__(self,target):
        if isinstance(target,OpCollection):
            return target.__radd__(self)
        elif isinstance(target,(OpString,OpUnit)):
            return OpCollection([self,target])
        elif target==0:
            return copy.copy(self)
        else:
            raise TypeError('OpString not allowed to add with %s.'%target.__class__)

    def __radd__(self,target):
        if isinstance(target,(OpCollection,OpUnit,OpString)):
            target.__add__(self)
        elif target==0:
            return OpString(self.opunits)
        else:
            raise TypeError('OpString not allowed to add with %s.'%target.__class__)

    def __mul__(self,target):
        if isinstance(target,OpString):
            res=OpString(self.opunits)
            res*=target
            return res
        elif isinstance(target,OpUnit):
            res=OpString(self.opunits)
            res*=target
            return res
        elif isinstance(target,numbers.Number):
            if target==0:
                return 0
            else:
                nop=copy.copy(self)
                nop.opunits[0]=nop.opunits[0]*target
                return nop
        else:
            raise TypeError('Can not multiply %s with %s(%s).'%(self.__class__,target,target.__class__))

    def __rmul__(self,target):
        if isinstance(target,OpString):
            return target.__mul__(self)
        elif isinstance(target,OpUnit):
            res=OpString(self.opunits)
            res._insert1(target,from_right=False,care_sign=True)
            return res
        elif isinstance(target,numbers.Number):
            return self.__mul__(target)
        else:
            raise TypeError('Can not multiply %s with %s.'%(target.__class__,self.__class__))

    def __imul__(self,target):
        if isinstance(target,OpString):
            for ou in target.opunits:
                self._insert1(ou,from_right=True,care_sign=True)
            return self
        elif isinstance(target,OpUnit):
            self._insert1(target,from_right=True,care_sign=True)
            return self
        elif isinstance(target,numbers.Number):
            if target==0:
                return 0
            self.opunits[0]=self.opunits[0]*target
            return self
        else:
            raise TypeError('Can not multiply %s with %s.'%(self.__class__,target.__class__))

    def __sub__(self,target):
        return self.__add__(-target)

    def __rsub__(self,target):
        return target.__add__(-self)

    def __isub__(self,target):
        return self.__iadd__(-target)

    def __div__(self,target):
        if isinstance(target,numbers.Number):
            res=copy.copy(self)
            res.opunits[0]=res.opunits[0]/target
            return res
        else:
            raise TypeError('Can not divide %s with %s.'%(self.__class__,target.__class__))

    def __idiv__(self,target):
        if isinstance(target,numbers.Number):
            self.opunits[0]=self.opunits[0]/target
            return self
        else:
            raise TypeError('Can not divide %s with %s.'%(self.__class__,target.__class__))

    def __neg__(self):
        return -1*self

    @property
    def nunit(self):
        '''Number of <OpUnit> instances'''
        return len(self.opunits)

    @property
    def hndim(self):
        '''The dimension of Hamiltonian'''
        if self.nunit!=0:
            return self.opunits[0].hndim

    @property
    def siteindices(self):
        '''The site indices.'''
        return [op.siteindex for op in self.opunits]

    @property
    def fermionic(self):
        '''Is fermionic type or not.'''
        return bool(sum([ou.fermionic for ou in self.opunits])%2)

    @property
    def maxsite(self):
        '''
        int, the maximum refered site.
        '''
        return self.siteindices[-1]

    def H(self,nsite):
        '''Get the Hamiltonian matrix.

        Parameters:
            :nsite: int, number of sites.

        Return:
            matrix, the operator.
        '''
        hndim=self.hndim
        one=identity(hndim)
        nl=self.siteindices[0]
        nr=nsite-self.siteindices[-1]-1
        op_center=identity(1)
        for i in xrange(nl,nsite-nr):
            ous=self.query(i)
            if len(ous)==0:
                ou=one
            elif len(ous)==1:
                ou=ous[0].get_data()
            else:
                ou=reduce(lambda x,y:x.dot(y),ous)
            op_center=kron(op_center,ou)
        return kron(kron(identity(hndim**nl),op_center),identity(hndim**nr))

    def query(self,i):
        '''
        Query the specific opunits.

        Parameters:
            :i: int, the site index.

        Return:
            list, the <OpUnit>s.
        '''
        return [op for op in self.opunits if op.siteindex==i]

    def get_mathstr(self,factor=1.):
        '''Get the math string.'''
        factor=factor
        if abs(factor-1)<1e-15:
            factor_str=''
        else:
            factor_str=format(factor,'.3f').rstrip('0').rstrip('.')
        return ''.join([op.get_mathstr() for op in self.opunits])

    def trim_I(self):
        '''
        Remove Identity Operators.
        '''
        i=0
        while(i<len(self.opunits)):
            if isinstance(self.opunits[i],OpUnitI) and self.opunits[i].factor==1.:
                del(self.opunits[i])
            else:
                i=i+1

    def _insert1(self,ou,from_right=True,care_sign=True):
        '''
        Insert 1 Operator.

        Parameters:
            :ou: <OpUnit>,
            :from_right: bool, insert from right.
            :care_sign: bool, care ferminoic sign if True.
        '''
        j=ou.siteindex
        sites=self.siteindices
        pos=searchsorted(sites,j)
        if j in sites:
            self.opunits[pos]=(self.opunits[pos]*ou) if from_right else (ou*self.opunits[pos])
        else:
            self.opunits.insert(pos,ou)
        if care_sign and ou.fermionic:
            ou_crossed=self.opunits[pos+1:] if from_right else self.opunits[:pos]
            self*=prod([-1 if oui.fermionic else 1 for oui in ou_crossed])

    def _tonormalorder(self,care_sign=True):   #known bug, can not handle the commutation relation.
        '''
        Sort operator units by site indices using bubble sorting, 

            * multiplying <OpUnit>s with same site indices into one
            * sort opunits by siteindices.
        '''
        nou=self.nunit
        def push_op(self,i,j):
            '''push operator j to i'''
            oui,ouj=self.opunits[i],self.opunits[j]
            nf=0
            if ouj.fermionic and care_sign:
                #count fermionic operators in interval [i,j).
                nf=len([ou for ou in self.opunits[i:j] if ou.fermionic])
            self.opunits.insert(i,self.opunits.pop(j))
            if nf%2==1:
                self*=-1

        for i in xrange(nou-1):
            sites=self.siteindices
            sitei=sites[i]
            minpos=i
            for j in xrange(i+1,nou):
                if sites[j]<sites[minpos]:
                    minpos=j
            if minpos!=i:
                push_op(self,i,minpos)

        sites=self.siteindices
        unique_sites,indices=unique(sites,return_index=True)
        unique_sites.sort()
        if all(unique_sites==sites):  #already in compact form.
            return
        ous=[]
        for sitei in unique_sites:
            indices=where(sites==sitei)[0]
            if len(indices)>1:
                ous.append(prod([array(self.opunits)[indices]]))
            else:
                ous.append(self.opunits[indices[0]])
        self.opunits=ous
        return self

    def toMPO(self,nsite):
        '''
        Turn to MPO format with bond dimension 1.

        Parameters:
            :nsite: int, number of sites.

        Return:
            <MPO>,
        '''
        hndim=self.hndim
        OL=[identity(hndim).reshape([1,hndim,hndim,1])]*nsite
        for ou in self.opunits:
            OL[ou.siteindex]=ou.get_data().reshape([1,hndim,hndim,1])
        return MPO(OL)

class OpCollection(object):
    '''
    Addition of serie of operator strings/units.
    e.g. S(i)*S(j)+S(k)*S(l)+S(m)

    Attributes:
        :ops: list, a list of <OpString> instances.

    Readonly Attributes:
        :hndim: int, single site Hilbert space dimenion.
        :nop: number of constructing <OpUnit>s/<OpUnit>s.
    '''
    def __init__(self,ops=None):
        if ops is None:
            self.ops=[]
        else:
            self.ops=list(ops)

    def __str__(self):
        return '<OpCollection> '+'+'.join([op.__repr__() for op in self.ops])

    def __repr__(self):
        return '<OpCollection> %s opstring'%len(self.ops)

    def __copy__(self):
        newone=OpCollection(ops=[copy.copy(os) for os in self.ops])
        return newone

    def __neg__(self):
        return -1*self

    def __iter__(self):
        return self.ops.__iter__()

    def __getitem__(self,i):
        return self.ops[i]

    def __mul__(self,target):
        if isinstance(target,numbers.Number):
            if target==0:
                return 0
            nopc=OpCollection(self.ops)
            for op in nopc.ops:
                op*=target
            return nopc
        elif isinstance(target,(OpUnit,OpString)):
            return OpCollection([op*target for op in self.ops])
        elif isinstance(target,OpCollection):
            ops=array([[op*top for top in target.ops] for op in self.ops]).ravel()
            return OpCollection(ops)
        else:
            raise TypeError('Can not multiply OpCollection with %s'%target.__class__)

    def __rmul__(self,target):
        if isinstance(target,numbers.Number):
            return self.__mul__(target)
        elif isinstance(target,(OpUnit,OpString)):
            return OpCollection([target*op for op in self.ops])
        elif isinstance(target,OpCollection):
            return target.__mul__(self)
        else:
            raise TypeError('Can not multiply OpCollection with %s'%target.__class__)

    def __imul__(self,target):
        if isinstance(target,numbers.Number):
            if target==0:
                return 0
            for op in self.ops:
                op*=target
            return nop
        elif isinstance(target,(OpUnit,OpString)):
            return OpCollection([target*op for op in self.ops])
        elif isinstance(target,OpCollection):
            self.ops=array([[op*top for top in target.ops] for op in self.ops]).ravel()
            return self
        else:
            raise TypeError('Can not multiply OpCollection with %s'%target.__class__)

    def __add__(self,target):
        if isinstance(target,(OpUnit,OpString)):
            return OpCollection(self.ops+[target])
        elif isinstance(target,OpCollection):
            return OpCollection(self.ops+target.ops)
        elif target==0:
            return OpCollection(self.ops)
        else:
            raise TypeError('Can not add %s with %s.'%(self.__class__,target.__class__))

    def __radd__(self,target):
        if isinstance(target,(OpUnit,OpString)):
            return OpCollection([target]+self.ops)
        elif isinstance(target,OpCollection):
            return target.__add__(self)
        elif target==0:
            return OpCollection(self.ops)
        else:
            raise TypeError('Can not add %s with %s.'%(target.__class__,self.__class__))

    def __iadd__(self,target):
        if isinstance(target,(OpString,OpUnit)):
            self.ops.append(target)
        elif isinstance(target,OpCollection):
            self.ops.extend(target.ops)
        else:
            raise TypeError('Can not add %s with %s.'%(self.__class__,target.__class__))
        return self

    def __sub__(self,target):
        opc=OpCollection(self.ops)
        if isinstance(target,(OpString,OpUnit)):
            if target in opc.ops:
                opc.ops.remove(target)
            else:
                opc+=(-target)
        elif isinstance(target,OpCollection):
            for op in target.ops:
                opc-=op
        else:
            raise TypeError('Can not subtract %s with %s.'%(self.__class__,target.__class__))
        return opc

    def __rsub__(self,target):
        if isinstance(target,OpCollection):
            return target.__sub__(self)
        else:
            raise TypeError('Can not subtract %s with %s.'%(target.__class__,self.__class__))

    def __isub__(self,target):
        if isinstance(target,(OpString,OpUnit)):
            if target in self.ops:
                self.ops.remove(target)
            else:
                self+=(-target)
        elif isinstance(target,OpCollection):
            for op in target.ops:
                self-=op
        else:
            raise TypeError('Can not subtract %s with %s.'%(self.__class__,target.__class__))
        return self

    @property
    def nop(self):
        '''Number of <OpString>/<OpUnit> instances'''
        return len(self.ops)

    @property
    def hndim(self):
        '''The dimension of Hamiltonian.'''
        if len(self.ops)==0:
            return None
        else:
            return self.ops[0].hndim

    @property
    def maxsite(self):
        '''
        int, the maximum refered site.
        '''
        ms=-1
        for opi in self.ops:
            ms=max(ms,opi.maxsite)
        return ms

    def H(self,nsite=None):
        '''
        Get the matrix hamiltonian representation.

        Parameters:
            :nsite: int, number of sites.

        Return:
            matrix, the operator.
        '''
        if nsite is None: nsite=self.maxsite+1
        return sum([opi.H(nsite) for opi in self.ops],axis=0)

    def query(self,*indices):
        '''
        Query Operators linking sites i, j...

        Parameters:
            :indices: integer, the site indices.

        Return:
            list, a list of operator string.
        '''
        if len(indices)==0: return self.ops
        collection=[]
        if len(indices)==1:
            index=indices[0]
            for op in self.ops:
                if index in op.siteindices:
                    collection.append(op)
        else:
            for op in self.ops:
                s=set(op.siteindices)
                if s.issuperset(indices):
                    collection.append(op)
        return collection

    def filter(self,func):
        '''
        Query Operators meets condition function func.

        Parameters:
            :func: function, the condition function on site indices.

        Return:
            list, a list of operator string.
        '''
        return filter(lambda op:func(op.siteindices),self.ops)

    def toMPO(self,method='direct',nsite=None,bmg=None):
        '''
        Construct Matrix product Operator corresponding to this <OpCollection> instance.

        Parameters:
            :method: str, 'direct' or 'addition'.
                
                * direct: construct directly using complicated manipulation.
                * additive: construct by <MPO> addition operations term by term.
            :nsite: int, number of sites.
            :bmg: <BlockMarkerGenerator>,

        Return:
            <MPO>(<BMPO> if bmg is not None).
        '''
        if nsite is None:nsite=self.maxsite+1
        if method=='direct':
            nop=self.nop
            opmatrix=zeros((nop,nsite),dtype='O')
            for i,op in enumerate(self.ops):
                if isinstance(op,OpUnit):
                    opmatrix[i,op.siteindex]=op
                else:
                    opmatrix[i,asarray(op.siteindices)]=op.opunits

            constr=MPOConstructor(nsite=nsite,hndim=self.hndim)
            for op in self.ops:
                constr.read1(op)
            mpo=constr.compile()
            if bmg is not None: mpo=mpo.use_bm(bmg)
            return mpo
        else:
            mpo_collection=[op.toMPO(nsite=nsite) for op in self.ops]
            OL=[]
            for i in xrange(nsite):
                if i==0:
                    oi=concatenate([mpo.get(i) for mpo in mpo_collection],axis=3)
                elif i==nsite-1:
                    oi=concatenate([mpo.get(i) for mpo in mpo_collection],axis=0)
                else:
                    oi=fmerge_mpo(concatenate([mpo.get(i) for mpo in mpo_collection],axis=0))
                OL.append(oi)
            mpo=MPO(OL) if bmg is None else BMPO(OL,bmg=bmg)
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
        opunits=op.opunits
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
    def compress(self,niter=2,tol=1e-8,maxN=Inf,kernal='svd'):
        '''
        Move l-index by one with specific direction.
        
        Parameters:
            :niter: int, number of iteractions.
            :tol: float, the tolerence for compression.
            :maxN: int, the maximum dimension.
            :kernel: 'svd'/'ldu', the compressing kernel.

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
        #nsite=self.nsite
        #slabel1,slabel2,llabel=labels
        #for l,ai in enumerate(self.OL):
        #    ai.labels[0]='%s_%s'%(llabel,l)
        #    ai.labels[1]='%s_%s'%(slabel1,l)
        #    ai.labels[2]='%s_%s'%(slabel2,l)
        #    ai.labels[3]='%s_%s'%(llabel,l+1)

    def compress(self,niter=2,tol=1e-8,maxN=Inf,kernal='svd'):
        nsite=self.nsite
        hndim=self.hndim
        use_bm=hasattr(self,'bmg')
        llink_axis,rlink_axis,s1_axis,s2_axis=0,3,1,2
        acc=1.
        S=ones(1)
        for iit in xrange(niter):
            #decide sweep variables
            right=iit%2==0
            iterator=xrange(1,nsite) if right else xrange(nsite-2,0,-1)
            for l in iterator:
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
                U,S,V=AB.svd(cbond=3,cbond_str=cbond_str,bmg=self.bmg if hasattr(self,'bmg') else None,signs=[1,1,-1,-1,1,1])

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
                if iit==niter-1 and ((right and l==nsite-1) or (not right and l==1)):  #stop condition.
                    print 'Compression of MPO Done!'
                    U=U*S

                #set datas
                self.set(l-1,U)
                self.set(l,V)
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
        #nsite=self.nsite
        #slabel1,slabel2,llabel=labels
        #for l,ai in enumerate(self.OL):
        #    ai.labels=[ai.labels[0].chstr('%s_%s'%(llabel,l)),
        #    ai.labels[1].chstr('%s_%s'%(slabel1,l)),
        #    ai.labels[2].chstr('%s_%s'%(slabel2,l)),
        #    ai.labels[3].chstr('%s_%s'%(llabel,l+1))]

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

class OpUnitI(OpUnit):
    '''
    The single site Unitary operator.
    '''
    def __init__(self,hndim,siteindex=UNSETTLED):
        super(OpUnitI,self).__init__(label='I',data=identity(hndim),siteindex=siteindex,factor=1.,math_str='I')

    def __eq__(self,target):
        '''Note: site index is not considered as internal attribute!'''
        tol=1e-8
        if isinstance(target,numbers.Number):
            return self.factor==target
        else:
            return super(OpUnitI,self).__eq__(target)

    def __mul__(self,target):
        if isinstance(target,OpUnit) and target.siteindex==self.siteindex:
            return copy.copy(target)
        if isinstance(target,numbers.Number):
            res=copy.copy(self)
            res.factor*=target
            return res
        else:
            return super(OpUnitI,self).__mul__(target)

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
        if isinstance(op,OpString):
            op.trim_I()
    return opc


