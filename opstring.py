'''
Operator Unit, String and Collection.
'''

from numpy import *
from scipy import sparse as sps
import copy,numbers

__all__=['OpUnit','OpString','OpCollection','OpUnitI']
UNSETTLED='-'

def _format_factor(num):
    if abs(num-1)<1e-5: return ''
    if abs(imag(num))<1e-5: num=real(num)
    if imag(num)==0 and abs(fmod(real(num),1))<1e-5: num=int(num)
    res='%s*'%around(num,decimals=3)
    return res

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
                return OpUnit(label='%s%s+%s%s'%(_format_factor(self.factor),self.label,_format_factor(target.factor),target.label),\
                        data=self.factor*self.data+target.factor*target.data,siteindex=self.siteindex,\
                        math_str='%s%s+%s%s'%(_format_factor(self.factor),self.__math_str__,_format_factor(target.factor),target.__math_str__),fermionic=self.fermionic)
            else:
                return OpCollection([self,target])
        elif isinstance(target,OpString):
            if target.nunit==1:
                return self.__add__(target.opunits[0])
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
        elif isinstance(target,OpString):
            if self.nunit==1 and target.nunit==1:
                return self.opunits[0]+target.opunits[1]
            else:
                return OpCollection([self,target])
        elif isinstance(target,OpUnit):
            if self.nunit==1:
                return self.opunits[0]+target
            else:
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
        if isinstance(target,OpString):
            if target.nunit==1:
                return self.__add__(target.opunits[0])
            return OpCollection(self.ops+[target])
        elif isinstance(target,OpUnit):
            ops_new=self.ops[:]
            merge=False
            for i,opi in enumerate(ops_new):
                if isinstance(opi,OpUnit) and opi.siteindex==target.siteindex:
                    merge=True
                    ops_new[i]=opi+target
                    break
            if not merge:
                ops_new.append(target)
            return OpCollection(ops_new)
        elif isinstance(target,OpCollection):
            return OpCollection(self.ops+target.ops)
        elif target==0:
            return OpCollection(self.ops)
        else:
            raise TypeError('Can not add %s with %s.'%(self.__class__,target.__class__))

    def __radd__(self,target):
        if isinstance(target,OpString):
            if target.nunit==1:
                return self.__radd__(target.opunits[0])
            return OpCollection([target]+self.ops)
        elif isinstance(target,OpUnit):
            ops_new=self.ops[:]
            merge=False
            for i,opi in enumerate(ops_new):
                if isinstance(opi,OpUnit) and opi.siteindex==target.siteindex:
                    merge=True
                    ops_new[i]=target+opi
                    break
            if not merge:
                ops_new.append(target)
            return OpCollection(ops_new)
        elif isinstance(target,OpCollection):
            return target.__add__(self)
        elif target==0:
            return OpCollection(self.ops)
        else:
            raise TypeError('Can not add %s with %s.'%(target.__class__,self.__class__))

    def __iadd__(self,target):
        if isinstance(target,OpString):
            if target.nunit==1:
                return self.__iadd__(target.opunits[0])
            self.ops.append(target)
        elif isinstance(target,OpUnit):
            ops_new=self.ops
            merge=False
            for i,opi in enumerate(ops_new):
                if isinstance(opi,OpUnit) and opi.siteindex==target.siteindex:
                    merge=True
                    ops_new[i]=target+opi
                if not merge:
                    ops_new.append(target)
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
