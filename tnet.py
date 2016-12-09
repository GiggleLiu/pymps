'''
Tensor NetWork.
'''

from numpy import *
from scipy.linalg import svd,qr,rq,norm,block_diag
from scipy import sparse as sps
import cPickle as pickle
import pdb,time,copy,warnings,numbers
from abc import ABCMeta, abstractmethod
from profilehooks import profile
from itertools import combinations

from utils import inherit_docstring_from
from blockmatrix import BlockMarker
import tensor
from sa import sap,anneal,SAP

__all__=['TNet','find_optcontract','TNCSAP']

ZERO_REF=1e-12

class Connection(object):
    '''
    Connection
    
    Attributes:
        :link2leg: 2d array.
        :leg2link: 1d array.
        :SL: list, link weights.
    '''
    def __init__(self,leg2link,SL=None):
        self.leg2link=asarray(leg2link)
        nlink=self.leg2link.max()+1
        self.link2leg=zeros([nlink,2],dtype='int32')
        for i in xrange(nlink):
            self.link2leg[i]=where(self.leg2link==i)[0]
        if SL is None:
            SL=[None]*self.nlink
        self.SL=SL

    @property
    def nlink(self): return self.link2leg.shape[0]

    @property
    def nleg(self): return self.leg2link.shape[0]

    @property
    def pairmask(self): return self.leg2link>=0

    def whichlink(self,legid):
        '''find link index from leg index.'''
        return self.leg2link[legid]

    def whichlegs(self,linkid):
        '''find leg indices from link index.'''
        return self.link2leg[linkid]

def _join_tensors(tensor1,tensor2):
    #NO: tensor with self loop.
    leg1,dim1=tensor1.labels,tensor1.shape
    leg2,dim2=tensor2.labels,tensor2.shape
    legs,dims=leg1+leg2,dim1+dim2
    labels,shape=[],[]
    dcomp=1
    for i,(leg,dim) in enumerate(zip(legs,dims)):
        if leg in labels:
            n=labels.index(leg)
            labels.pop(n)
            shape.pop(n)
        else:
            labels.append(leg)
            shape.append(dim)
            dcomp*=dim
    return dcomp,CoTensor(labels,tuple(shape))

class TNet(object):
    '''
    Tensor Network, the geometry is constant but dimensions are flexible.

    Attributes:
        :tensors: list, items are tensors.
        :tensorptr: 1d array, pointer for tensor to get labels and dimensions.
        :legs: 1d array, legs for tensors.

    Read Only Attributes:
        :dims: 1d array, dimensions of tensors in and array.
        :ntensor: int, # of tensors.
        :nleg: int, # of legs.
    '''
    def __init__(self,tensors):
        self.tensors=tensors
        lbs=reduce(lambda x,y:x+y,[t.labels for t in tensors])
        self.legs=asarray(lbs)
        nr=[ndim(t) for t in self.tensors]
        self.tensorptr=append([0],cumsum(nr))  #pointer to dims and lbs

    @property
    def dims(self):
        '''Dimensions of tensors in an array.'''
        dims=reduce(lambda x,y:x+y,[t.shape for t in self.tensors])
        return dims

    @property
    def ntensor(self):
        return len(self.tensors)

    @property
    def nleg(self):
        return self.tensorptr[-1]

    def lid2tid(self,lid):
        '''Parse leg index to tensor index.'''
        tt=searchsorted(self.tensorptr,lid+1)-1
        ti=lid-self.tensorptr[tt]
        return tt,ti

    def tid2lid(self,tid):
        '''Parse leg index to tensor index.'''
        return self.tensorptr[tid[0]]+tid[1]

    def complexity(self,order,return_info=False):
        '''
        Calculate the complexity of the contraction order.

        Parameters:
            :order: list of len2-tuple, the combinational order.

        Return:
            (int,tuple), complexity and informations.
        '''
        dcomps=[]
        tensors=self.tensors[:]
        tensors_histo=[tensors[:]]
        for i,j in order:
            dcomp,tensor=_join_tensors(tensors[i],tensors[j])
            dcomps.append(dcomp)
            tensors[i]=tensor
            tensors.pop(j)
            if return_info:
                tensors_histo.append(tensors[:])
        if return_info:
            return sum(dcomps),tensors_histo
        else:
            return sum(dcomps)

    def contract(self,order):
        '''Contract tensors by combinational order.'''
        if self.ntensor==0: return 1.
        tensors=self.tensors[:]
        for ti,tj in order:
            ntensor=tensors[ti]*tensors[tj]
            tensors[ti]=ntensor
            tensors.pop(tj)
        return ntensor

    def get_connection(self):
        '''
        Construct Connection from information of tensors.
        '''
        #detect same labels
        mask=-ones(self.nleg,dtype='int32')
        lbs=list(self.legs)
        j=0
        for i1,lb1 in enumerate(lbs):
            if mask[i1]!=-1: continue
            lbs2=lbs[i1+1:]
            if lb1 in lbs2:
                i2=lbs2.index(lb1)
                mask[i2+i1+1]=j
                mask[i1]=j
                j=j+1
        return Connection(mask,None)


############################### Contraction #############################

class CoTensor(object):
    '''Fake tensor with out data.'''
    def __init__(self,labels,shape):
        self.labels=labels
        self.shape=shape

class TNCSAP(SAP):
    '''
    Tensor Network Contraction Simulated Annealing Problem

    Notations:

        * state, list, the order of contraction.
        * proposal, (int, list), (permute the position of i, i+1, history of contraction)
        * cost, int, the # of floops for specific order of contraction.
    '''
    def __init__(self,tnet):
        self.tnet=tnet
        ntensor=tnet.ntensor
        self.combs=[]
        for i in xrange(ntensor,1,-1):
            self.combs.append(array(list(combinations(xrange(i),2)),dtype='int32'))
        self.Nr=append([0],cumsum([len(c)-1 for c in self.combs]))-1

    def _random_step(self):
        i=random.randint(self.Nr[-1]+1)
        res=searchsorted(self.Nr,i)-1
        return res

    def _decode(self,order):
        res=[comb[oi] for comb,oi in zip(self.combs,order)]
        return res

    @inherit_docstring_from(SAP)
    def get_cost(self,state):
        cost=self.tnet.complexity(self._decode(state))
        return cost

    @inherit_docstring_from(SAP)
    def propose(self,state,cost):
        order=state
        istep=self._random_step()
        N=self.tnet.ntensor-istep
        norder=order[:]
        norder[istep]=(norder[istep]+1)%(N*(N-1)/2)
        ncomp=self.tnet.complexity(self._decode(norder))
        dcomp=ncomp-cost
        return (istep,norder),dcomp

    @inherit_docstring_from(SAP)
    def accept(self,proposal,state):
        (istep,norder),dcomp=proposal
        return norder

    @inherit_docstring_from(SAP)
    def get_random_state(self):
        order=array([random.randint(N*(N-1)/2) for N in xrange(self.tnet.ntensor,1,-1)])  #upto N=ntensor,...,2
        return order

def find_optcontract(tnet,maxstep=1e6,minstep=1.,ntemp=50,nswap=80,nrun=10):
    '''
    Find optimal contraction using Simulated Annealing.

    Parameters:
        :tnet: <TNet>,
        :maxstep, minstep: int, the maximum and minimum 'temperature'
        :ntemp: int, number of temperature slices.
        :nswap: int/None, number of swap operations to get the optimal sequence, None for (# of tensor)^2.
        :nrun: int, number of runs.

    Return:
        (flops,order).
    '''
    cap=TNCSAP(tnet)
    flops,state=anneal(cap,tempscales=exp(linspace(log(maxstep),log(minstep),ntemp)),nms=nswap,nrun=nrun)
    return flops,cap._decode(state)
