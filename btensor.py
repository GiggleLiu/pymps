'''
Tensor Class
'''

import numpy as np
import numbers,copy,pdb
import numbers,itertools
from numpy import array
from numpy.linalg import norm
from scipy.linalg import svd
from abc import ABCMeta, abstractmethod

from tba.hgen import c2ind,SuperSpaceConfig,inherit_docstring_from
from blockmatrix import block_diag,SimpleBMG,join_bms,BlockMarker,trunc_bm
from tensor import TensorBase,BLabel,Tensor

__all__=['BTensor']

class BTensor(TensorBase):
    ''''
    Tensor with block markers.

    Attributes:
        :data: list, the datas.
        :labels: list, the labels.
        :blockmarkers: list, list of <BlockMarker>/None.
        :nzblocks: 2Darray, list of non-zero blocks, the entries are indices, not labels.
    '''
    def __init__(self,data,labels,blockmarkers,nzblocks):
        #data check
        #first, data size and nzblocks.
        assert(len(nzblocks)==len(data))
        size1=array([d.shape for d in data])
        size2=array([[bm.blocksize(blki,useqn=False) for bm,blki in zip(blockmarkers,blk)] for blk in nzblocks])
        assert(np.allclose(size1,size2))

        #setup data
        self.data=data
        self.labels=labels
        self.blockmarkers=blockmarkers
        self.nzblocks=array(nzblocks)

    @property
    def ndim(self):
        '''The dimension.'''
        return len(self.labels)

    @property
    def shape(self):
        '''Get the shape.'''
        return tuple(b.N for b in self.blockmarkers)

    @property
    def dtype(self):
        '''Get the data type'''
        return self.data[0].dtype

    @property
    def nnzblock(self):
        '''Get the number of nonzero blocks.'''
        return len(self.nzblocks)

    def __str__(self):
        return '<BTensor(%s)> %s'%(','.join(self.labels),' x '.join(['%s'%n for n in self.shape]))

    @inherit_docstring_from(TensorBase)
    def todense(self):
        if self.nnzblock==0:
            return np.zeros(self.shape,dtype=self.dtype)
        arr=np.zeros(self.shape,dtype=self.dtype)
        bms=self.blockmarkers
        for data,b in zip(self.data,self.nzblocks):
            arr[tuple(bmi.get_slice(bi,useqn=False) for bi,bmi in zip(b,bms))]=data
        res=Tensor(arr,labels=self.labels[:])
        return res

    @inherit_docstring_from(TensorBase)
    def mul_axis(self,vec,axis):
        if isinstance(axis,str):
            axis=self.labels.index(axis)
        t=self.make_copy(copydata=True)
        bm=self.blockmarkers[axis]
        nzblocks=t.nzblocks[:,axis]
        vec=vec.reshape([-1]+[1]*(self.ndim-axis-1))
        for dt,bl in zip(t.data,nzblocks):
            dt*=bm.extract_block(vec,ij=(bl,),axes=(0,),useqn=False)
        return t

    def make_copy(self,labels=None,copydata=True):
        '''
        Make a copy of this tensor.

        labels:
            The new labels.
        copydata:
            Copy the data to the new tensor.
        '''
        if labels is None:
            labels=self.labels[:]

        if copydata:
            data=copy.deepcopy(self.data)
        else:
            data=self.data[:]
        t=BTensor(data=data,labels=labels,blockmarkers=self.blockmarkers[:],nzblocks=self.nzblocks[...])
        return t

    def take(self,key,axis):
        '''
        Take subspace from this Tensor.

        key:
            The key, integer.
        axis:
            The axis to take.
        '''
        if isinstance(axis,str):
            axis=self.labels.index(axis)
        remaining_axes=range(axis)+range(axis+1,self.ndim)
        bm=self.blockmarkers[axis]
        bid,cid=bm.ind2b(key)
        mask=[bi==bid for bi in self.nzblocks[:,axis]]
        data=[di.take(cid,axis=axis) for mi,di in zip(mask,self.data) if mi]
        nzblocks=[bi for mi,bi in zip(mask,self.nzblocks[:,remaining_axes]) if mi]
        t=BTensor(data=data,labels=[self.labels[i] for i in remaining_axes],\
                blockmarkers=[self.blockmarkers[i] for i in remaining_axes],nzblocks=nzblocks)
        return t

    def query(self,block):
        '''
        Query data in specific block.

        Parameters:
            :block: tuple, the target block.

        Return:
            ndarray, the data.
        '''
        diffs=norm(self.nzblocks-block,axis=1)
        ind=np.argmin(diffs)
        if diffs[ind]<1e-10:
            return self.data[ind]
        else:
            return np.zeros([bm.blocksize(bi,useqn=False) for bm,bi in zip(self.blockmarkers,block)],dtype=self.dtype)

    def chorder(self,order):
        '''
        Reorder the indices of this tensor.

        order:
            The new order of the axes.
        '''
        assert(len(order))==self.ndim
        if isinstance(order[0],str):
            order=[self.labels.index(od) for od in order]
        data=[np.transpose(di,order) for di in self.data]
        t=BTensor(data,labels=list(array(self.labels)[order]),\
                blockmarkers=[self.blockmarkers[i] for i in order],nzblocks=self.nzblocks[:,order])
        return t

    def merge_axes(self,sls,nlabel=None):
        '''
        Merge multiple axes into one.

        Parameters:
            :sls: slice, axes range to merge.
            :nlabel: str/None, the new label, addition of old labels if None.

        Return:
            <TensorBase>
        '''
        #check for axes
        sorted_axes=np.mgrid[sls]

        #get new shape
        shape=self.shape
        newshape=shape[:sls.start]+(np.prod(shape[sls]),)+shape[sls.stop:]

        #get new labels
        labels=self.labels
        if nlabel is None:
            nlabel=''.join(labels[sls])
        newlabels=labels[:sls.start]+[nlabel]+labels[sls.stop:]

        #get new block markers.
        bm_mid,pm=join_bms(self.blockmarkers[sls])
        nblockmarkers=self.blockmarkers[:sls.start]+[bm_mid]+self.blockmarkers[sls.stop:]

        #get new data, data is reshaped.
        newdata=[]
        for dt in self.data:
            shape=dt.shape
            newshape=shape[:sls.start]+(np.prod(shape[sls]),)+shape[sls.stop:]
            newdata.append(dt.reshape(newshape))

        #merge nzblocks
        #join block i,j -> i*nblock(2) + j, which is c2ind.
        NL=[bm.nblock for bm in self.blockmarkers[sls]]
        newnzblocks=np.concatenate([self.nzblocks[:,:sls.start],c2ind(self.nzblocks[:,sls],\
                N=NL)[:,np.newaxis],self.nzblocks[:,sls.stop:]],axis=1)

        #generate the new tensor
        return BTensor(newdata,labels=newlabels,blockmarkers=nblockmarkers,nzblocks=newnzblocks)

    def split_axis(self,axis,dims,nlabels):
        '''
        Split one axis into multiple.

        Parameters:
            :axis: int/str, the axes to merge.
            :dims: tuple, the new dimensions, prod(dims)==self.shape[axis].
            :nlabels: list, the new labels.

        Return:
            <TensorBase>
        '''
        if isinstance(axis,str):
            axis=self.labels.index(axis)

        #get new shape
        shape=list(self.shape)
        newshape=shape[:axis]+list(dims)+shape[axis+1:]

        #get new labels
        newlabels=self.labels[:axis]+nlabels+self.labels[axis+1:]

        #generate the new tensor
        return Tensor(self.data.reshape(newshape),labels=newlabels)

    def sum(self,axis=None):
        '''
        sum over specific axis.

        Parameters:
            :axis: int/tuple/None, the axes/axis to perform sumation.

        Return:
            number/<BTensor>
        '''
        if axis is None:
            return sum([d.sum() for d in self.data])
        elif isinstance(axis,int):
            raise NotImplementedError()
        elif isinstance(axis,tuple):
            bt=self
            for ax in axis:
                bt=bt.sum(ax)
        else:
            raise TypeError()

    @inherit_docstring_from(TensorBase)
    def get_block(self):
        raise NotImplementedError()
