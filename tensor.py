'''
TensorBase and dense Tensor Class.
'''

import numpy as np
from numpy import array
from abc import ABCMeta, abstractmethod
import copy,pdb,numbers,itertools

from utils import inherit_docstring_from
from blockmatrix import join_bms,BlockMarker

__all__=['TensorBase','Tensor','tdot','BLabel']

class BLabel(str):
    '''
    Label string with block marker.

    Attibutes:
        :bm: <BlockMarker>,
    '''
    def __new__(cls,value,bm):
        obj=str.__new__(cls,value)
        obj.bm=bm
        return obj

    def __getnewargs__(self):
        return (self.__class__,self.bm)

    def __copy__(self):
        return BLabel(self,self.bm)

    def __deepcopy__(self, memo):
        return BLabel(self,copy.deepcopy(self.bm))

    def chbm(self,bm):
        '''Get a new <BLabel> with different block marker.'''
        return BLabel(str(self),bm)

    def chstr(self,s):
        '''Get a new <BLabel> with different string.'''
        return BLabel(s,self.bm)

def tdot(tensor1,tensor2):
    '''
    Tensor dot between two tensors, faster than contract in most case?

    Parameters:
        :tensor1,tensor2: <Tensor>, two tensors.

    Return:
        <Tensor>,
    '''
    #detect same labels
    lb1s,lb2s=tensor1.labels,tensor2.labels[:]
    axes1,axes2=[],[]
    nlb=[]
    for i1,lb1 in enumerate(lb1s):
        if lb1 in lb2s:
            i2=lb2s.index(lb1)
            axes1.append(i1)
            axes2.append(i2)
        else:
            nlb.append(lb1)
    nlb=nlb+[lb2s[i2] for i2 in xrange(len(lb2s)) if i2 not in axes2]
    res=np.tensordot(tensor1,tensor2,axes=(axes1,axes2))
    res=Tensor(res,labels=nlb)
    floops=np.sqrt(np.prod(tensor1.shape)*np.prod(tensor2.shape)*np.prod(res.shape))
    return res

class TensorBase(object):
    '''
    The base abstract class for tensor.
    '''

    __metaclass__ = ABCMeta

    @property
    def ndim(self):
        '''The dimension.'''
        pass

    @abstractmethod
    def todense(self):
        '''Parse this tensor to dense version - <Tensor>.'''
        pass

    @abstractmethod
    def mul_axis(self,vec,axis):
        '''
        Multiply a vector on specific axis.

        Parameters:
            :vec: 1d array, the vector.
            :axis: int/str, the axis or label.

        Return:
            <TensorBase>
        '''
        pass

    @abstractmethod
    def make_copy(self,labels=None,copydata=True):
        '''
        Make a copy of this tensor.

        Parameters:
            :labels: list/None, the new labels, use the old ones if None.
            :copydata: bool, copy the data to the new tensor if True.

        Return:
            <TensorBase>
        '''
        pass

    @abstractmethod
    def take(self,key,axis):
        '''
        Take subspace from this Tensor.

        Parameters:
            :key: 0-d/1-d array, the key
            :axis: int, the axis to take.

        Return:
            <TensorBase>
        '''
        pass

    @abstractmethod
    def chorder(self,order):
        '''
        Reorder the axes of this tensor.

        Parameters:
            :order: tuple, the new order of the axes.

        Return:
            <TensorBase>
        '''
        pass

    @abstractmethod
    def merge_axes(self,sls,nlabel=None,signs=None,bmg=None):
        '''
        Merge multiple axes into one.

        Parameters:
            :sls: slice, the axes to merge.
            :nlabel: str/None, the new label, addition of old labels if None.
            :signs: 1darray, the signs(flow direction) for each merged block marker.

        Return:
            <TensorBase>
        '''
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_block(self,block):
        '''
        Query data in specific block.

        Parameters:
            :block: tuple, the target block.

        Return:
            ndarray, the data.
        '''
        pass

    def chlabel(self,axis,nlabel):
        '''
        Change the label in place.

        Parameters:
            :axis: int/str, the axis/label to change.
            :nlabel: str, the new label.

        Return:
            self
        '''
        labels=self.labels[:]
        if isinstance(axis,str):
            axis=labels.index(axis)
        if isinstance(axis,int):
            labels[axis]=nlabel
            self.labels=labels
            return self
        else:
            raise TypeError('Wrong type for axis indicator: %s.'%axis.__class__)

class Tensor(np.ndarray,TensorBase):
    '''
    Tensor class subclassing ndarray, with each dimension labeled by a string(or BLabel).

    Construct:
        Tensor(shape,labels,**kwargs):
            Create a <Tensor> with random data with specified shape.
        Tensor(array,labels,**kwargs):
            Create a <Tensor> converted from array.

    Attributes:
        :labels: list, the labels of axes.

        *refer numpy.ndarray for more details.*
    '''
    __array_priority__=0  #if it is >0, __str__ will not work.
    def __new__(subtype,param,labels,*args,**kwargs):
        if isinstance(param,tuple):
            dim=len(param)
            if dim!=len(labels):
                raise ValueError('Inconsistant number of dimension(%s) and labels(%s).'%(dim,len(labels)))
            obj=np.ndarray.__new__(subtype,shape=param,*args,**kwargs)
        elif isinstance(param,subtype):
            obj=param
        else:
            dim=np.ndim(param)
            if dim!=len(labels):
                raise ValueError('Inconsistant number of dimension(%s) and labels(%s).'%(dim,len(labels)))
            obj=np.asarray(param,*args,**kwargs).view(subtype)
        obj.labels=labels
        return obj

    def __array_finalize__(self,obj):
        if obj is None:
            return
        self.labels=obj.labels[:] if hasattr(obj,'labels') else None

    def __str__(self):
        return '<Tensor(%s)>\n%s'%(','.join([lb+'*' if isinstance(lb,BLabel) else lb for lb in self.labels]),super(Tensor,self).__str__())

    def __repr__(self):
        return '<Tensor(%s)>'%(','.join([lb+'*' if isinstance(lb,BLabel) else lb for lb in self.labels]),)

    def __mul__(self,target):
        if isinstance(target,Tensor):
            return tdot(self,target)
        else:
            res=super(Tensor,self).__mul__(target)
            return res

    def __rmul__(self,target):
        if isinstance(target,Tensor):
            return target.__mul__(self)
        else:
            return super(Tensor,self).__rmul__(target)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state=super(Tensor,self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state=pickled_state[2]+(self.labels,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.labels=state[-1]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(Tensor,self).__setstate__(state[0:-1])

    def conj(self):
        res=Tensor(np.asarray(self).conj(),labels=self.labels[:])
        return res

    @inherit_docstring_from(TensorBase)
    def make_copy(self,labels=None,copydata=True):
        if labels is not None:
            assert(len(labels)==len(self.shape))
        else:
            labels=self.labels[:]
        if copydata:
            res=copy.deepcopy(self)
            res.labels=labels
            return res
        else:
            res=self.view(Tensor)
            res.labels=labels
            return res

    @inherit_docstring_from(TensorBase)
    def todense(self):
        return self.make_copy()

    @inherit_docstring_from(TensorBase)
    def take(self,key,axis,useqn=False):
        if isinstance(axis,str):
            axis=self.labels.index(axis)

        #regenerate the labels,
        labels=self.labels[:]
        if np.ndim(key)==0:# or (useqn and np.ndim(key)==1):
            #0d case, delete a dimension.
            lb=labels.pop(axis)
        elif np.ndim(key)==1:
            if useqn:
                key=mgrid(lb.bm.get_slice(lb.bm.index_qn(key).item()))
            if hasattr(self.labels[axis],'bm'):
                #1d case, shrink a dimension.
                bm=self.labels[axis].bm
                if key.dtype=='bool':
                    key=np.where(key)[0]
                #inflate and take the desired dimensions
                bm_infl=bm.inflate()
                qns=bm_infl.qns[key]
                bm=BlockMarker(qns=qns,Nr=np.arange(len(qns)+1))
                labels[axis]=labels[axis].chbm(bm)
        else:
            raise ValueError
        ts=super(Tensor,self).take(key,axis=axis)
        return Tensor(ts,labels=labels)

    @inherit_docstring_from(TensorBase)
    def chorder(self,order):
        assert(len(order))==np.ndim(self)
        if isinstance(order[0],str):
            order=[self.labels.index(od) for od in order]
        t=Tensor(np.transpose(self,order),labels=[self.labels[i] for i in order])
        return t

    @inherit_docstring_from(TensorBase)
    def mul_axis(self,vec,axis):
        if isinstance(axis,str):
            axis=self.labels.index(axis)
        if axis<0: axis+=self.ndim
        if isinstance(axis,int):
            vec=np.asarray(vec).reshape([-1]+[1]*(self.ndim-axis-1))
            res=vec*self
            return res
        else:
            raise TypeError('Wrong type for axis indicator: %s.'%axis.__class__)

    @inherit_docstring_from(TensorBase)
    def merge_axes(self,sls,nlabel=None,signs=None,bmg=None):
        axes=np.mgrid[sls]
        labels=self.labels
        #get new shape
        shape=array(self.shape)
        newshape=list(shape[:sls.start])+[np.prod(shape[sls])]+list(shape[sls.stop:])
        ts=np.asarray(self.reshape(newshape))

        #get new labels
        if nlabel is None:
            nlabel=''.join([labels[i] for i in axes])

        #get new block markers.
        if all([isinstance(labels[i],BLabel) for i in axes]):
            labels=self.labels
            if bmg is None:
                bm_mid=join_bms([labels[ax].bm for ax in axes],signs=signs)
            else:
                bm_mid=bmg.join_bms([labels[ax].bm for ax in axes],signs=signs)
            nlabel=BLabel(nlabel,bm_mid)
            #ts=ts.take(pm,axis=sls.start)
        newlabels=labels[:sls.start]+[nlabel]+labels[sls.stop:]

        #generate the new tensor
        return Tensor(ts,labels=newlabels)

    @inherit_docstring_from(TensorBase)
    def split_axis(self,axis,dims,nlabels):
        if isinstance(axis,str):
            axis=self.labels.index(axis)
        #get new shape
        shape=list(self.shape)
        newshape=shape[:axis]+list(dims)+shape[axis+1:]
        #get new labels
        newlabels=self.labels[:axis]+nlabels+self.labels[axis+1:]
        #generate the new tensor
        return Tensor(self.data.reshape(newshape),labels=newlabels)

    @inherit_docstring_from(TensorBase)
    def get_block(self,block):
        if not isinstance(self.labels[0],BLabel):
            raise Exception('This tensor is not blocked!')
        return self[tuple([lb.bm.get_slice(b) for b,lb in zip(block,self.labels)])]

    def tobtensor(self,bms=None):
        '''
        Parse to <BTensor>.

        Return:
            <BTensor>,
        '''
        data={}
        if bms is None: bms=[l.bm for l in self.labels]
        #detect and extract datas.
        for blk in itertools.product(*[range(bm.nblock) for bm in bms]):
            datai=self[tuple(bm.get_slice(i) for bm,i in zip(bms,blk))]
            if not np.allclose(datai,0):
                data[blk]=datai
        from btensor import BTensor
        return BTensor(data,self.labels[:])

    def b_reorder(self,axes=None,return_pm=False):
        '''
        Reorder rows, columns to make tensor blocked.

        Parameters:
            :axes: tuple, the target axes to be reorderd
            :return_pm: bool, return the permutation series.

        Return:
            <Tensor>.
        '''
        ts=self
        pms=[]
        labels=self.labels[:]
        if axes is None: axes=range(np.ndim(self))
        for i in axes:
            bm_new,pm=labels[i].bm.compact_form()
            labels[i]=labels[i].chbm(bm_new)
            if not np.allclose(pm,np.arange(len(pm))):
                ts=super(Tensor,ts).take(pm,axis=i)
            pms.append(pm)
        ts.labels=labels
        if return_pm:
            return ts,pms
        else:
            return ts

