'''
Tensor Class
'''

import numpy as np
import numbers,copy,pdb
import numbers,itertools
import numbers
from numpy import array
from numpy.linalg import norm
from scipy.linalg import svd
from abc import ABCMeta, abstractmethod

from utils import inherit_docstring_from
from blockmatrix import block_diag,SimpleBMG,join_bms,BlockMarker,trunc_bm
from tensor import TensorBase,BLabel,Tensor

__all__=['BTensor']


class BTensor(TensorBase):
    ''''
    Tensor with block markers.

    Attributes:
        :data: dict, the data elements with qn_ids as keys.
        :labels: list, the labels.
    '''
    def __init__(self,data,labels):
        #setup data
        self.data=data
        self.labels=labels

    @property
    def ndim(self):
        '''The dimension.'''
        return len(self.labels)

    @property
    def shape(self):
        '''Get the shape.'''
        return tuple(lb.bm.N for lb in self.labels)

    @property
    def dtype(self):
        '''Get the data type'''
        if len(self.data)==0: return np.complex128
        return next(self.data.itervalues()).dtype

    @property
    def nnzblock(self):
        '''Get the number of nonzero blocks.'''
        return len(self.data)

    def __str__(self):
        s='<BTensor(%s)> %s'%(','.join(self.labels),' x '.join(['%s'%n for n in self.shape]))
        for blk,data in self.data.iteritems():
            s+='\n%s -> %s'%(blk,'x'.join(str(x) for x in data.shape))
        return s

    def __mul__(self,target):
        if isinstance(target,BTensor):
            #1. get remaining axes - (raxes1, raxes2) and contracted axes - (caxes1, caxes2).
            lb1s,lb2s=self.labels,target.labels
            caxes1,caxes2,raxes1=[],[],[]
            for i1,lb1 in enumerate(lb1s):
                if lb1 in lb2s:
                    caxes1.append(i1)
                    caxes2.append(lb2s.index(lb1))
                else:
                    raxes1.append(i1)
            raxes2=[i2 for i2 in xrange(len(lb2s)) if i2 not in caxes2]

            #2. set entries
            ndata={}
            for bi,datai in self.data.iteritems():
                for bj,dataj in target.data.iteritems():
                    if tuple(bi[ax] for ax in caxes1)==tuple(bj[ax] for ax in caxes2):
                        tblk=tuple(bi[ax] for ax in raxes1)+tuple(bj[ax] for ax in raxes2)
                        ndata[tblk]=ndata.get(tblk,0)+np.tensordot(datai,dataj,axes=(caxes1,caxes2))
            return BTensor(ndata,labels=[lb1s[ax] for ax in raxes1]+[lb2s[ax] for ax in raxes2])
        elif isinstance(target,numbers.Number):
            return BTensor(dict((blk,target*data) for blk,data in self.data.iteritems()),self.labels[:])
        else:
            raise TypeError

    def __rmul__(self,target):
        if isinstance(target,numbers.Number):
            return self.__mul__(target)
        elif isinstance(target,BTensor):
            return target.__mul__(self)
        else:
            raise TypeError

    def __imul__(self,target):
        if isinstance(target,numbers.Number):
            for data in self.data.values():
                data*=target
        else:
            raise TypeError

    def __div__(self,target):
        if isinstance(target,numbers.Number):
            return BTensor(dict((blk,data/target) for blk,data in self.data.iteritems()),self.labels[:])
        else:
            raise TypeError

    def __idiv__(self,target):
        if isinstance(target,numbers.Number):
            for data in self.data.values():
                data/=target
        else:
            raise TypeError

    @inherit_docstring_from(TensorBase)
    def todense(self):
        if self.nnzblock==0:
            return np.zeros(self.shape,dtype=self.dtype)
        arr=np.zeros(self.shape,dtype=self.dtype)
        bms=[lb.bm for lb in self.labels]
        for q,data in self.data.iteritems():
            arr[tuple(bmi.get_slice(qi) for qi,bmi in zip(q,bms))]=data
        res=Tensor(arr,labels=self.labels[:])
        return res

    @inherit_docstring_from(TensorBase)
    def mul_axis(self,vec,axis):
        if isinstance(axis,str):
            axis=self.labels.index(axis)
        if axis<0: axis+=self.ndim
        t=self.make_copy(copydata=False)
        bm=self.labels[axis].bm
        vec=vec.reshape([-1]+[1]*(self.ndim-axis-1))
        for k,data in t.data.iteritems():
            t.data[k]=data*bm.extract_block(vec,ij=(k[axis],),axes=(0,))
        return t

    @inherit_docstring_from(TensorBase)
    def make_copy(self,labels=None,copydata=True):
        if labels is None:
            labels=self.labels[:]

        if copydata:
            data=dict((x[:],y[...]) for x,y in self.data.iteritems())
        else:
            data=dict(self.data)
        t=BTensor(data=data,labels=labels)
        return t

    @inherit_docstring_from(TensorBase)
    def take(self,key,axis,useqn=False):
        if isinstance(axis,str):
            axis=self.labels.index(axis)

        #regenerate the labels,
        labels=self.labels[:]
        if np.ndim(key)==0:
            #0d case, delete a dimension.
            lb=labels.pop(axis)
            datas={}
            bind,cind=lb.bm.ind2b(key)
            for bi,data in self.data.iteritems():
                if bi[axis]==bind:
                    datas[bi[:axis]+bi[axis+1:]]=data.take(cind,axis=axis)
            t=BTensor(data=datas,labels=labels)
            return t
        elif np.ndim(key)==1:
            if useqn:
                key=mgrid(lb.bm.get_slice(lb.bm.index_qn(key).item()))
            if hasattr(self.labels[axis],'bm'):
                #1d case, shrink one dimension.
                bm=self.labels[axis].bm
                if key.dtype=='bool':
                    key=np.where(key)[0]
                #inflate and take the desired dimensions
                bm_infl=bm.inflate()
                qns=bm_infl.qns[key]
                nbm=BlockMarker(qns=qns,Nr=np.arange(len(qns)+1))
                labels[axis]=labels[axis].chbm(nbm)
                #get data
                remaining_axes=range(axis)+range(axis+1,self.ndim)
                bid,cid=bm.ind2b(key)
                datas=[]
                for i,(k,bidi,cidi) in enumerate(zip(key,bid,cid)):
                    for bi,data in self.data.iteritems():
                        if bi[axis]==bidi:
                            datas[bi[:axis]+(i,)+bi[axis+1:]]=data.take(range(cidi,cidi+1),axis=axis)
                t=BTensor(data=datas,labels=labels)
                return t
        else:
            raise ValueError


    @inherit_docstring_from(TensorBase)
    def chorder(self,order):
        assert(len(order))==self.ndim
        if isinstance(order[0],str):
            order=[self.labels.index(od) for od in order]
        data=dict((tuple(bi[i] for i in order),np.transpose(di,order)) for bi,di in self.data.iteritems())
        t=BTensor(data,labels=[self.labels[i] for i in order])
        return t

    @inherit_docstring_from(TensorBase)
    def merge_axes(self,sls,nlabel=None,signs=None,bmg=None):
        axes=np.mgrid[sls]
        labels=self.labels
        #get new labels
        if nlabel is None:
            nlabel=''.join([labels[i] for i in axes])

        #get new block markers.
        labels=self.labels
        bms=[labels[ax].bm for ax in axes]
        if bmg is None:
            bm_mid=join_bms(bms,signs=signs)  #natural join that keep last dimension not expanded.
        else:
            bm_mid=bmg.join_bms(bms,signs=signs)
        nlabel=BLabel(nlabel,bm_mid)
        newlabels=labels[:sls.start]+[nlabel]+labels[sls.stop:]

        #mapping bms -> bmid
        nbs=[bm.N for bm in bms]; nbs[-1]=bms[-1].nblock
        rbs=np.cumprod(nbs[::-1])[::-1][1:]
        ndata={}
        for b0,data in self.data.iteritems():
            sl=[bm.get_slice(b0i) for bm,b0i in zip(bms,b0[sls])]
            for si in itertools.product(*[range(s.start,s.stop) for s in sl[:-1]]):
                bmid=sum(rbs*si)+b0[sls.stop-1]
                nb=b0[:sls.start]+(bmid,)+b0[sls.stop:]
                ndata[tuple(nb)]=data[tuple(slice(None) for i in xrange(sls.start))+tuple(bmi.ind2b(sx)[1] for sx,bmi in zip(si,bms))]

        #generate the new tensor
        return BTensor(ndata,labels=newlabels)

    @inherit_docstring_from(TensorBase)
    def split_axis(self,axis,dims,nlabels):
        raise NotImplementedError

    def sum(self,axis=None):
        '''
        sum over specific axis.

        Parameters:
            :axis: int/tuple/None, the axes/axis to perform sumation.

        Return:
            number/<BTensor>
        '''
        if axis is None:
            return sum([d.sum() for d in self.data.values()])
        elif isinstance(axis,int):
            #0d case, delete a dimension.
            lb=labels.pop(axis)
            datas={}
            for bi,data in self.data.iteritems():
                if bi[axis]==bid:
                    datas[bi[:axis]+bi[axis+1:]]=data.sum(axis=axis)
            t=BTensor(data=datas,labels=labels)
            return t
        elif isinstance(axis,tuple):
            bt=self
            for ax in axis:
                bt=bt.sum(ax)
        else:
            raise TypeError()

    @inherit_docstring_from(TensorBase)
    def get_block(self,block):
        return self.data[block]
