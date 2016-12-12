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

from tba.lattice import ind2c
from utils import inherit_docstring_from
from blockmatrix import block_diag,SimpleBMG,join_bms,BlockMarker,trunc_bm
from tensor import TensorBase,BLabel,Tensor,ZERO_REF

__all__=['BTensor']


class BTensor(TensorBase):
    ''''
    Tensor with block markers.

    Attributes:
        :data: dict, the data elements with qn_ids as keys.
        :labels: list, the labels.
    '''
    def __init__(self,data,labels):
        #check data
        if not all(hasattr(lb,'bm') for lb in labels): raise TypeError
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

    def __abs__(self):
        return BTensor(dict((key,abs(data)) for key,data in self.data.iteritems()),self.labels[:])

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
                        val=np.tensordot(datai,dataj,axes=(caxes1,caxes2))
                        ndata[tblk]=ndata[tblk]+val if ndata.has_key(tblk) else val
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

    def __pow__(self,target):
        return BTensor(dict((key,data**target) for key,data in self.data.iteritems()),self.labels[:])

    @inherit_docstring_from(TensorBase)
    def todense(self):
        res=Tensor(self.toarray(),labels=self.labels[:])
        return res

    @inherit_docstring_from(TensorBase)
    def toarray(self):
        if self.nnzblock==0:
            return np.zeros(self.shape,dtype=self.dtype)
        arr=np.zeros(self.shape,dtype=self.dtype)
        bms=[lb.bm for lb in self.labels]
        for q,data in self.data.iteritems():
            arr[tuple(bmi.get_slice(qi) for qi,bmi in zip(q,bms))]=data
        return arr

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
    def take(self,key,axis):
        if isinstance(axis,str):
            axis=self.labels.index(axis)
        if axis<0: axis+=self.ndim

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
            key=np.asarray(key)
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
            bid,cid=bm.ind2b(key)
            datas={}
            for i,(k,bidi,cidi) in enumerate(zip(key,bid,cid)):
                for bi,data in self.data.iteritems():
                    if bi[axis]==bidi:
                        datas[bi[:axis]+(i,)+bi[axis+1:]]=data.take(range(cidi,cidi+1),axis=axis)
            t=BTensor(data=datas,labels=labels)
            return t
        else:
            raise ValueError

    def take_b(self,key,axis):
        if isinstance(axis,str):
            axis=self.labels.index(axis)
        if axis<0: axis+=self.ndim

        #regenerate the labels,
        labels=self.labels[:]
        if np.ndim(key)==0:
            key=[key]
        elif np.ndim(key)>1:
            raise ValueError
        #1d case, shrink one dimension.
        bm=self.labels[axis].bm
        if isinstance(key,np.ndarray) and key.dtype=='bool':
            key=np.where(key)[0]
        #change block marker
        bm=self.labels[axis].bm
        qns,nr=bm.qns[key],bm.nr[key]
        nbm=BlockMarker(qns=qns,Nr=np.append([0],np.cumsum(nr)))
        labels[axis]=labels[axis].chbm(nbm)

        #get data
        datas={}
        for i,k in enumerate(key):
            for bi,data in self.data.iteritems():
                if bi[axis]==k:
                    datas[bi[:axis]+(i,)+bi[axis+1:]]=data
        t=BTensor(data=datas,labels=labels)
        return t

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
    def split_axis(self,axis,nlabels):
        if isinstance(axis,str):
            axis=self.labels.index(axis)
        if axis<0: axis+=self.ndim
        if not all(hasattr(lb,'bm') for lb in nlabels): raise ValueError
        dims=[lb.bm.N for lb in nlabels]

        #get new labels
        newlabels=self.labels[:axis]+nlabels+self.labels[axis+1:]
        #generate the new tensor
        bms=[lb.bm for lb in nlabels]
        bm0=self.labels[axis].bm
        ndata={}
        for blk,data in self.data.iteritems():
            b0=blk[axis]
            indices0=bm0.get_slice(b0); indices0=np.arange(indices0.start,indices0.stop)
            cs=ind2c(indices0,N=dims)
            bcs=[bm.ind2b(ci) for ci,bm in zip(cs.T,bms)]
            bs,cs=np.concatenate([bc[0][:,None] for bc in bcs],axis=1),np.concatenate([bc[1][:,None] for bc in bcs],axis=1)
            for ii,(nb,nc) in enumerate(zip(bs,cs)):
                nblk=blk[:axis]+tuple(nb)+blk[axis+1:]
                nshape=tuple(bm.blocksize(bi) for bm,bi in zip(bms,nb))
                if not ndata.has_key(nblk): ndata[nblk]=np.zeros(data.shape[:axis]+nshape+data.shape[axis+1:],dtype=self.dtype)
                ndata[nblk][(slice(None),)*axis+tuple(nc)]=data.take(ii,axis).reshape(data.shape[:axis]+(1,)*len(nlabels)+data.shape[axis+1:])
        return BTensor(ndata,labels=newlabels)

    def split_axis_b(self,axis,nlabels):
        if isinstance(axis,str):
            axis=self.labels.index(axis)
        if axis<0: axis+=self.ndim
        if not all(hasattr(lb,'bm') for lb in nlabels): raise ValueError
        dims=[lb.bm.N for lb in nlabels]

        #get new labels
        newlabels=self.labels[:axis]+nlabels+self.labels[axis+1:]
        #get new datas, assume unity of blocks, not all axes can be split.
        bms=[lb.bm for lb in nlabels]
        bm0=self.labels[axis].bm
        ndata={}
        for blk,data in self.data.iteritems():
            indices0=bm0.Nr[blk[axis]]
            inds=ind2c(indices0,dims)
            bcs=[bm.ind2b(ci) for ci,bm in zip(inds,bms)]
            nb=tuple(bc[0] for bc in bcs)
            nc=tuple(bc[1] for bc in bcs[:-1])
            #set data
            nblk=blk[:axis]+nb+blk[axis+1:]
            nshape=tuple(bm.blocksize(bi) for bm,bi in zip(bms,nb))
            if not ndata.has_key(nblk): ndata[nblk]=np.zeros(data.shape[:axis]+nshape+data.shape[axis+1:],dtype=self.dtype)
            ndata[nblk][(slice(None),)*axis+tuple(nc)]=data.reshape(data.shape[:axis]+(1,)*(len(nlabels)-1)+(-1,)+data.shape[axis+1:])
        return BTensor(ndata,labels=newlabels)

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
            axis=(axis,)
        if isinstance(axis,tuple):
            axis=tuple(ax+self.ndim if ax<0 else ax for ax in axis)
            datas={}
            for bi,data in self.data.iteritems():
                blk=tuple(b for i,b in enumerate(bi) if i not in axis)
                datas[blk]=(datas[blk]+data.sum(axis=axis)) if datas.has_key(blk) else data.sum(axis=axis)
            t=BTensor(data=datas,labels=[lb for i,lb in enumerate(self.labels) if i not in axis])
            return t
        else:
            raise TypeError()

    @inherit_docstring_from(TensorBase)
    def get_block(self,block):
        return self.data.get(block,np.zeros([lb.bm.blocksize(n) for lb,n in zip(self.labels,block)],dtype=self.dtype))

    def conj(self):
        '''Conjugate.'''
        return BTensor(dict((blk,data.conj()) for blk,data in self.data.iteritems()),self.labels[:])

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
        for axis in axes:
            bm_new0,info=labels[axis].bm.sort(return_info=True)
            bm_new,Nr=bm_new0.compact_form(return_info=True) #Nr is the grouping of blocks.
            labels[axis]=labels[axis].chbm(bm_new)
            #reorder
            pm=info['pm_b']
            apm=np.argsort(pm)
            ndata={}
            nr0=bm_new0.nr
            for blk,data in ts.data.iteritems():
                ind=apm[blk[axis]]
                bid=np.searchsorted(Nr,ind+1)-1
                nblk=blk[:axis]+(bid,)+blk[axis+1:]
                if Nr[bid+1]-Nr[bid]!=1:
                    N0=Nr[bid]
                    iid=ind-N0
                    if not ndata.has_key(nblk):
                        ndata[nblk]=np.zeros([lb.bm.blocksize(bi) for bi,lb in zip(nblk,labels)],dtype=self.dtype)
                    offset=sum(nr0[N0:N0+iid])
                    ndata[nblk][(slice(None),)*axis+(slice(offset,offset+nr0[N0+iid]),)]=data
                else:
                    ndata[nblk]=(ndata[nblk]+data) if ndata.has_key(nblk) else data
            pms.append(pm)
            ts=BTensor(ndata,labels)
        if return_pm:
            return ts,pms
        else:
            return ts

    @inherit_docstring_from(TensorBase)
    def eliminate_zeros(self,tol=ZERO_REF):
        for blk,data in self.data.iteritems():
            if data.size==0:
                self.data.remove(blk)
            else:
                mask=abs(data)<ZERO_REF
                if all(mask):
                    self.data.remove(blk)
                else:
                    data[mask]=0
        return self
