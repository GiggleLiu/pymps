'''
Tensor Class
'''

import numpy as np
import numbers,copy,pdb
import numbers,itertools
from numpy import array,einsum
from numpy.linalg import norm
from scipy.linalg import svd
from abc import ABCMeta, abstractmethod

from tba.hgen import c2ind,SuperSpaceConfig,inherit_docstring_from
from blockmatrix import block_diag,SimpleBMG,BlockMarker
from tensor import tdot,Tensor,BLabel
from btensor import BTensor

__all__=['random_tensor','random_bbtensor','check_validity_tensor',
        'gen_eincode','contract','random_btensor','svdbd','random_bdmatrix']

def gen_eincode(*labels):
    '''
    Generate einsum string from contraction strings of tensors.

    labels:
        The labels for contraction.
    '''
    all_list=sum(labels,[])
    unique_list,legs=[],[]
    for item in all_list:
        if item in unique_list:
            legs.remove(item)
        else:
            unique_list.append(item)
            legs.append(item)
    mapping=dict((label,chr(97+i)) for i,label in enumerate(unique_list))
    tokens=[]
    for lbs in labels:
        tokens.append(''.join([mapping[l] for l in lbs]))
    token_r=''.join([mapping[l] for l in legs])
    return '%s->%s'%(','.join(tokens),token_r),legs

def contract(*tensors):
    '''
    Contract a collection of tensors
    
    Parameters:
        :tensors: <Tensor>s, A list of <Tensor> instances.

    Return:
        <Tensor>
    '''
    if len(tensors)==0:
        raise ValueError('Not enough parameters.')
    if len(tensors)==1:
        tensors=tensors[0]
    labels=[t.labels for t in tensors]
    #asign dummy tokens.
    eincode,leglabels=gen_eincode(*labels)
    return Tensor(einsum(eincode,*tensors),labels=leglabels)

def random_bbtensor(sites=None,labels=None,nnzblock=100):
    '''
    Generate a random Block-BTensor.

    Parameters:
        :labels: list/None, the labels.
        :sites: int, the number of sites(prop to blocks).
        :nnzblock: int, the approximate number of non zeros entries.

    Return:
        <BTensor>
    '''
    spaceconfig=SuperSpaceConfig([2,1,1])
    #get block markers
    if sites is None:
        sites=[2,3,4]
    ndim=len(sites)
    if labels is None:
        labels=['a_%s'%i for i in xrange(ndim)]
    bms=[SimpleBMG(spaceconfig=spaceconfig,qstring='QM').random_bm(nsite=sites[i]) for i in xrange(ndim)]

    #get unique nzblocks.
    nbs=[bm.nblock for bm in bms]
    nzblocks=np.concatenate([np.random.randint(0,nb,nnzblock)[:,np.newaxis] for nb in nbs],axis=1)
    b = np.ascontiguousarray(nzblocks).view(np.dtype((np.void, nzblocks.dtype.itemsize * nzblocks.shape[1])))
    idx=np.unique(b,return_index=True)[1]
    nzblocks=nzblocks[idx]

    #get entries
    data=dict((tuple(blk),np.random.random([bm.blocksize(blki) for blki,bm in zip(blk,bms)])) for blk in nzblocks)

    #generate BTensor
    return BTensor(data,labels=[BLabel(lb,bm) for lb,bm in zip(labels,bms)])

def random_tensor(shape=None,labels=None):
    '''
    Generate a random Tensor.

    Parameters:
        :shape: the shape of tensor.
        :labels: the labels of axes.

    Return:
        <Tensor>
    '''
    if shape is None:
        shape=(20,30,40)
    ndim=len(shape)
    if labels is None:
        labels=['a_%s'%i for i in xrange(ndim)]
    data=np.random.random(shape)

    #generate Tensor
    return Tensor(data,labels=labels)

def random_btensor(bms,label_strs=None,fill_rate=0.2):
    '''
    Generate a random Block Dense Tensor.

    Parameters:
        :bms: list, the block markers.
        :label_strs: list/None, the labels.
        :fill_rate: propotion of the number of filled blocks.

    Return:
        <Tensor>
    '''
    #get block markers
    ndim=len(bms)
    if label_strs is None:
        labels=[BLabel('a_%s'%i,bm) for i,bm in enumerate(bms)]
    else:
        labels=[BLabel(lb,bm) for lb,bm in zip(label_strs,bms)]
    ts=Tensor(np.zeros([bm.N for bm in bms]),labels=labels)
    #insert datas
    nnzblock=int(fill_rate*np.prod([bm.nblock for bm in bms]))
    for i in xrange(nnzblock):
        target=ts[tuple([bm.get_slice(np.random.randint(0,bm.nblock)) for bm in bms])]
        target[...]=np.random.random(target.shape)

    #generate Blocked Tensor
    return ts

def random_bdmatrix(bm=None,dtype='complex128'):
    '''
    Generate a random Block Diagonal 2D Tensor.

    Parameters:
        :bm: <BlockMarker>

    Return:
        <Tensor>,
    '''
    if dtype=='complex128':
        cells=[(np.random.random([ni,ni])+1j*np.random.random([ni,ni]))*2/ni for ni in bm.nr]
    elif dtype=='float64':
        cells=[np.random.random([ni,ni])*2/ni for ni in bm.nr]
    else:
        raise ValueError()
    ts=Tensor(block_diag(*cells),labels=[BLabel('r',bm),BLabel('c',bm)])
    return ts

def check_validity_tensor(ts):
    '''Check if it is a valid tensor.'''
    valid=True
    for i in xrange(np.ndim(ts)):
        if hasattr(ts.labels[i],'bm'):
            if not ts.shape[i]==ts.labels[i].bm.N: valid=False
    if isinstance(ts.data,dict):
        #check data
        for blk,d in ts.data.iteritems():
            if not tuple(lb.bm.blocksize(bi) for lb,bi in zip(ts.labels,blk))==d.shape:
                valid=False
    return valid

def svdbd_map(A,mapping_rule=None,full_matrices=False):
    '''
    Get the svd decomposition for dense tensor with block structure.

    Parameters:
        :A: 2D<Tensor>, the input matrix, with <BLabel>s.
        :mapping_rule: function, the mapping between left block and right blocks, using labels.

    Return:
        (eigenvalues,eigenvectors) if return vecs==True.
        (eigenvalues,) if return vecs==False.
    '''
    #check datas
    if mapping_rule is None: mapping_rule=lambda x:x
    bm1,bm2=A.labels[0].bm,A.labels[1].bm
    extb1,extb2=bm1.extract_block,bm2.extract_block
    qns1,qns2=bm1.labels,list(bm2.labels)
    SL,UL,VL,SL2=[],[],[],[]

    um_l,m_r=[],[]  #un-matched blocks for left and matched for right
    for i,lbi in enumerate(qns1):
        lbj=mapping_rule(lbi) if mapping_rule is not None else lbi
        try:
            j=qns2.index(lbj)
            m_r.append(j)
        except:
            um_l.append(i)
            size=bm1.blocksize(bm1.index_qn(lbi)[0])
            UL.append(identity(size))
            SL.append(sps.csr_matrix((size,size)))
            continue
        mi=extb2(extb1(A,(bm1.index_qn(lbi).item(),),axes=(0,)),(bm2.index_qn(lbj).item(),),axes=(1,))
        if mi.shape[0]==0 or mi.shape[1]==0:
            ui,vi=zeros([mi.shape[0]]*2),zeros([mi.shape[1]]*2)
            SL.append(sps.csr_matrix(tuple([mi.shape[0]]*2)))
            SL2.append(sps.csr_matrix(tuple([mi.shape[1]]*2)))
        else:
            ui,si,vi=svd(mi,full_matrices=full_matrices)
            if mi.shape[1]>mi.shape[0]:
                si1,si2=si,append(si,zeros(mi.shape[1]-mi.shape[0]))
            elif mi.shape[1]<mi.shape[0]:
                si1,si2=append(si,zeros(mi.shape[0]-mi.shape[1])),si
            else:
                si1=si2=si
            SL.append(sps.diags(si1,0))
            SL2.append(sps.diags(si2,0))
        UL.append(ui)
        VL.append(vi)
    for j in xrange(bm2.nblock):
        if not j in m_r:
            m_r.append(j)
            size=bm2.nr[j]
            VL.append(identity(size))
            SL.append(sps.csr_matrix((0,0)))
            SL2.append(sps.csr_matrix(tuple([size]*2)))
    order=argsort(m_r)
    #reorder S and V, and build matrices
    m_l=ones(len(SL),dtype='bool')
    m_l[um_l]=False
    um_r=ones(len(SL),dtype='bool')
    um_r[m_r]=False

    Smat=ndarray([len(SL)]*2,dtype='O')
    Smat[m_l,m_r]=array(SL)[m_l]
    Smat[um_l,um_r]=array(SL)[um_l]
    Smat2=ndarray([len(SL2)]*2,dtype='O')
    Smat2[arange(len(SL2)),m_r]=SL2
    VL=array(VL)[order]
    return block_diag(*UL),array(sps.bmat(Smat)),block_diag(*VL),array(sps.bmat(Smat2))

def svdbd(A,cbond_str='X'):
    '''
    Get the svd decomposition for dense tensor with block structure.

    Parameters:
        :A: 2D<Tensor>, the input matrix, with <BLabel>s.
        :cbond_str: the labes string for center bond.

    Return:
        (U,S,V) that U*S*V = A
    '''
    #check and prepair datas
    bm1,bm2=A.labels[0].bm,A.labels[1].bm
    #add support for null block marker
    if bm1.qns.shape[1]==0:
        U,S,V=svd(A,full_matrices=False)
        center_label=BLabel(cbond_str,BlockMarker(qns=np.zeros([1,0],dtype='int32'),Nr=array([0,len(S)])))
        U=Tensor(U,labels=[A.labels[0].bm,center_label])
        V=Tensor(V,labels=[center_label,A.labels[1].bm])
        return U,S,V
    extb1,extb2=bm1.extract_block,bm2.extract_block
    qns1,qns2=bm1.qns,bm2.qns
    qns1_1d = qns1.copy().view([('',qns1.dtype)]*qns1.shape[1])
    qns2_1d = qns2.copy().view([('',qns2.dtype)]*qns2.shape[1])
    common_qns_1d=np.intersect1d(qns1_1d,qns2_1d)
    common_qns_2d=common_qns_1d.view(bm1.qns.dtype).reshape(-1,bm1.qns.shape[-1])

    #do SVD
    UL,SL,VL=[],[],[]
    for lbi in common_qns_2d:
        cell=extb2(extb1(A,(bm1.index_qn(lbi).item(),),axes=(0,)),(bm2.index_qn(lbi).item(),),axes=(1,))
        Ui,Si,Vi=svd(cell,full_matrices=False)
        UL.append(Ui); SL.append(Si); VL.append(Vi)

    #get correct shape of UL
    ptr=0
    for i,lbi_1d in enumerate(qns1_1d):
        if lbi_1d!=common_qns_1d[ptr]:
            UL.insert(i,np.zeros([bm1.blocksize(i),0],dtype=A.dtype))
        elif ptr!=len(common_qns_1d)-1:
            ptr=ptr+1

    #the same for VL
    ptr=0
    for i,lbi_1d in enumerate(qns2_1d):
        if lbi_1d!=common_qns_1d[ptr]:
            VL.insert(i,np.zeros([0,bm2.blocksize(i)],dtype=A.dtype))
        elif ptr!=len(common_qns_1d)-1:
            ptr=ptr+1
    nr=[len(si) for si in SL]
    Nr=np.append([0],np.cumsum(nr))
    b0=BLabel(cbond_str,BlockMarker(Nr=Nr,qns=common_qns_2d))
    U,S,V=Tensor(block_diag(*UL),labels=[A.labels[0],b0]),np.concatenate(SL),Tensor(block_diag(*VL),labels=[b0,A.labels[1]])

    #detect a shape error raised by the wrong ordering of block marker.
    if A.shape[0]!=U.shape[0]:
        raise Exception('Error! 1. check block markers!')
    return U,S,V
