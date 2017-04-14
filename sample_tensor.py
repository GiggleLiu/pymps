'''
Tensor product.

Task:
    1. Generate a random block diagonal 2D tensor T2 and truncate half of the 2nd axis.
    2. Perform Block SVD, get new block diagonal U and V.
'''

from numpy import random,arange,allclose,argsort,sort
import pdb

from tba.hgen import SpinSpaceConfig
from blockmatrix import SimpleBMG
from tensor import Tensor
from tensorlib import random_bdmatrix,random_btensor,svdbd

#the single site Hilbert space(Spin) config, 2 spin, 1 orbital.
spaceconfig=SpinSpaceConfig([1,2])
#the generator of block marker,
#using magnetization `M = nup-ndn` as good quantum number.
bmg=SimpleBMG(spaceconfig=spaceconfig,qstring='M')
#generate a random block marker of 6 site and truncation rate 0.3,
bm=bmg.random_bm(nsite=6,trunc_rate=0.3)  
#use the generated block marker to generate a block diagonal matrix.
T2=random_bdmatrix(bm=bm)
#truncate the 2nd axis,
indices=arange(T2.shape[1]); random.shuffle(indices)
remained_bonds=sort(indices[:T2.shape[1]/2])
T2=T2.take(remained_bonds,axis=1)
T3,pm=T2.b_reorder(axes=(1,),return_pm=True)
#perform SVD decomposition, both block markers
#of T2 should be in compact form - all labels in `correct ordered`.
U,S,V=svdbd(T3)
#calculate the SVD error
err=abs(U.mul_axis(S,1)*V-T3).sum().item()
print 'The tolerence of our block-SVD decomposition is %s.'%err
