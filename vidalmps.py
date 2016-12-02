'''
Matrix Product State.
'''

from numpy import *
from tba.hgen import inherit_docstring_from
import pdb,time,copy

import tensor
from mps import MPSBase

__all__=['VidalMPS']

ZERO_REF=1e-12

class VidalMPS(MPSBase):
    '''
    Matrix Product state in the standard Vidal form.

    Construct:
        VidalMPS(GL,LL,labels)

    Attributes:
        :GL: list of 3D array, a list of Gamma Matrices.
        :LL: list of 1D array, a list of Lambda Matrices(the diagonal part).
        :labels: len-2 list of str, the labels for auto-labeling in MPS [site, link].
        :factor: float, the overall factor for this state.
    '''
    def __init__(self,GL,LL,labels,factor):
        assert(len(GL)==len(LL)+1)
        assert(len(labels)==2)
        assert(ndim(factor)==0)
        self.labels=labels
        self.factor=factor
        s,a=labels
        gl=[]
        for i,G in enumerate(GL):
            lbs=[None]*3
            lbs[self.site_axis]='%s_%s'%(s,i)
            lbs[self.llink_axis]='%s_%s'%(a,i)
            lbs[self.rlink_axis]='%s_%s'%(a,i+1)
            gl.append(tensor.Tensor(G,labels=lbs))
        self.GL=gl
        self.LL=LL

    def __str__(self):
        string='<VMPS,%s>\n'%(self.nsite)
        for i in xrange(self.nsite):
            g=self.GL[i]
            string+='  G[s=%s] (%s x %s) (%s,%s,%s)\n'%(\
                g.shape[self.site_axis],g.shape[self.llink_axis],g.shape[self.rlink_axis],\
                g.labels[self.site_axis],g.labels[self.llink_axis],g.labels[self.rlink_axis])
            if i!=self.nsite-1:
                l=self.LL[i]
                string+='  L (%s x %s)\n'%(l.shape[0],l.shape[0])
        return string

    @property
    @inherit_docstring_from(MPSBase)
    def hndim(self):
        return self.GL[0].shape[self.site_axis]

    @property
    @inherit_docstring_from(MPSBase)
    def nsite(self):
        return len(self.GL)

    @property
    @inherit_docstring_from(MPSBase)
    def state(self):
        ULG=[gi.reshape([-1,gi.shape[self.rlink_axis]]) for gi in self.GL]
        LL=self.LL
        state=identity(1)
        for ui,li in zip(ULG[::-1],[array([self.factor])]+LL[::-1]):
            state=asarray(state.reshape([li.shape[0],-1]))*li[:,newaxis]
            state=reshape(state,(ui.shape[1],-1))
            state=ui.dot(state)
        return state.ravel()

    def get_rho(self,l=None):
        '''
        Density matrix between site l and l+1.
        
        l:
            The site index.
        '''
        if l is None:
            rl=[diag(li**2) for li in self.LL]
            return rl
        else:
            rl=diag(self.LL[l]**2)
            return rl

    def tocanonical(self,labels=None,l=0):
        '''
        Get the canonical form for this MPS.

        Parameters:
            :labels: list,
            :l: int, specify the canonicality of ket.
        '''
        if labels is None:
            labels=self.labels[:]
        nsite=self.nsite
        hndim=self.hndim
        assert(0<=l and l<=nsite)
        GL=self.GL
        LL=[ones(1)]+self.LL+[ones(1)]
        ML=[LL[i][:,newaxis,newaxis]*asarray(GL[i]) for i in xrange(l)]
        ML.extend([asarray(GL[i])*LL[i+1] for i in xrange(l,nsite)])
        S=LL[l]*self.factor
        return MPS(ML,l,S,labels=labels)
