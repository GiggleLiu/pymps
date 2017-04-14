'''
This is a sample for the usage of MPS.

Task:
    1. Create two random orthogonal ket `kA`,`kB`.
    2. Define correlation operator op(i)=nup(0)*ndn(i)
        Evaluate <op> = (<kA|+<kB|)/sqrt(2) op (|kA>+|kB>)/sqrt(2).
'''

from numpy import sqrt

from mpslib import random_mps
from mpolib import opunit_N
from tba.hgen import SuperSpaceConfig
from contraction import get_expect

########### Generate 2 random kets ######
#the single site Hilbert space config(Fermiononic),
#2 spin, 1 atom, 1 orbital in a unit cell.
spaceconfig=SuperSpaceConfig([1,2,1])
#generate 2 random mpses of 10 site and chi = 20/30,
kA=random_mps(hndim=spaceconfig.hndim,nsite=10,maxN=20)
kB=random_mps(hndim=spaceconfig.hndim,nsite=10,maxN=30)
#compress it, also make it canonical.
kA.compress(tol=1e-8)
kB.compress(tol=1e-8)

#nomalize them
kA=kA/sqrt(kA.tobra()*kA).item()
kB=kB-(kA.tobra()*kB).item()*kA  #orthogonalize with kA
kB=kB/sqrt(kB.tobra()*kB).item()
#add them, and renormalize them
ket=(kA+kB)/sqrt(2.)
#the results are not canonical, 
#so recanonicalization is needed in many application.
ket=ket.recanonicalize(tol=1e-10,maxN=40)
#assure they are normalized with tol=1e-5.
assert(abs(ket.tobra()*ket-1)<1e-5)

########### Generate the operator ######
nup=opunit_N(spaceconfig=spaceconfig,index=0)
ndn=opunit_N(spaceconfig=spaceconfig,index=1)
#evaluate <op>
res=[get_expect(nup.as_site(0)*ndn.as_site(i),ket).item()\
        for i in xrange(ket.nsite)]
print 'We Get Correlations: %s'%res
