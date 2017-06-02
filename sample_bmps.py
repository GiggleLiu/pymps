'''
This is a sample for the usage of MPS.

Task:
    1. Create a random ket with block structure kA,
    2. Make the ket left canonical and tell the good quantum number of ket.
    3. Generate an <MPO> H for Heisenberg model.
    4. Get <H> under kA.
'''

from numpy import sqrt,Inf

from mpslib import random_bmps
from opstring import OpCollection
from mpo import OPC2MPO
from mpolib import opunit_S
from tba.hgen import SpinSpaceConfig
from contraction import Contractor
from blockmatrix import SimpleBMG

########### Generate a random State ######
#the single site Hilbert space(Spin) config, 2 spin, 1 orbital.
spaceconfig=SpinSpaceConfig([1,2])
#the generator of block marker,
#using magnetization `M = nup-ndn` as good quantum number.
bmg=SimpleBMG(spaceconfig=spaceconfig,qstring='M')
#generate a random mps of 10 site and chi = 20,
#which is under the supervise of above <BlockMarkerGenerator>.
kA=random_bmps(bmg=bmg,nsite=10,maxN=20)
#nomalize it
kA=kA/sqrt(kA.tobra()*kA).item()

########### Get the Good quantum Number ######
kA<<kA.nsite,1e-8,Inf  #number of steps, tolerence, maximum bond dimension.
#get the block marker, 
#the (quantum number)flow direction is left->right, 
#so the right most bond contains the final good quantum number.
bm=kA.get(kA.nsite-1).labels[kA.rlink_axis].bm
print 'The Good Quantum Number of kA is %s = %s'%(kA.bmg.qstring,bm.qns[0])

########### Construct Hamiltonian ######
h,J=0.2,1  #the magnetic field, AFM couling.
opc=OpCollection()
Sx=opunit_S(spaceconfig=spaceconfig,which='z')
Sy=opunit_S(spaceconfig=spaceconfig,which='y')
Sz=opunit_S(spaceconfig=spaceconfig,which='z')
#add on-site terms
for i in xrange(kA.nsite):
    opc+=h*Sz.as_site(i)
#add coupling terms
for i in xrange(kA.nsite-1):
    opc+=Sx.as_site(i)*Sx.as_site(i+1)
    opc+=Sy.as_site(i)*Sy.as_site(i+1)
    opc+=Sz.as_site(i)*Sz.as_site(i+1)
H=OPC2MPO(opc,method='additive')  #bond dimension 37, contraction needed!
H.compress()    #bond dimension 4, the optimal case.

#Get <H>
con=Contractor(H,kA)
energy=con.evaluate().real
print '<H> = %s'%energy
