'''
Sample program for block matrix.

Task:
    1. Get a block marker for spin-1/2 fermionic system with 5-sites.
    2. Generate a random block diagonal matrix with respect to this block marker.
    3. Visualize it.
'''

import matplotlib.pyplot as plt

from .spaceconfig import SuperSpaceConfig
from .blockmarker import SimpleBMG
from .blocklib import show_bm, random_bdmat

# the single site Hilbert space config(Fermiononic)
# with 2 spin, 1 atom, 1 orbital in a unit cell.
spaceconfig = SuperSpaceConfig([2, 1, 1])
# use a block marker generator with quantum number
#'Q'-number of electron and 'M'-2*spin in z direction.
bmg = SimpleBMG('QM', spaceconfig=spaceconfig)
# update the null block marker for 5 times.
bm = bmg.bm0
for i in range(5):
    bm = bmg.update1(bm)
    bm = bm.sort().compact_form()
# generate a random block diagonal matrix.
A = random_bdmat(bm)
# visualize
plt.figure()
plt.pcolormesh(A)
show_bm(bm)
plt.show()
