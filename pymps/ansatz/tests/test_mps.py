#!/usr/bin/python
'''
Tests for MPS and MPO
'''
from __future__ import division
from numpy import *
from numpy.linalg import norm, svd
from copy import deepcopy
from numpy.testing import dec, assert_, assert_raises, assert_almost_equal, assert_allclose
from matplotlib.pyplot import *
import pdb
import time

from ...spaceconfig import SpinSpaceConfig
from ...toolbox.utils import quicksave, quickload
from ...blockmarker import SimpleBMG

from ..mpo import *
from ..mpolib import *
from ..mpslib import *
from ..contraction import *
from ..mps import MPS
from ..plotlib import show_mps


class MPSTest(object):
    '''Tests for MPS.'''

    def __init__(self, hndim=3, nsite=10, normalize=False):
        self.hndim = hndim
        self.nsite = nsite
        self.mps1 = None
        self.mps2 = None
        self.tol = 1e-12
        self.set_mps(1, normalize=normalize)
        self.set_mps(2, normalize=normalize)

    def set_mps(self, i, mps=None, normalize=False):
        '''Set up mps.'''
        assert(i == 1 or i == 2)
        if mps is None:
            hndim = self.hndim
            nsite = self.nsite  # number of sites
            # a random state in form of 1D array.
            vec = random.random(hndim**nsite) + 1j * \
                random.random(hndim**nsite)
            if normalize:
                vec /= norm(vec)
            print('Generating random MPS.')
            # parse the state into a <MPS> instance.
            mps = state2MPS(vec, sitedim=hndim, l=nsite // 2, method='svd')
            t0 = time.time()
            nstate = mps.state  # recover the 1D array state representation.
            t1 = time.time()
            assert_allclose(nstate, vec)
            print('State tolerence %s, Elapse -> %s' %
                  (sum(abs(nstate - vec)), t1 - t0))
        else:
            mps = mps
            vec = None
        if i == 1:
            self.mps1 = mps
        else:
            self.mps2 = mps

    def test_query(self):
        mps = self.mps1
        t0 = time.time()
        # query the state of a specified site config.
        qu = mps.query((self.hndim - 1) * ones(self.nsite))
        t1 = time.time()
        vec = mps.state
        print('Query the last site: %s(true: %s), Elapse -> %s' %
              (qu, vec[-1], t1 - t0))
        assert_(abs(qu - vec[-1]) < self.tol)

    def test_show(self):
        sleeptime = 1
        mps = self.mps1
        ion()
        show_mps(mps)
        show_mps(mps.tobra(labels=mps.labels), offset=(0, 2))
        pause(sleeptime)

    def test_canonical(self):
        mps = self.mps1
        canon = check_canonical(mps)
        print('Checking for unitary for M-matrices.\n', canon)
        assert_(all(canon))

    def test_compress(self):
        '''
        Test for addition of two <MPS> instances.
        '''
        mps, mps2 = self.mps1, self.mps1
        mps << mps.l
        mps2 << mps2.l
        vec, vec2 = mps.state, mps2.state
        vecadded = vec + vec2
        mpsadded = mps + mps2
        # mpsadded=mps_sum([mps,mps2])
        t0 = time.time()
        mpsadded.compress(tol=0, maxN=Inf)
        t1 = time.time()
        # recover the 1D array state representation.
        nstateadded = mpsadded.state
        print('State tolerence(after compress, l = %s) %s Elapse -> %s' %
              (mps.l, sum(abs(nstateadded - vecadded)), t1 - t0))
        assert_allclose(nstateadded, vecadded)  # the state should be unchanged

        # for l=0 case
        mps << mps.l
        mps2 << mps2.l
        vecadded = vec + vec2
        mpsadded = mps_sum([mps, mps2])
        mpsadded.compress(maxN=500)
        print('State tolerence(after compress, l = %s) %s Elapse -> %s' %
              (mps.l, sum(abs(nstateadded - vecadded)), t1 - t0))
        assert_allclose(nstateadded, vecadded)  # the state should be unchanged

        # for l=0 case
        mps >> mps.nsite - mps.l
        mps2 >> mps2.nsite - mps2.l
        vecadded = vec + vec2
        mpsadded = mps_sum([mps, mps2])
        mpsadded.compress()
        print('State tolerence(after compress, l = %s) %s Elapse -> %s' %
              (mps.l, sum(abs(nstateadded - vecadded)), t1 - t0))
        assert_allclose(nstateadded, vecadded)  # the state should be unchanged

    def test_addmuldiv(self):
        '''
        Test for addition of two <MPS> instances.
        '''
        mps, mps2 = self.mps1, self.mps2
        mps2 << mps2.l - mps.l
        vec, vec2 = mps.state, mps2.state
        vecadded = vec + vec2
        t0 = time.time()
        mpsadded = mps_sum([mps, mps2]).recanonicalize()
        t1 = time.time()
        # recover the 1D array state representation.
        nstateadded = mpsadded.state
        print('State tolerence(added), %s Elapse -> %s' %
              (sum(abs(nstateadded - vecadded)), t1 - t0))
        assert_allclose(nstateadded, vecadded)

        # test for multiply
        factor = random.random()
        vec_mul = vec * factor
        mps_mul = mps * factor
        mps *= factor
        assert_allclose(mps.state, vec_mul)
        assert_allclose(mps_mul.state, vec_mul)

        # test for devision
        factor = random.random()
        vec_div = vec_mul / factor
        mps_div = mps / factor
        mps /= factor
        assert_allclose(mps.state, vec_div)
        assert_allclose(mps_div.state, vec_div)

    def test_move(self):
        '''
        Test for canonical move of <MPS>.
        '''
        ion()
        fig = gcf()
        sleeptime = 1
        mps = self.mps1
        vec = mps.state
        cla()
        show_mps(mps)
        pause(sleeptime)
        l0 = mps.l
        t0 = time.time()
        mps << (1, 1e-5, Inf)
        t1 = time.time()
        nstate = mps.state  # recover the 1D array state representation.
        print('State tolerence %s, Elapse -> %s' %
              (sum(abs(nstate - vec)), t1 - t0))
        l1 = mps.l
        assert_(l1 - l0 == -1)
        cla()
        show_mps(mps)
        pause(sleeptime)

        t0 = time.time()
        mps << (4, 1e-6, Inf)
        t1 = time.time()
        l2 = mps.l
        nstate = mps.state  # recover the 1D array state representation.
        print('State tolerence %s, Elapse -> %s' %
              (sum(abs(nstate - vec)), t1 - t0))
        assert_(l2 - l1 == -4)
        cla()
        show_mps(mps)
        pause(sleeptime)

    def test_copy(self):
        '''
        Test deepcopy and shallow copy.
        '''
        mps = self.mps1
        mps2 = self.mps2
        newlabel_s = ['site_shallow', 'link_shallow']
        mps_shallow = mps.toket(labels=newlabel_s)
        bra_shallow = mps.tobra(labels=newlabel_s)
        # labels are not synchronized!
        assert_(mps_shallow.labels == newlabel_s)
        assert_(bra_shallow.labels == newlabel_s)
        mps_shallow.labels[0] = 'AAA'
        assert_(not mps.labels[0] == 'AAA')
        assert_(not bra_shallow.labels[0] == 'AAA')

        # but datas between shallow copyed one and original one are synchronized!
        mps_shallow.ML[3][...] = mps2.ML[3]
        assert_allclose(mps_shallow.get(3, attach_S=''), mps2.get(3, ''))
        assert_allclose(mps.get(3, attach_S=''), mps2.get(3, attach_S=''))

        assert_allclose(mps.get(3, attach_S=''), mps2.get(3, ''))

    def test_saveload(self):
        '''
        Test for save and load for mps.
        '''
        filename = 'test_mps_saveload.dat'
        mps = self.mps1
        quicksave(filename, mps)
        mps2 = quickload(filename)
        vec = mps.state
        vec2 = mps2.state
        assert_allclose(vec, vec2)

    def test_all(self):
        ion()
        print('Testing for query mps')
        self.test_query()
        print('Testing for mps canonicality')
        self.test_canonical()
        print('Testing for compressing two mpses')
        self.test_compress()
        # print 'Testing for overlap between two mpses'
        # self.test_contraction()
        print('Testing for adding of two mpses/multiplying and dividing by a factor')
        self.test_addmuldiv()

        print('Testing for deep and shallow copy and ket-bra transformation.')
        self.test_copy()
        print('Testing for save and load.')
        self.test_saveload()

        print('Testing for displaying')
        self.test_show()
        print('Testing for canonical move.')
        self.test_move()


def test_prodmps():
    print('Testing for generating random product state and transform to MPS.')
    nsite = 5
    hndim = 2
    print('Testing method1, tell the occupied indices')
    config = random.randint(0, hndim, nsite)
    vecs = zeros([nsite, hndim])
    vecs[arange(nsite), config] = 1
    vec = vecs[0]
    for i in range(nsite - 1):
        vec = kron(vec, vecs[i + 1])
    print('Get random product state(nsite=%s,hndim=%s) %s' %
          (nsite, hndim, config))
    mps = product_state(config, hndim)
    assert_allclose(vec, mps.state, atol=1e-8)

    print('test using block markers')
    bmg = SimpleBMG(spaceconfig=SpinSpaceConfig([1, 2]), qstring='M')
    mps_b = product_state(config, hndim, bmg=bmg)
    mps_b0 = product_state(config, hndim)
    assert_(check_validity_mps(mps_b0))

    print('Testing method2, tell the state on each site.')
    vecs = random.random([nsite, hndim])
    vecs = vecs / norm(vecs, axis=1)[:, newaxis]
    mps = product_state(vecs)
    vec = vecs[0]
    for i in range(nsite - 1):
        vec = kron(vec, vecs[i + 1])
    assert_allclose(vec, mps.state, atol=1e-8)

    print('Generate a MPS for random product state.')
    # don't know how to test it.
    mps = random_product_state(nsite=nsite, hndim=hndim)
    assert_(all(check_canonical(mps)))


def test_random_mps():
    print('Testing for generating random mps.')
    mps = random_mps(hndim=2, nsite=10, maxN=20)
    print('Testing Using a block marker.')
    bmg = SimpleBMG(spaceconfig=SpinSpaceConfig([1, 2]), qstring='M')
    bmps = mps.use_bm(bmg)
    assert_(check_validity_mps(mps))


if __name__ == '__main__':
    test_random_mps()
    test_prodmps()
    t = MPSTest(normalize=False, nsite=6)
    t.test_all()
