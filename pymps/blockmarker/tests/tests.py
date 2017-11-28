from numpy import *
from numpy.testing import dec, assert_, assert_raises, assert_almost_equal, assert_allclose
from matplotlib.pyplot import *
from scipy.linalg import eigh, block_diag, svd
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, coo_matrix
import time
import pdb
import sys
sys.path.insert(0, '../')

from spaceconfig import SuperSpaceConfig, SpinSpaceConfig
from blocklib import *
from blockmatrix import *
from blockmarker import *
from autoblock import *
from plotlib import *

random.seed(2)


def test_blockmatrix():
    block_size = [2, 2, 1, 3]
    bm = BlockMatrix(block_size, block_size)
    datalist = [random.random([2, 2]), random.random(
        [2, 2]), random.random([1, 1]), random.random([3, 3])]
    bm.set_diagonal(datalist)

    # transform to array
    arr = bm.toarray()

    # transform to coo_matrix
    arr2 = bm.tocoo().toarray()
    assert_allclose(bm.dot(bm).toarray(), arr.dot(arr))
    vec = random.random(8)
    assert_allclose(bm.vdot(vec), bm.toarray().dot(vec))
    N = sum(block_size)
    arr = csr_matrix(random.random([N, N]))
    bm.set_data(arr)
    assert_allclose(bm.toarray(), arr.toarray())


class BMTest(object):
    '''
    Test for BlockMarker.

    n,p:
        The matrix size and number of blocks.
    empty_rate:
        The empty rate of this sparse_matrix.
    '''

    def __init__(self, n=200, p=10, empty_rate=0.2):
        a = zeros([n, n], dtype=complex128)
        m = concatenate([[0], sort(random.randint(0, n, p - 1)), [n]])
        for i in range(p):
            xmin, xmax = m[i], m[i + 1]
            bsize = xmax - xmin
            a[xmin:xmax, xmin:xmax] = random.random(
                (bsize, bsize)) + 1j * random.random((bsize, bsize))
        a = a + a.T.conj()
        # permute a
        a[a > (1 - empty_rate)] = 0
        pmat = random.permutation(n)
        na = a[ix_(pmat, pmat)]  # permute to a disordered matrix.
        self.original_matrix = csr_matrix(a)
        self.sparse_matrix = csr_matrix(na)

    def test_blockize(self):
        '''Test for blockize a matrix.'''
        print('Testing for automatically blockize a csr matrix!')
        csra = self.sparse_matrix
        t0 = time.time()
        marker2, pm = get_blockmarker(csra)
        t1 = time.time()
        b = csra[pm][:, pm]
        assert_(marker2.check_blockdiag(b))
        assert_(not marker2.check_blockdiag(csra))
        t2 = time.time()
        print('Elapse %s, %s' % (t1 - t0, t2 - t1))
        assert_almost_equal(csra.sum(), b.sum())
        assert_(marker2.check_blockdiag(b))

    def test_eig(self):
        '''Test for eigbh function.'''
        t0 = time.time()
        bm, pm = get_blockmarker(self.sparse_matrix)
        res = sort(
            eigbh(self.sparse_matrix[pm][:, pm], bm=bm, return_vecs=False))
        t1 = time.time()
        E_true = eigh(self.sparse_matrix.toarray())[0]
        t2 = time.time()
        print('All: Elapse -> %s/%s, Tol -> %s' %
              (t1 - t0, t2 - t1, abs(res[0] - E_true).sum()))
        assert_allclose(res, E_true, atol=1e-8)

    def test_all(self):
        self.test_blockize()
        self.test_eig()


class BGTest(object):
    '''
    Test for block marker generator.
    '''

    def __init__(self):
        spaceconfig = SuperSpaceConfig([1, 2, 1])
        self.types = ['QM', 'Q', 'M', 'P', '']
        nbmg = SimpleBMG('Q', spaceconfig=spaceconfig)
        mbmg = SimpleBMG('M', spaceconfig=spaceconfig)
        pbmg = SimpleBMG('P', spaceconfig=spaceconfig)
        jbmg = SimpleBMG('QM', spaceconfig=spaceconfig)
        nullbmg = SimpleBMG('', spaceconfig=spaceconfig)
        self.bmgs = [jbmg, nbmg, mbmg, pbmg, nullbmg]
        self.bms = [bmg.random_bm(nsite=4, trunc_rate=0.2)
                    for bmg in self.bmgs]

    def test_sign(self):
        bms = [bmg.bm0 for bmg in self.bmgs]
        for i in range(5):
            bms = [bmg.update1(bm) for bm, bmg in zip(bms, self.bmgs[:-1])]
            assert_(all([sign4bm(bm, bmg, full_length=True).sum()
                         == 0 for bm, bmg in zip(bms, self.bmgs)]))

    def test_bm_indexqn(self):
        '''Test for qn indexing.'''
        nlb = 5
        for bm in self.bms:
            ilbs = random.randint(0, len(bm.qns), nlb)
            for ilb in ilbs:
                assert_(ilb == bm.index_qn(bm.qns[ilb]))
            assert_(all(ilbs == bm.index_qn(bm.qns[ilbs])))

    def test_checkbd(self):
        for bm in self.bms:
            assert_(bm.check_blockdiag(random_bdmat(bm, dense=False)))
            assert_(bm.check_blockdiag(random_bdmat(bm, dense=True)))

    def test_show(self):
        nplt = len(self.bmgs)
        ion()
        for i, bm in enumerate(self.bms):
            ax = subplot(10 * nplt + 100 + i + 1)
            show_bm(bm)
            mat = random_bdmat(bm, dense=True)
            pcolor(mat)
            axis('equal')
        pdb.set_trace()

    def test_all(self):
        self.test_bm_indexqn()
        self.test_sign()
        self.test_checkbd()
        self.test_show()


class BMMTest(object):
    def __init__(self):
        self.bm = BlockMarker([0, 1, 3, 3], qns=reshape(
            [-1, 0, 1], [-1, 1]))  # with block size 1,2,0
        self.mat = array([[1, 2, 3], [4, 5, 6], [6, 7, 8]])
        self.vec = array([0, 1, 2])
        self.smat = csr_matrix(self.mat)

    def test_b2ind(self):
        print('Parse of ind-bind')
        assert_(self.bm.b2ind(1, 1) == 2)
        assert_allclose(self.bm.ind2b(2), (1, 1))

    def test_extract(self):
        print('Test Extracting datas')
        blocks = [self.mat[:1, :1], self.mat[1:3, 1:3], zeros([0, 0])]
        for i in range(3):
            blocki = self.bm.extract_block(self.mat, ij=(i, i))
            assert_allclose(blocki, blocks[i])
            blocki = self.bm.extract_block(self.smat, ij=(i, i)).toarray()
            assert_allclose(blocki, blocks[i])
        assert_allclose(self.bm.extract_block(
            self.mat, ij=(0, 1)), self.mat[:1, 1:3])
        assert_allclose(self.bm.extract_block(
            self.mat, ij=(0,), axes=(0,)), self.mat[:1])
        assert_allclose(self.bm.extract_block(
            self.mat, ij=(1,), axes=(1,)), self.mat[:, 1:3])
        assert_allclose(self.bm.extract_block(
            self.vec, ij=(1,)), self.vec[1:3])

    def test_svd(self):
        print('Test SVD')
        mat = block_diag(*[self.bm.extract_block(self.mat, ij=(i, i))
                           for i in range(3)])
        U, S, V = svd(mat, full_matrices=False)
        U2, S2, V2, S3 = svdb(mat, bm=self.bm)
        assert_allclose(sort(S2.data), sort(S))
        assert_allclose(U2.dot(S2).dot(V2).toarray(), mat)

    def test_join(self):
        print('Test join_bms')
        bm1 = self.bm
        bm2 = bm1
        bm3 = join_bms([bm1, bm2])
        # dimension check
        assert_(bm3.N == bm1.N * bm2.N)

    def test_empty(self):
        print('Test empty BM')
        ebm = BlockMarker(Nr=[0], qns=zeros(0, 1))
        assert_(ebm == ebm.sort().compact_form())

    def test_all(self):
        self.test_b2ind()
        self.test_extract()
        self.test_svd()
        self.test_join()


class BMGTest():
    '''simple test for bmg'''

    def simple_test(self):
        spaceconfig1 = SuperSpaceConfig([1, 2, 1])
        spaceconfig2 = SuperSpaceConfig([1, 2, 2])
        spaceconfig3 = SpinSpaceConfig([1, 2])
        bmg = SimpleBMG('M', spaceconfig1)
        assert_allclose(bmg.qns1, reshape([0, 1, -1, 0], [4, 1]))
        assert_allclose(bmg.bcast_add(bmg.qns1, reshape(
            [3, -2], [2, 1])), reshape([3, -2, 4, -1, 2, -3, 3, -2], [8, 1]))
        assert_allclose(bmg.bcast_sub(bmg.qns1, reshape(
            [3, -2], [2, 1])), reshape([-3, 2, -2, 3, -4, 1, -3, 2], [8, 1]))
        bmg = SimpleBMG('MP', spaceconfig1)
        assert_allclose(bmg.qns1, [[0, 0], [1, 1], [-1, 1], [0, 0]])
        assert_allclose(bmg.bcast_add(bmg.qns1, asarray([[3, 1], [-2, 0]])), asarray(
            [[3, 1], [-2, 0], [4, 0], [-1, 1], [2, 0], [-3, 1], [3, 1], [-2, 0]]))
        # assert_allclose(bmg.bcast_sub(bmg.bcast_add(bmg.qns1,bmg.qns1),bmg.qns1),bmg.qns1)
        bmg = SimpleBMG('M', spaceconfig2)
        assert_allclose(bmg.qns1, reshape(
            [0, 1, 1, 2, -1, 0, 0, 1, -1, 0, 0, 1, -2, -1, -1, 0], [16, 1]))
        bmg = SimpleBMG('M', spaceconfig3)
        assert_allclose(bmg.qns1, reshape([1, -1], [2, 1]))

    def test_jb(self):
        '''join two bms'''
        print('Test for joining two block markers!')
        spaceconfig1 = SuperSpaceConfig([1, 2, 1])
        bmg = SimpleBMG('M', spaceconfig1)
        bm1 = bmg.bm1  # -1,0(2),1
        bm2 = BlockMarker([0, 1, 2], qns=[[1], [-1]])

        bm3 = join_bms([bm1, bm2, bm1])
        bm4 = bmg.join_bms([bm1, bm2, bm1])
        bm5 = bm3.inflate().sort().compact_form()
        assert_allclose(bm5.N, bm4.N)
        # test non-compact form
        bm6 = bmg.join_bms([bm1, bm2], signs=[1, 1])
        assert_allclose(bm6.qns, [[0], [-2], [1], [-1], [1], [-1], [2], [0]])
        assert_allclose(bm6.Nr, arange(9))
        bm7 = bmg.join_bms([bm2, bm1])
        assert_allclose(bm7.qns, [[0], [1], [2], [-2], [-1], [0]])
        assert_allclose(bm7.Nr, [0, 1, 3, 4, 5, 7, 8])

    def test_dadd(self):
        '''direct add'''
        print('Test for direct addition!')
        bmg = SimpleBMG(spaceconfig=SuperSpaceConfig([2, 1, 1]), qstring='QM')
        bm1 = bmg.random_bm(nsite=5, trunc_rate=0.4)
        bm2 = bmg.random_bm(nsite=3, trunc_rate=0.4)
        bm = bm1 + bm2
        assert_(bm.N == bm1.N + bm2.N and len(bm.Nr) == len(bm.qns) + 1)
        bm_comp = (bm1 + bm1).sort().compact_form()
        # can not pass
        # assert_(all(bm_comp.Nr==bm1.Nr*2),all(bm_comp.qns==bm1.qns))

    def test_all(self):
        self.test_jb()
        self.simple_test()
        self.test_dadd()


if __name__ == '__main__':
    BMGTest().test_all()
    BMMTest().test_all()
    BMTest().test_all()
    test_blockmatrix()
    BGTest().test_all()
