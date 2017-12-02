from numpy import *
from numpy.linalg import norm, svd
from numpy.testing import dec, assert_, assert_raises, assert_almost_equal, assert_allclose
import sys
import pdb
import time, pytest

from ...toolbox.utils import quicksave, quickload, nullobj
from ...blockmarker import SimpleBMG, BlockMarker
from ...spaceconfig import SpinSpaceConfig, SuperSpaceConfig
from ..tensor import *
from ..random import *
from ..linalg import *
from ..btensor import BTensor
from ..tensorlib import *
from ..zero_flux import is_zero_flux, zero_flux_blocks, btdot


@pytest.fixture
def btensor2d():
    self = nullobj()
    # make data that sparse enough
    mat = array([[3, 3, 0, 0, 0],
                 [0, 0, 1, 2, 2],
                 [4, 4, 0, 0, 0],
                 [4, 4, 0, 0, 0]])
    bm1 = BlockMarker([0, 1, 2, 4], [[1], [0], [-1]])
    bm2 = BlockMarker([0, 2, 3, 5], [[-1], [1], [0]])
    labels = [BLabel('a', bm1), BLabel('b', bm2)]
    self.dtensor = Tensor(mat, labels=labels)
    data = {(0, 0): mat[:1, :2], (1, 1): mat[1:2, 2:3],
            (1, 2): mat[1:2, 3:], (2, 0): mat[2:, :2]}
    self.btensor = BTensor(data, labels)
    assert_(check_validity_tensor(self.btensor))
    assert_(check_validity_tensor(self.dtensor))

    # get second version.
    mat2 = array([[3, 3, 0, 0, 0],
                  [0, 0, 1, 2, 2],
                  [4, 4, 0, 0, 0]])
    bm1 = BlockMarker([0, 1, 2, 3], [[-2], [0], [1]])
    bm2 = BlockMarker([0, 2, 3, 5], [[-1], [0], [1]])
    labels = [BLabel('c', bm1), BLabel('b', bm2)]
    self.dtensor2 = Tensor(mat2, labels=labels)
    data = {(0, 0): mat2[:1, :2], (1, 1): mat2[1:2, 2:3],
            (1, 2): mat2[1:2, 3:], (2, 0): mat2[2:, :2]}
    self.btensor2 = BTensor(data, labels)
    assert_(check_validity_tensor(self.btensor2))
    assert_(check_validity_tensor(self.dtensor2))
    return self

def test_mul(btensor2d):
    print('Test for multiplication')
    dt = btensor2d.dtensor * btensor2d.dtensor2
    bt = btensor2d.btensor * btensor2d.btensor2
    assert_allclose(dt, bt.todense())
    assert_(all(dt.labels[i] == bt.labels[i]
                for i in range(len(dt.labels))))

def test_bdiag(btensor2d):
    print('Test for tensor_block_diag.')
    for axes in [(-2,), (0, 1)]:
        dres = tensor_block_diag([btensor2d.dtensor, btensor2d.dtensor2], axes=axes)
        bres = tensor_block_diag([btensor2d.btensor, btensor2d.btensor2], axes=axes)
        assert_(
            all([lb1.bm == lb2.bm for lb1, lb2 in zip(dres.labels, bres.labels)]))
        assert_allclose(dres, bres.todense())

def test_data_parsing(btensor2d):
    print('test for data parsing')
    assert_allclose(btensor2d.dtensor, btensor2d.dtensor.tobtensor().todense())
    assert_allclose(btensor2d.dtensor, btensor2d.btensor.todense())
    print((btensor2d.btensor))

def test_mul_axis(btensor2d):
    print('Testing multiplying 1D array to specific axis.')
    axis = 1
    arr = random.random(btensor2d.dtensor.shape[axis])
    assert_allclose(btensor2d.btensor.mul_axis(
        arr, axis=axis).todense(), btensor2d.dtensor.mul_axis(arr, axis=axis))

def test_sum(btensor2d):
    print('Test sum of tensor.')
    assert_(btensor2d.btensor.sum() == btensor2d.dtensor.sum())
    assert_allclose(btensor2d.btensor.sum(axis=-1).todense(),
                    btensor2d.dtensor.sum(axis=-1))
    assert_allclose(btensor2d.btensor.sum(axis=(-2, -1)).todense(),
                    btensor2d.dtensor.sum(axis=(-2, -1)))

def test_take(btensor2d):
    print('Testing taking axes!')
    axis = 0
    i = 2
    axis_label = btensor2d.btensor.labels[axis]
    t1 = btensor2d.dtensor.take(i, axis=axis)
    t11 = btensor2d.dtensor.take(i, axis=axis_label)
    t22 = btensor2d.btensor.take(i, axis=axis_label)
    res = [4, 4, 0, 0, 0]
    assert_allclose(t11, res)
    assert_allclose(res, t22.todense())
    assert_allclose(t1, res)
    print('Testing taking block - axes!')
    ib, axis = 2, 1
    res = zeros([4, 2])
    res[1] = 2
    assert_allclose(btensor2d.dtensor.take_b(ib, axis=axis), res)
    assert_allclose(btensor2d.btensor.take_b(ib, axis=axis).todense(), res)

def test_chorder(btensor2d):
    print('Testing chorder!')
    norder = [1, 0]
    labels = [btensor2d.dtensor.labels[i] for i in norder]
    t1 = btensor2d.dtensor.chorder(norder)
    t2 = btensor2d.btensor.chorder(norder)
    assert_allclose(t1, t2.todense())
    t11 = btensor2d.dtensor.chorder(labels)
    # t22=btensor2d.btensor.chorder(labels)
    # assert_allclose(t11,t22.todense())
    assert_allclose(t11, t1)

def test_merge_split(btensor2d):
    print('Testing merge!')
    axes = slice(0, 2)
    t1 = btensor2d.btensor.merge_axes(axes)
    t2 = btensor2d.dtensor.merge_axes(axes)
    lbs = btensor2d.btensor.labels[axes]
    tval = asarray(btensor2d.dtensor).reshape(
        [prod(btensor2d.dtensor.shape[:2])] + list(btensor2d.dtensor.shape[2:]))
    assert_allclose(t2, tval)
    assert_allclose(t2, t1.todense())
    print('Testing split!')
    t11 = t1.split_axis(axis=axes.start, nlabels=lbs)
    # [lb.chbm(lb.bm.inflate()) if i!=len(lbs)-1 else lb for i,lb in enumerate(lbs)])
    t11b = t1.split_axis_b(axis=axes.start, nlabels=lbs)
    t22 = t2.split_axis(axis=axes.start, nlabels=lbs)
    assert_(all([lb1 == lb2 for lb1, lb2 in zip(
        t11.labels, btensor2d.dtensor.labels)]))
    assert_(all([lb1 == lb2 for lb1, lb2 in zip(
        t11b.labels, btensor2d.dtensor.labels)]))
    assert_(all([lb1 == lb2 for lb1, lb2 in zip(
        t22.labels, btensor2d.dtensor.labels)]))
    assert_allclose(t22, btensor2d.dtensor)
    assert_allclose(t11.todense(), btensor2d.dtensor)
    assert_allclose(t11b.todense(), btensor2d.dtensor)

def test_reorder(btensor2d):
    print('Testing reordering!')
    t1 = btensor2d.dtensor.b_reorder()
    t2 = btensor2d.btensor.b_reorder()
    assert_allclose(t1, t2.todense())
    assert_(
        all([lb1.bm == lb2.bm for lb1, lb2 in zip(t1.labels, t2.labels)]))
    assert_(check_validity_tensor(t1))
    assert_(check_validity_tensor(t2))

def btensor2d_all():
    self = btensor2d()
    test_bdiag(self)
    test_sum(self)
    test_mul(self)
    test_data_parsing(self)
    test_mul_axis(self)
    test_chorder(self)
    test_take(self)
    test_reorder(self)
    test_merge_split(self)


@pytest.fixture
def btensor3d():
    self = nullobj()
    # make data that sparse enough
    mat = array([[[-3, -3, 0, 0],
                  [0, 0, 1, 2]],
                 [[4, 4, 0, 0],
                  [4, 4, 0, 0]]])
    bm1 = BlockMarker([0, 1, 2], [[0], [1]])
    bm2 = BlockMarker([0, 1, 2], [[1], [-1]])
    bm3 = BlockMarker([0, 2, 4], [[-1], [0]])
    labels = [BLabel('a', bm1), BLabel('b', bm2), BLabel('c', bm3)]
    self.dtensor = Tensor(mat, labels=labels)
    data = {(0, 0, 0): mat[:1, :1, :2], (0, 1, 1): mat[:1, 1:2, 2:],
            (1, 0, 0): mat[1:, :1, :2], (1, 1, 0): mat[1:, 1:, :2]}
    self.btensor = BTensor(data, labels)
    assert_(check_validity_tensor(self.btensor))
    assert_(check_validity_tensor(self.dtensor))

    mat = array([[[-3, -3, 0, 0]],
                 [[4, 4, 0, 0]]])
    bm1 = BlockMarker([0, 1, 2], [[3], [1]])
    bm2 = BlockMarker([0, 1], [[3]])
    bm3 = BlockMarker([0, 2, 4], [[1], [0]])
    labels = [BLabel('a', bm1), BLabel('d', bm2), BLabel('c', bm3)]
    self.dtensor2 = Tensor(mat, labels=labels)
    data = {(0, 0, 0): mat[:1, :, :2], (1, 0, 0): mat[1:, :, :2]}
    self.btensor2 = BTensor(data, labels)
    assert_(check_validity_tensor(self.btensor2))
    assert_(check_validity_tensor(self.dtensor2))
    return self

def test_mul3(btensor3d):
    print('Test for multiplication')
    dt = btensor3d.dtensor * btensor3d.dtensor2
    bt = btensor3d.btensor * btensor3d.btensor2
    assert_allclose(dt, bt.todense())
    assert_(all(dt.labels[i] == bt.labels[i]
                for i in range(len(dt.labels))))

def test_data_parsing3(btensor3d):
    print('test for data parsing')
    assert_allclose(btensor3d.dtensor, btensor3d.dtensor.tobtensor().todense())
    assert_allclose(btensor3d.dtensor, btensor3d.btensor.todense())
    print((btensor3d.btensor))

def test_sum3(btensor3d):
    print('Test sum of tensor.')
    assert_(btensor3d.btensor.sum() == btensor3d.dtensor.sum())
    assert_allclose(btensor3d.btensor.sum(axis=-1).todense(),
                    btensor3d.dtensor.sum(axis=-1))
    assert_allclose(btensor3d.btensor.sum(axis=(-2, -1)).todense(),
                    btensor3d.dtensor.sum(axis=(-2, -1)))

def test_mul_axis3(btensor3d):
    print('Testing multiplying 1D array to specific axis.')
    for axis in [1, -1]:
        arr = random.random(btensor3d.dtensor.shape[axis])
        assert_allclose(btensor3d.btensor.mul_axis(
            arr, axis=axis).todense(), btensor3d.dtensor.mul_axis(arr, axis=axis))

def test_take3(btensor3d):
    print('Testing taking axes!')
    axis = 2
    i = 1
    axis_label = btensor3d.btensor.labels[axis]
    t1 = btensor3d.dtensor.take(i, axis=axis)
    # t2=btensor3d.btensor.take(i,axis=axis)
    # assert_allclose(t1,t2.todense())
    t11 = btensor3d.dtensor.take(i, axis=axis_label)
    t22 = btensor3d.btensor.take(i, axis=axis_label)
    assert_allclose(t11, t22.todense())
    assert_allclose(t1, t11)
    print('Testing taking block - axes!')
    ib, axis = 0, 2
    res = [[[-3, -3], [0, 0]], [[4, 4], [4, 4]]]
    assert_allclose(btensor3d.dtensor.take_b(ib, axis=axis), res)
    assert_allclose(btensor3d.btensor.take_b(ib, axis=axis).todense(), res)

def test_chorder3(btensor3d):
    print('Testing chorder!')
    norder = [2, 1, 0]
    labels = [btensor3d.dtensor.labels[i] for i in norder]
    t1 = btensor3d.dtensor.chorder(norder)
    t2 = btensor3d.btensor.chorder(norder)
    assert_allclose(t1, t2.todense())
    t11 = btensor3d.dtensor.chorder(labels)
    # t22=btensor3d.btensor.chorder(labels)
    # assert_allclose(t11,t22.todense())
    assert_allclose(t11, t1)

def test_merge_split3(btensor3d):
    print('Testing merge!')
    axes = slice(1, 3)
    lbs = btensor3d.dtensor.labels[axes]
    t1 = btensor3d.btensor.merge_axes(axes)
    t2 = btensor3d.dtensor.merge_axes(axes)
    assert_allclose(t2, t1.todense())
    print('Testing split!')
    t11 = t1.split_axis(axis=axes.start, nlabels=lbs)
    # [lb.chbm(lb.bm.inflate()) if i!=len(lbs)-1 else lb for i,lb in enumerate(lbs)])
    t11b = t1.split_axis_b(axis=axes.start, nlabels=lbs)
    t22 = t2.split_axis(axis=axes.start, nlabels=lbs)
    assert_(all([lb1 == lb2 for lb1, lb2 in zip(
        t11.labels, btensor3d.dtensor.labels)]))
    assert_(all([lb1 == lb2 for lb1, lb2 in zip(
        t11b.labels, btensor3d.dtensor.labels)]))
    assert_(all([lb1 == lb2 for lb1, lb2 in zip(
        t22.labels, btensor3d.dtensor.labels)]))
    assert_allclose(t22, btensor3d.dtensor)
    assert_allclose(t11.todense(), btensor3d.dtensor)
    assert_allclose(t11b.todense(), btensor3d.dtensor)

def test_reorder3(btensor3d):
    print('Testing reordering!')
    t1 = btensor3d.dtensor2.b_reorder()
    t2 = btensor3d.btensor2.b_reorder()
    assert_allclose(t1, t2.todense())
    assert_(
        all([lb1.bm == lb2.bm for lb1, lb2 in zip(t1.labels, t2.labels)]))
    assert_(check_validity_tensor(t1))
    assert_(check_validity_tensor(t2))

def test_bdiag3(btensor3d):
    print('Test for tensor_block_diag.')
    for axes in [(-2,), (1, 2)]:
        dres = tensor_block_diag([btensor3d.dtensor, btensor3d.dtensor2], axes=axes)
        bres = tensor_block_diag([btensor3d.btensor, btensor3d.btensor2], axes=axes)
        assert_(
            all([lb1.bm == lb2.bm for lb1, lb2 in zip(dres.labels, bres.labels)]))
        assert_allclose(dres, bres.todense())

def test_bother3(btensor3d):
    print('Test abs for btensor.')
    assert_allclose(abs(btensor3d.btensor).todense(), abs(btensor3d.dtensor))
    print('Test power for btensor.')
    assert_allclose((btensor3d.btensor**3).todense(), btensor3d.dtensor**3)

def btensor3d_all():
    self = btensor3d()
    test_sum3(self)
    test_mul3(self)
    test_data_parsing3(self)
    test_mul_axis3(self)
    test_chorder3(self)
    test_take3(self)
    test_merge_split3(self)
    test_reorder3(self)
    test_bother3(self)


def test_tensor():
    a = random.random((210, 30, 70, 10))
    ts1 = Tensor(a, labels=['s1', 's5', 'a2', 'bb'])
    ts2 = Tensor((210, 10, 30), ['s1', 'a1', 's5'])
    ts2[:] = random.random((210, 10, 30))

    print('testing for tensor dot')
    t0 = time.time()
    res1 = contract(ts1, ts2)
    t1 = time.time()
    res2 = tdot(ts1, ts2)
    t2 = time.time()
    assert_allclose(res1, res2)
    assert_(res1.labels == res2.labels)
    print(('Elapse -> Einsum %s, Tensordot %s' % (t1 - t0, t2 - t1)))


def test_btensor():
    print('Testing Blocked dense Tensor.')
    print('test for generation.')
    spaceconfig = SuperSpaceConfig([1, 2, 1])
    bmg = SimpleBMG(spaceconfig=spaceconfig, qstring='QM')
    bms1 = [bmg.random_bm(nsite=i) for i in [3, 1, 3]]
    bms2 = [bmg.random_bm(nsite=i) for i in [1, 3, 4]]
    ts1 = random_btensor(bms=bms1, label_strs=['B', 'A', 'K'])
    ts2 = random_btensor(bms=bms2, label_strs=['A', 'B', 'C'])
    print('test for display')
    print((ts1, [ts1, ts2]))

    print('test for __mul__')
    print('Test for multiplication.')
    res = ts1 * ts2
    assert_(res.labels == ['K', 'C'])
    print('test for make_copy')
    ts11 = ts1.make_copy(copydata=False)
    ts111 = ts11.make_copy(copydata=True, labels=['a', 'b', 'c'])
    ts11.labels[0] = 'xxoo'
    assert_(ts1.labels[0] != 'xxoo')
    ts1[0, 0, 0] = 15
    assert_(ts11[0, 0, 0] == 15 and ts111[0, 0, 0] != 15)
    print('Test conjugate.')
    ts1111 = ts1.conj()
    ts1111.labels[2] = 'love'
    assert_(ts1.labels[2] != 'love')
    print('test for chorder')
    tsc = ts1.chorder([2, 1, 0])
    assert_(all([lb1 == lb2 for lb1, lb2 in zip(tsc.labels, ts1.labels[::-1])]))
    print('test for query_block')
    bs = (2, 0, 7)
    blk = ts1.get_block(bs)
    assert_allclose(blk.shape, [lb.bm.blocksize(ib)
                                for ib, lb in zip(bs, ts1.labels)])
    # test merge axes
    spaceconfig = SuperSpaceConfig([1, 2, 1])
    bmg = SimpleBMG(spaceconfig=spaceconfig, qstring='QM')
    tsm = ts1.merge_axes(sls=slice(1, 3), bmg=bmg)
    assert_((tsm.labels[0].bm.N, tsm.labels[1].bm.N) == tsm.shape)
    assert_allclose(tsm, ts1.reshape([ts1.shape[0], -1]))


def test_svdbd():
    '''Test for function svdbd'''
    print('Test svdbd')
    spaceconfig = SuperSpaceConfig([1, 2, 1])
    bmg = SimpleBMG(spaceconfig=spaceconfig, qstring='QM')
    bms = [bmg.random_bm(nsite=i) for i in [3, 4]]
    ts = random_btensor(bms=bms, label_strs=['B', 'A'])
    # make it block diagonal
    ts[...] = 0
    bm1, bm2 = ts.labels[0].bm, ts.labels[1].bm
    qns1 = ts.labels[0].bm.qns
    qns2 = ts.labels[1].bm.qns
    for qn in qns1:
        if any(all(qns2 == qn, axis=1)):
            cell = ts[tuple([bm1.get_slice(bm1.index_qn(qn).item()),
                             bm2.get_slice(bm2.index_qn(qn).item())])]
            cell[...] = random.random(cell.shape)
    U, S, V = svdbd(ts)
    assert_allclose(ts, (U * S).dot(V))
    U2, S2, V2 = svdbd(ts.tobtensor())
    assert_allclose(ts, ((U2.mul_axis(S2, -1)) * V2).todense())


def test_tensor_svd():
    kernel_list = ['svd', 'ldu', 'dpl_r', 'dpl_c']
    for kernel in kernel_list:
        print(('Test @Tensor.svd, using kernel %s' % kernel))
        spaceconfig = SpinSpaceConfig([1, 2])
        bmg = SimpleBMG(spaceconfig=spaceconfig, qstring='M')
        bm1 = bmg.bm1_  # [1, -1]
        bm2 = BlockMarker(qns=array([[0], [2]], dtype=bm1.qns.dtype), Nr=[0, 2, 3])
        t = Tensor([[[0, 0, -1], [3, 2, 0]], [[4, 9, 0], [0, 0, 0]]],
                   labels=[BLabel('X', bm1), BLabel('Y', bm1), BLabel('Z', bm2)])
        assert_(is_zero_flux(t, signs=[-1, -1, 1], bmg=bmg))
        for cbond in [1, 2]:
            print(('cbond = ', cbond))
            data = t.svd(cbond=1, cbond_str='C',
                         signs=[-1, -1, 1], bmg=bmg, kernel=kernel)
            if kernel == 'svd' or kernel == 'ldu':
                U = data[0].mul_axis(data[1], cbond)
            else:
                U = data[0]
            assert_allclose(t - U * data[-1], 0, atol=1e-8)


def test_tensor_eigh():
    kernel = 'eigh'
    print(('Test @Tensor.svd, using kernel %s' % kernel))
    spaceconfig = SpinSpaceConfig([1, 2])
    bmg = SimpleBMG(spaceconfig=spaceconfig, qstring='M')
    bm1 = bmg.bm1_  # [1, -1]
    bm2 = BlockMarker(qns=array([[-2], [0], [2]], dtype=bm1.qns.dtype), Nr=[0, 1, 3, 4])
    t = Tensor([[[2, 0, 0, 0], [0, 1, -1, 0]], [[0, -1, 1, 0], [0, 0, 0, -3]]],
               labels=[BLabel('X', bm1), BLabel('Y', bm1), BLabel('Z', bm2)])
    assert_(is_zero_flux(t, signs=[-1, -1, -1], bmg=bmg))
    cbond = 2
    print(('cbond = ', cbond))
    data = t.svd(cbond=2, cbond_str='C',
                 signs=[-1, -1, -1], bmg=bmg, kernel=kernel)
    U = data[0].mul_axis(data[1], cbond)
    assert_allclose(t - U * data[-1], 0, atol=1e-8)

def test_nzblocks():
    spaceconfig = SuperSpaceConfig([1, 2, 1])
    print('Testing nz_blocks')
    bmg = SimpleBMG(spaceconfig=spaceconfig, qstring='QM')
    bm1 = bmg.bm1_  # [1, -1]
    bm2 = bmg.join_bms([bm1, bm1], signs=[1,1]).sort().compact_form()
    bms = [bm1, bm2, bm1]
    signs = [1,-1,1]
    nzblocks = zero_flux_blocks([bm.qns for bm in bms], signs, bmg)
    assert_(len(nzblocks[0])==bm1.nblock**2)
    for b in zip(*nzblocks):
        assert_allclose(sum([bm.qns[ind]*s for bm, ind, s in zip(bms, b, signs)], axis=0), 0)


def test_random_zeroflux():
    info = {}
    signs = [1,1,-1]
    print('Testing generating random zero-flux tensor.')
    ts = random_zeroflux_tensor([1,2,3], signs=signs, info=info)
    assert_(is_zero_flux(ts, signs=signs, bmg=info['bmg']))


def test_btdot():
    sign1 = [1,1,-1]
    sign2 = [1,-1,1]
    info = {}
    print('Testing btdot')
    ts1 = random_zeroflux_tensor([1,2,3], signs=sign1, info=info)
    ts2 = random_zeroflux_tensor([ts1.labels[2].bm,4,ts1.labels[0].bm], signs=sign2, info=info)
    ts2.chlabel('o_2', 1)
    ts2.chlabel(ts1.labels[0], 2)
    ts2.chlabel(ts1.labels[2], 0)
    ts = btdot(ts1, ts2, sign1, sign2, info['bmg'])
    ts_t = tdot(ts1, ts2)
    assert_allclose(ts, ts_t)

def test_blabel():
    print('test BLabel. chbm,chstr')
    bl = BLabel('x', BlockMarker(Nr=[0, 1, 3]))
    bl1 = bl.chstr('y')
    bl2 = bl.chbm(BlockMarker(Nr=[0, 1, 4]))
    assert_(bl1 == 'y' and bl2 == 'x')
    assert_(bl1.bm is bl.bm and not (bl2.bm is bl1.bm))

    print('test pickle BLabel')
    fn = 'test_save_blabel.dat'
    quicksave(fn, [bl])
    bl3 = quickload(fn)[0]
    assert_(bl == bl3 and bl.bm == bl3.bm)

    print('test copy BLabel')
    import copy
    bl3 = copy.copy(bl)
    assert_(bl == bl3 and bl.bm == bl3.bm)
    bl3 = copy.deepcopy(bl)
    assert_(bl == bl3 and bl.bm == bl3.bm)


if __name__ == '__main__':
    test_btdot()
    test_nzblocks()
    test_tensor_svd()
    test_tensor_eigh()
    btensor2d_all()
    btensor3d_all()
    test_btensor()
    test_blabel()
    test_tensor()
    test_svdbd()
    test_random_zeroflux()
