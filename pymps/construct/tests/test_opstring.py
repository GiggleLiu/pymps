from __future__ import division
from numpy import *
from numpy.testing import dec, assert_, assert_raises, assert_almost_equal, assert_allclose
from scipy.linalg import expm
import scipy.sparse as sps
import copy
import pdb
from functools import reduce
from tba.hgen import Bilinear, Qlinear, Xlinear, op_c, op_cdag, perm_parity

from ...spaceconfig import *
from ...ansatz.mpo import *
from ...ansatz.mpslib import check_flow_mpx, random_mps
from ..opstring import *
from ..opstringlib import *


def random_fop(spaceconfig, siteindex):
    opc = random.choice([opunit_c, opunit_cdag])
    cop = opc(spaceconfig, random.randint(spaceconfig.ndim))
    assert_(cop.fermionic)
    return cop.as_site(siteindex)


def random_bop(spaceconfig, siteindex):
    ss = ['x', 'y', 'z', '+', '-']
    s = random.choice(ss)
    cop = opunit_S(which=s, spaceconfig=spaceconfig)
    assert_(not cop.fermionic)
    return cop.as_site(siteindex)


def test_I():
    spaceconfig = SuperSpaceConfig([1, 2, 1])
    ou1 = opunit_S(spaceconfig, 'x')
    I = OpUnitI(hndim=spaceconfig.hndim)
    res1 = ou1 * I
    res2 = I * ou1
    res3 = I * I
    assert_allclose(res1.get_data(), res2.get_data())
    assert_(res3.__class__ == OpUnitI)


def test_ss():
    '''test for spin operators'''
    print('Test Fermionic type.')
    scfg = [1, 1, 2, 1]
    spaceconfig = SuperSpaceConfig(scfg)
    ss = ['x', 'y', 'z', '+', '-']
    sdatas = zeros([5, 4, 4], dtype='complex128')
    sdatas[0, 1, 2] = 0.5
    sdatas[0, 2, 1] = 0.5
    sdatas[1, 1, 2] = -0.5j
    sdatas[1, 2, 1] = 0.5j
    sdatas[2, 1, 1] = 0.5
    sdatas[2, 2, 2] = -0.5
    sdatas[3, 1, 2] = 1.
    sdatas[4, 2, 1] = 1.
    for s, sdata in zip(ss, sdatas):
        ou1 = opunit_S(spaceconfig, s)
        assert_allclose(ou1.data, sdata)
    print('Test Bonson type.')
    sdatas = sdatas[:, 1:-1, 1:-1]
    spaceconfig = SpinSpaceConfig([1, 2])
    for s, sdata in zip(ss, sdatas):
        ou1 = opunit_S(spaceconfig, s)
        assert_allclose(ou1.data, sdata)


def test_cs():
    '''test for c operators'''
    scfg = [1, 2, 1]
    spaceconfig = SuperSpaceConfig(scfg)
    cup = opunit_c(spaceconfig, 0)
    cupd = opunit_cdag(spaceconfig, 0)
    cdn = opunit_c(spaceconfig, 1)
    cdnd = opunit_cdag(spaceconfig, 1)
    print('Test n')
    nup = opunit_N(spaceconfig, 0)
    ndn = opunit_N(spaceconfig, 1)
    assert_(nup == cupd * cup)
    assert_(ndn == cdnd * cdn)
    nupdata = zeros([4, 4])
    nupdata[1, 1] = 1
    nupdata[3, 3] = 1
    assert_allclose(nup.get_data(), nupdata)

    ndndata = zeros([4, 4])
    ndndata[2, 2] = 1
    ndndata[3, 3] = 1
    assert_allclose(ndn.get_data(), ndndata)

    print('Test z')
    z = opunit_Z(spaceconfig)
    z2 = expm(1j * pi * (cupd * cup + cdnd * cdn).get_data())
    assert_allclose(z.get_data().diagonal(), [1, -1, -1, 1])
    assert_allclose(z2.diagonal(), [1, -1, -1, 1])

    if False:
        print('Test sp, sm, sy')
        sp = cupd * cdn
        sm = cdnd * cup
        sy = -0.5j * cupd * cdn + 0.5j * cdnd * cup
        ssy = opunit_Sy(spaceconfig)

        ssp = opunit_Sp(spaceconfig)
        ssm = opunit_Sm(spaceconfig)
        assert_(ssp == sp)
        assert_(ssm == sm)


class CastTest(object):
    def test_casting2(self):
        '''test for type casting to opstring, and opcollection.'''
        factor = 2.
        config = [1, 10, 2, 3]
        scfg = SuperSpaceConfig(config)
        config2 = [1, 1, 2, 3]
        scfg2 = SuperSpaceConfig(config2)
        # test for bilinear casting
        c1 = [random.randint(0, config[i]) for i in range(4)]
        c2 = [random.randint(0, config[i]) for i in range(4)]
        c1_ = list(c1)
        c1_[1] = 0
        c2_ = list(c2)
        c2_[1] = 0
        bl = Bilinear(spaceconfig=scfg, index1=scfg.c2ind(c1),
                      index2=scfg.c2ind(c2))
        str1 = xl2string(bl, factor)
        index1 = scfg2.c2ind(c1_)
        index2 = scfg2.c2ind(c2_)
        data1 = op_cdag(scfg2, index1)()
        data2 = op_c(scfg2, index2)()
        if c1[1] > c2[1]:
            data1, data2 = data2, data1
        print('Using Fermions: ', c1, c2)
        if isinstance(str1, OpString):
            print('Get opstring -> ', str1)
            assert_(str1.opunits[0].factor * (1 if c1[-3] <= c2[-3]
                                              else -1) == factor and str1.opunits[1].factor == 1.)
            assert_allclose(str1.opunits[0].data, data1, atol=1e-5)
            assert_allclose(str1.opunits[1].data, data2, atol=1e-5)
        else:
            print('Get opunit -> ', [str1])
            assert_(str1.factor * (1 if c1[-3] <= c2[-3] else -1) == factor)
            assert_allclose(str1.data, data1.dot(data2), atol=1e-5)

    def test_casting4(self):
        '''test for type casting for qlinear'''
        factor = 2.
        config = [1, 5, 2, 3]
        scfg = SuperSpaceConfig(config)
        config2 = [1, 1, 2, 3]
        scfg2 = SuperSpaceConfig(config2)
        # test for bilinear casting
        cs = [[random.randint(0, config[i]) for i in range(4)]
              for i in range(4)]
        cs_ = []
        for ci in cs:
            ci = list(ci)
            ci[1] = 0
            cs_.append(ci)
        bl = Qlinear(spaceconfig=scfg, indices=array(
            [scfg.c2ind(ci) for ci in cs]))
        indices = [scfg2.c2ind(ci) for ci in cs_]
        str1 = xl2string(bl, factor)
        sites = array([c[-3] for c in cs])
        print('Get sites', sites)
        parity = 1 - 2 * (sum([sum(sites[:i] > sites[i])
                               for i in range(1, 4)]) % 2)
        print('parity', parity)
        datas = [op_cdag(scfg2, ind)() if i < 2 else op_c(scfg2, ind)()
                 for i, ind in enumerate(indices)]
        sites = array([ci[1] for ci in cs])
        datas = array(datas)
        print('Using Fermions: ', cs)
        if isinstance(str1, OpString):
            print('Get opstring -> ', str1)
            for i, ou in enumerate(str1.opunits):
                data = datas[where(sites == ou.siteindex)[0]]
                if data.shape[0] > 1:
                    cdata = None
                    for datai in data:
                        if cdata is None:
                            cdata = datai
                        else:
                            cdata = cdata.dot(datai)
                else:
                    cdata = data[0]
                assert_allclose(ou.data, cdata, atol=1e-5)
            assert_(prod([ou.factor for ou in str1.opunits])
                    == factor * parity)
        else:
            print('Get opunit -> ', [str1])
            assert_(str1.factor * parity == factor)
            assert_allclose(str1.data, datas[0].dot(
                datas[1]).dot(datas[2]).dot(datas[3]), atol=1e-5)


def test_insertZ():
    scfg = [1, 2, 1]
    spaceconfig = SuperSpaceConfig(scfg)
    opstring = 0.5 * opunit_C(dag=True, spaceconfig=spaceconfig, index=0, siteindex=2) * \
        opunit_C(dag=False, spaceconfig=spaceconfig, index=1, siteindex=3)
    opstring2 = copy.copy(opstring)
    insert_Zs(opstring2, spaceconfig=spaceconfig)
    print(opstring, opstring2)


def test_oumul():
    scfg = [1, 2, 1]
    nsite = 10
    spaceconfig = SuperSpaceConfig(scfg)
    print('Multiplication of OpUnits')
    sites = arange(nsite)
    fous = [random_fop(spaceconfig, siteindex=site) for site in sites]
    bous = [random_bop(spaceconfig, siteindex=site) for site in sites]
    print('Multiply at same site')
    for oa in [fous[4], bous[4]]:
        for ob in [fous[4], bous[4]]:
            res = oa * ob
            assert_(isinstance(res, OpUnit) and allclose(
                res.get_data(), oa.get_data().dot(ob.get_data())))
    print('Multiply at different site')
    for oa in [fous[4], bous[4]]:
        for ob in [fous[3], fous[5], bous[3], bous[5]]:
            sign = (-1)**(oa.fermionic and ob.fermionic and ob.siteindex < oa.siteindex)
            res = oa * ob
            if oa.siteindex > ob.siteindex:
                assert_(isinstance(res, OpString) and allclose(res.opunits[0].get_data(
                ), sign * ob.get_data()) and allclose(res.opunits[1].get_data(), oa.get_data()))
            else:
                assert_(isinstance(res, OpString) and allclose(res.opunits[0].get_data(
                ), sign * oa.get_data()) and allclose(res.opunits[1].get_data(), ob.get_data()))


def test_osmul(is_fermion):
    nsite = 10
    print('Multiplication of OpStrings.')
    sites = arange(nsite)
    random.shuffle(sites)
    if is_fermion:
        scfg = [1, 2, 1]
        spaceconfig = SuperSpaceConfig(scfg)
        sign1, sign2, sign3 = (-1)**perm_parity(argsort(sites[:nsite // 2])), (-1)**perm_parity(
            argsort(sites[nsite // 2:])), (-1)**perm_parity(sites)
        sign4 = -sign3
        fous = [random_fop(spaceconfig, siteindex=site) for site in sites]
    else:
        scfg = [1, 2]
        spaceconfig = SpinSpaceConfig(scfg)
        sign1, sign2, sign3, sign4 = 1, 1, 1, 1
        fous = [random_bop(spaceconfig, siteindex=site) for site in sites]
    res1 = prod(fous[:nsite // 2])
    res2 = prod(fous[nsite // 2:])
    res3 = res1 * res2
    # res2 if fermionic, so res2*res1 will have a negative sign with sign3
    res4 = OpString(res2.opunits)
    res4 *= res1
    for i, (res, sign) in enumerate(zip([res1, res2, res3, res4], [sign1, sign2, sign3, sign4])):
        # match num of units
        assert_(res.nunit == nsite // 2 if i < 2 else nsite)
        # sites in accending order
        assert_(all(diff(res.siteindices) > 0))
        # data unchanged
        for ou in res.opunits:
            assert_allclose(
                fous[list(sites).index(ou.siteindex)].data, ou.data)
        # sign correct
        assert_almost_equal(prod([ou.factor for ou in res.opunits]), sign)


def test_add():
    scfg = [1, 2, 1]
    nsite = 4
    spaceconfig = SuperSpaceConfig(scfg)
    print('Test Addition of OpUnit,OpString and OpCollection.')
    sites = arange(nsite)
    ous = [random_fop(spaceconfig, siteindex=site) for site in sites]
    oss = [ous[i] * ous[i + 1] for i in range(len(sites) - 1)]
    oc1 = ous[0] + ous[2]
    oc2 = ous[0] + oss[2]
    oc3 = oss[0] + oss[2]
    oc4 = oc1 + oc2 + oc3
    assert_(isinstance(oc1, OpCollection) and isinstance(
        oc2, OpCollection) and isinstance(oc3, OpCollection))
    assert_allclose(oc1.H(nsite=nsite), ous[0].H(
        nsite=nsite) + ous[2].H(nsite=nsite))
    assert_allclose(oc2.H(nsite=nsite), ous[0].H(
        nsite=nsite) + oss[2].H(nsite=nsite))
    assert_allclose(oc3.H(nsite=nsite), oss[0].H(
        nsite=nsite) + oss[2].H(nsite=nsite))
    assert_allclose(oc4.H(nsite=nsite), oc1.H(nsite=nsite) +
                    oc2.H(nsite=nsite) + oc3.H(nsite=nsite))
    print('Test Combined Addition and Multiplication of OpCollection.')
    assert_allclose((oc1 * oc3).H(nsite), (oc1.H(nsite).dot(oc3.H(nsite))))
    oc2 = oc1 * ous[1]
    assert_allclose((oc1 * oc2 + oc2 * oc3).H(nsite),
                    (oc1.H(nsite).dot(oc2.H(nsite)) + oc2.H(nsite).dot(oc3.H(nsite))))


def test_6site():
    tl = 0.5**arange(5)
    el = zeros(6)
    el[0] = -0.1

    # create operator units
    spaceconfig = SuperSpaceConfig([2, 1, 1])
    cup = opunit_c(spaceconfig, 0)
    cupd = opunit_cdag(spaceconfig, 0)
    cdn = opunit_c(spaceconfig, 1)
    cdnd = opunit_cdag(spaceconfig, 1)
    nup = opunit_N(spaceconfig, 0)
    ndn = opunit_N(spaceconfig, 1)

    opc = reduce(lambda x, y: x + y, [ti * (cupd.as_site(i) * cup.as_site(i + 1) + cupd.as_site(i + 1) * cup.as_site(i)
                                            + cdnd.as_site(i) * cdn.as_site(i + 1) + cdnd.as_site(i + 1) * cdn.as_site(i)) for i, ti in enumerate(tl)])
    opc += reduce(lambda x, y: x + y,
                  [ei * nup.as_site(i) + ei * ndn.as_site(i) for i, ei in enumerate(el)])
    insert_Zs(opc, spaceconfig=spaceconfig)
    H = opc.H(nsite=6, dense=False)
    print('Non-Iteracting EG = %s' % sps.linalg.eigsh(H, k=1, which='SA')[0])

    # add hubbard interaction
    U = 0.2
    opc += U * nup.as_site(0) * ndn.as_site(0)
    H = opc.H(nsite=6, dense=False)
    print('Iteracting EG = %s' % sps.linalg.eigsh(H, k=1, which='SA')[0])

if __name__ == '__main__':
    test_6site()
    test_cs()
    # test_ss()
    # test_I()
    test_insertZ()
    test_add()
    # test_osmul(True)
    test_osmul(False)
    # test_oumul()
    CastTest().test_casting2()
    CastTest().test_casting4()
