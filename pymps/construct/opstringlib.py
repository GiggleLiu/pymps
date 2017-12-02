'''
Library for MPOs
'''
import numpy as np

from ..spaceconfig import SuperSpaceConfig, SpinSpaceConfig
from ..toolbox.spin import s, s1
from .opstring import OpUnit, OpString, OpCollection, UNSETTLED, OpUnitI

__all__ = ['OpUnitI', 'opunit_Sx', 'opunit_Sy', 'opunit_Sz', 'opunit_Sm', 'opunit_Sp', 'opunit_S',
           'opunit_C', 'opunit_c', 'opunit_cdag', 'xl2string', 'op2collection',
           'opunit_N', 'opunit_Z', 'insert_Zs', 'check_validity_op']

##################Generation of Special <OpUnit>#########################


def opunit_S(spaceconfig, which, siteindex=UNSETTLED):
    '''
    Get S_? operator unit.

    Args:
        spaceconfig: <SuperSpaceConfig>/<SpinSpaceConfig>,  the space configuration.
        which: char, specify the `?`.

    Returns:
        <OpUnit>
    '''
    ss = ['x', 'y', 'z', '+', '-']
    assert(which in ss)
    index = ss.index(which)
    sfactor = 1.
    if isinstance(spaceconfig, SpinSpaceConfig):
        if index < 3:
            data = _sigma(spaceconfig, index + 1) / sfactor
        elif index == 3:
            data = (_sigma(spaceconfig, 1) + 1j *
                    _sigma(spaceconfig, 2)) / sfactor
        else:
            data = (_sigma(spaceconfig, 1) - 1j *
                    _sigma(spaceconfig, 2)) / sfactor
    else:
        raise NotImplementedError()
    return OpUnit(label='S' + which, data=data, math_str=r'S_{%s}' % which, siteindex=siteindex)


def opunit_Sx(spaceconfig): return opunit_S(spaceconfig, 'x')


def opunit_Sy(spaceconfig): return opunit_S(spaceconfig, 'y')


def opunit_Sz(spaceconfig): return opunit_S(spaceconfig, 'z')


def opunit_Sp(spaceconfig): return opunit_S(spaceconfig, '+')


def opunit_Sm(spaceconfig): return opunit_S(spaceconfig, '-')


def opunit_C(spaceconfig, index, dag, siteindex=UNSETTLED):
    '''
    Get creation and annilation operator units.

    Args:
        spaceconfig (<SuperSpaceConfig>/<SpinSpaceConfig>):  the space configuration.
        index (integer): the flavor of electron.
        dag (bool): creation or not(annilation).

    Returns:
        <OpUnit>
    '''
    if not isinstance(spaceconfig, SuperSpaceConfig):
        raise

    occ, unocc = [index], []
    if dag:
        occ, unocc = unocc, occ
    # notice that this state is used as index2, and the reverse is index1
    index2, info = spaceconfig.indices_occ(
        occ=occ, unocc=unocc, return_info=True)
    count_e, index1 = info['e_between'], info['rindex']

    # coping the fermionic sign
    sparam = np.ones(len(index1))
    # index passes eletrons [0,index)
    # the sign is equal to the electrons site<index
    sparam[count_e[0] % 2 == 1] *= -1
    data = np.zeros(shape=[spaceconfig.hndim] * 2, dtype='complex128')
    data[index1, index2] = sparam

    res = OpUnit(label='c%s' % index + ('dag' if dag else ''), data=data,
                 math_str=r'c_{%s}' % index + (r'^{\dag}' if dag else ''), fermionic=True, siteindex=siteindex)
    return res


def opunit_cdag(spaceconfig, index):
    return opunit_C(spaceconfig, index, dag=True, siteindex='-')


def opunit_c(spaceconfig, index):
    return opunit_C(spaceconfig, index, dag=False, siteindex='-')


def opunit_N(spaceconfig, index=None, siteindex=UNSETTLED):
    '''
    Get pariticle number operator.

    Args:
        spaceconfig (<SuperSpaceConfig>):  the space configuration.
        index (int/None, the index of flavor): None for all.

    Returns:
        <OpUnit>, the particle number operator.
    '''
    if not isinstance(spaceconfig,SuperSpaceConfig):
        raise
    hndim = spaceconfig.hndim
    configs = spaceconfig.ind2config(np.arange(hndim))

    res = OpUnit(label='N', data=np.diag(configs[:,index]), math_str=r'N',
                 fermionic=False, siteindex=siteindex)
    return res


def opunit_Z(spaceconfig, siteindex=UNSETTLED):
    '''
    Get fermionic parity operator units.

    Args:
        spaceconfig (<SuperSpaceConfig>):  the space configuration.

    Returns:
        <OpUnit>, the parity operator.
    '''
    data_diag = spaceconfig.get_quantum_number('Q')
    data = np.zeros([spaceconfig.hndim]*2, dtype = 'complex128')
    np.fill_diagonal(data, np.round(np.exp(1j * np.pi * data_diag).real))
    res = OpUnit(label='Z', data=data, math_str=r'Z',
                 fermionic=False, siteindex=siteindex)
    return res


def xl2string(xl, param=1.):
    '''
    cast x-linear to <OpString>

    Args:
        nl (<Bilinear>/<Qlinear>/<Nlinear>):
        param (weight):
    '''
    nbody = xl.nbody
    units = []
    scfg = xl.spaceconfig
    atom_axis = scfg.get_axis('atom')
    config = list(scfg.config)
    config[atom_axis] = 1
    spaceconfig = SuperSpaceConfig(config[-3:])
    indices = np.ravel(xl.indices)
    indices_ndag = xl.indices_ndag
    for i in range(nbody):
        index = indices[i]
        ci = scfg.ind2c(index)
        siteindex = ci[atom_axis]
        ci[atom_axis] = 0
        index = spaceconfig.c2ind(ci)
        ui = opunit_C(spaceconfig=spaceconfig, index=index,
                      dag=True if i < indices_ndag else False)
        ui.siteindex = siteindex
        units.append(ui)
    opstring = complex(param * xl.factor) * np.prod(units)
    return opstring


def op2collection(op, param=1.):
    '''
    cast operators and specific params to <OpCollection>s
    '''
    opc = []
    xlinears = op.suboperators
    param = op.factor * param
    for xl in xlinears:
        opc.append(xl2string(xl, param))
    return sum(opc)


def insert_Zs(op, spaceconfig):
    '''
    Insert fermionic signs between ferminonic operators.

    Args:
        spaceconfig (<SpaceConfig>): the configuration of hilbert space.
    '''
    if isinstance(op, OpCollection):
        for opstring in op.ops:
            if isinstance(opstring, OpString):
                insert_Zs(opstring, spaceconfig)
    elif isinstance(op, OpString):
        z0 = opunit_Z(spaceconfig=spaceconfig)
        fsites = np.reshape(
            [ou.siteindex for ou in op.opunits if ou.fermionic], [-1, 2])
        for ispan in range(len(fsites)):
            for i in range(fsites[ispan, 0], fsites[ispan, 1]):
                op *= z0.as_site(i)


def check_validity_op(op):
    '''
    Check the validity of an operator.

        1. the order of strings.
        2. against fermionic opstring/opunits.

    Args:
        op (<OpCollection>/<OpString>/<OpUnit>):
    '''
    if isinstance(op, OpCollection):
        return all([check_validity_op(opi) for opi in op.ops])
    if isinstance(op, OpUnit):
        return not op.fermionic

    # check for opstring.
    return all(np.diff(op.siteindices) >= 0) and not op.fermionic


def _sigma(spaceconfig, index):
    '''
    Pauli operators.
    '''
    if spaceconfig.nspin == 2:
        si = s[index] / 2.
    elif spaceconfig.nspin == 3:
        si = s1[index]
    else:
        raise NotImplementedError()
    natom = spaceconfig.natom
    if natom != 1:
        si = kron(si, identity(natom))
    return si
