'''
Library for MPOs
'''
from numpy import *
import time,pdb,copy
import numpy as np

from tba.hgen import op_from_mats,op_c,op_cdag,op_simple_onsite
from tba.hgen import SuperSpaceConfig,SpinSpaceConfig
from mpo import OpUnit,OpString,OpCollection,UNSETTLED,BMPO,MPO
from mpo import _mpo_sum
from tensorlib import random_bdmatrix

__all__=['opunit_Sx','opunit_Sy','opunit_Sz','opunit_Sm','opunit_Sp','opunit_S',\
        'opunit_C','opunit_c','opunit_cdag','xl2string','op2collection',\
        'opunit_N','opunit_Z','insert_Zs','random_bmpo','random_mpo','check_validity_mpo',\
        'mpo_sum','check_validity_op']

##################Generation of Special <OpUnit>#########################
def opunit_S(spaceconfig,which,siteindex=UNSETTLED):
    '''
    Get S_? operator unit.

    Parameters:
        spaceconfig: <SuperSpaceConfig>/<SpinSpaceConfig>,  the space configuration.
        which: char, specify the `?`.

    Return:
        <OpUnit>
    '''
    ss=['x','y','z','+','-']
    assert(which in ss)
    index=ss.index(which)
    sfactor=1.
    if isinstance(spaceconfig,SpinSpaceConfig):
        if index<3:
            data=spaceconfig.sigma(index+1)/sfactor
        elif index==3:
            data=(spaceconfig.sigma(1)+1j*spaceconfig.sigma(2))/sfactor
        else:
            data=(spaceconfig.sigma(1)-1j*spaceconfig.sigma(2))/sfactor
    else:
        if index<3:
            data=op_from_mats(label='S?',spaceconfig=spaceconfig,mats=[spaceconfig.sigma(index+1)/sfactor],bonds=None)()
        else:
            datax=op_from_mats(label='Sx',spaceconfig=spaceconfig,mats=[spaceconfig.sigma(1)/sfactor],bonds=None)()
            datay=op_from_mats(label='Sy',spaceconfig=spaceconfig,mats=[spaceconfig.sigma(2)/sfactor],bonds=None)()
            data=(datax+1j*datay) if index==3 else (datax-1j*datay)
    return OpUnit(label='S'+which,data=data,math_str=r'S_{%s}'%which,siteindex=siteindex)

opunit_Sx=lambda spaceconfig:opunit_S(spaceconfig,'x')
opunit_Sy=lambda spaceconfig:opunit_S(spaceconfig,'y')
opunit_Sz=lambda spaceconfig:opunit_S(spaceconfig,'z')
opunit_Sp=lambda spaceconfig:opunit_S(spaceconfig,'+')
opunit_Sm=lambda spaceconfig:opunit_S(spaceconfig,'-')

def opunit_C(spaceconfig,index,dag,siteindex=UNSETTLED):
    '''
    Get creation and annilation operator units.

    Parameters:
        :spaceconfig: <SuperSpaceConfig>/<SpinSpaceConfig>,  the space configuration.
        :index: integer, the flavor of electron.
        :dag: bool, creation or not(annilation).

    Return:
        <OpUnit>
    '''
    if dag:
        data=op_cdag(spaceconfig,index)()
    else:
        data=op_c(spaceconfig,index)()
    res=OpUnit(label='c%s'%index+('dag' if dag else ''),data=data,math_str=r'c_{%s}'%index+(r'^{\dag}' if dag else ''),fermionic=True,siteindex=siteindex)
    return res

opunit_cdag=lambda spaceconfig,index:opunit_C(spaceconfig,index,dag=True,siteindex='-')
opunit_c=lambda spaceconfig,index:opunit_C(spaceconfig,index,dag=False,siteindex='-')

def opunit_N(spaceconfig,index=None,siteindex=UNSETTLED):
    '''
    Get pariticle number operator.

    Parameters:
        :spaceconfig: <SuperSpaceConfig>,  the space configuration.
        :index: int/None, the index of flavor, None for all.

    Return:
        <OpUnit>, the particle number operator.
    '''
    data=op_simple_onsite('n',spaceconfig,index=index)()
    res=OpUnit(label='N',data=data,math_str=r'N',fermionic=False,siteindex=siteindex)
    return res

def opunit_Z(spaceconfig,siteindex=UNSETTLED):
    '''
    Get fermionic parity operator units.

    Parameters:
        :spaceconfig: <SuperSpaceConfig>,  the space configuration.

    Return:
        <OpUnit>, the parity operator.
    '''
    data=op_simple_onsite('n',spaceconfig)()
    fill_diagonal(data,np.round(exp(1j*pi*data.diagonal()).real))
    res=OpUnit(label='Z',data=data,math_str=r'Z',fermionic=False,siteindex=siteindex)
    return res

def xl2string(xl,param=1.):
    '''
    cast x-linear to <OpString>
    
    Parameters:
        :nl: <Bilinear>/<Qlinear>/<Nlinear>,
        :param: weight,
    '''
    nbody=xl.nbody
    units=[]
    scfg=xl.spaceconfig
    atom_axis=scfg.get_axis('atom')
    config=list(scfg.config)
    config[atom_axis]=1
    spaceconfig=SuperSpaceConfig(config[-3:])
    indices=ravel(xl.indices)
    indices_ndag=xl.indices_ndag
    for i in xrange(nbody):
        index=indices[i]
        ci=scfg.ind2c(index)
        siteindex=ci[atom_axis]
        ci[atom_axis]=0
        index=spaceconfig.c2ind(ci)
        ui=opunit_C(spaceconfig=spaceconfig,index=index,dag=True if i<indices_ndag else False)
        ui.siteindex=siteindex
        units.append(ui)
    opstring=complex(param*xl.factor)*prod(units)
    return opstring

def op2collection(op,param=1.):
    '''
    cast operators and specific params to <OpCollection>s
    '''
    opc=[]
    xlinears=op.suboperators
    param=op.factor*param
    for xl in xlinears:
        opc.append(xl2string(xl,param))
    return sum(opc)

def random_bmpo(bmg,nsite=10,maxN=6):
    '''
    Random <BMPS>.

    Parameters:
        :nsite: int, number of sites.
        :bmg: <BlockMarkerGenerator>,
        :maxN: int, the maximum bond dimension.

    Return:
        <BMPO>,
    '''
    hndim=len(bmg.qns1)
    #first generate block markers.
    bmi=bmg.bm0
    bm1=bmg.bm1_
    bms=[bmi]
    OL=[]
    for i in xrange(nsite):
        bmi,pm=bmg.join_bms([bmi,bm1,bm1],signs=[1,1,-1],compact_form=True)
        #create a random block diagonal matrix
        ts=random_bdmatrix(bmi)
        dim=min(maxN,(hndim**2)**(nsite-i-1))
        if bmi.N>dim:
            #do random truncation!
            kpmask=zeros(bmi.N,dtype='bool')
            randarr=arange(bmi.N)
            random.shuffle(randarr)
            kpmask[randarr[:dim]]=True
            ts=ts.take(kpmask,axis=1,useqn=False)
        else:
            kpmask=None
        #unsort left labels, truncate right labels
        ts=ts.take(argsort(pm),axis=0,useqn=False)
        bmi=ts.labels[1].bm
        ts=reshape(ts,[-1,hndim,hndim,ts.shape[-1]])
        bms.append(bmi)
        OL.append(ts)
    mpo=BMPO(OL=OL,bmg=bmg)
    return mpo

def random_mpo(hndim=2,nsite=10,maxN=6,hermitian=True):
    '''
    Random <MPO>.

    Parameters:
        :hndim: int, the single site hilbert space dimension.
        :nsite: int, number of sites.
        :maxN: int, the maximum bond dimension.
        :hermitian: bool, get a hermitian MPO if True.

    Return:
        <MPO>,
    '''
    OL=[]
    rdim=1
    for i in xrange(nsite):
        ldim=rdim
        rdim*=hndim**2
        rdim=min(maxN,(hndim**2)**(nsite-i-1),rdim)
        ts=(random.random([ldim,hndim,hndim,rdim])+0j)/sqrt(ldim*rdim*hndim**2)
        if hermitian:
            ts=ts+transpose(ts,(0,2,1,3)).conj()  #make it hermitian
        OL.append(ts)
    mpo=MPO(OL=OL)
    return mpo

def insert_Zs(op,spaceconfig):
    '''
    Insert fermionic signs between ferminonic operators.
    
    Parameters:
        :spaceconfig: <SpaceConfig>, the configuration of hilbert space.
    '''
    if isinstance(op,OpCollection):
        for opstring in op.ops:
            if isinstance(opstring,OpString):
                insert_Zs(opstring,spaceconfig)
    elif isinstance(op,OpString):
        z0=opunit_Z(spaceconfig=spaceconfig)
        fsites=reshape([ou.siteindex for ou in op.opunits if ou.fermionic],[-1,2])
        for ispan in xrange(len(fsites)):
            for i in xrange(fsites[ispan,0],fsites[ispan,1]):
                op*=z0.as_site(i)

def check_validity_mpo(mpo):
    '''
    check the validity of mpo.

    Parameters:
        :mpo: <MPO>,

    Return:
        bool,
    '''
    valid=True
    #1. the link dimension check
    nsite=mpo.nsite
    llink_axis,s1_axis,s2_axis,rlink_axis=0,1,2,3
    hndim=mpo.hndim
    for i in xrange(nsite-1):
        valid=valid and mpo.get(i+1).shape[llink_axis]==mpo.get(i).shape[rlink_axis]
        if hasattr(mpo,'bmg'):
            valid=valid and mpo.get(i+1).labels[llink_axis].bm==mpo.get(i).labels[rlink_axis].bm

    for i in xrange(nsite):
        cell=mpo.get(i)
        assert(ndim(cell)==4)
        valid=valid and cell.shape[s1_axis]==hndim
        valid=valid and cell.shape[s2_axis]==hndim
        #2. the block marker check
        for i in xrange(3):
            if hasattr(cell.labels[i],'bm'):
                valid=valid and cell.shape[i]==cell.labels[i].bm.N
    return valid

def mpo_sum(mpos,labels=['m','s','b']):
    '''
    Parameters:
        :mpos: list, list of <MPO>s.
        :labels: list, list of string as new labels(site-up,site-down,link).

    Return:
        <MPO>,
    '''
    return _mpo_sum(mpos,labels=labels)

def check_validity_op(op):
    '''
    Check the validity of an operator.

        1. the order of strings.
        2. against fermionic opstring/opunits.
    
    Parameters:
        :op: <OpCollection>/<OpString>/<OpUnit>,
    '''
    if isinstance(op,OpCollection):
        return all([check_validity_op(opi) for opi in op.ops])
    if isinstance(op,OpUnit):
        return not op.fermionic

    #check for opstring.
    return all(diff(op.siteindices)>=0) and not op.fermionic
