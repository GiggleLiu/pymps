======================
API
======================

Tensor
------------------------------
.. automodule:: tensor
.. autoclass:: tensor.Tensor
    :show-inheritance:
    :members: make_copy,chorder,take,mul_axis,merge_axes,split_axis,get_block,b_reorder
.. autoclass:: tensor.BLabel
    :show-inheritance:
    :members: chbm,chstr

Tensor Related Functions
------------------------------
.. autofunction:: tensor.tdot
.. autofunction:: tensorlib.random_tensor
.. autofunction:: tensorlib.random_btensor
.. autofunction:: tensorlib.random_bdmatrix
.. autofunction:: tensorlib.check_validity_tensor
.. autofunction:: tensorlib.contract
.. autofunction:: tensorlib.svdbd

Matrix Product State
------------------------------

.. automodule:: mps

.. autoclass:: mps.MPS
    :show-inheritance:
    :inherited-members: link_axis,rlink_axis,site_axis
    :members: hndim,nsite,l,state,get,set,get_ML,check_link,toket,tobra,tovidal,use_bm,show,query,chlabel,recanonicalize,compress

.. autoclass:: mps.BMPS
    :show-inheritance:
    :members: unuse_bm

MPS Related Functions
------------------------------
.. autofunction:: mpslib.state2MPS
.. autofunction:: mpslib.mps_sum
.. autofunction:: mpslib.random_mps
.. autofunction:: mpslib.random_bmps
.. autofunction:: mpslib.product_state
.. autofunction:: mpslib.random_product_state

.. autofunction:: mpslib.check_validity_mps
.. autofunction:: mpslib.check_flow_mpx
.. autofunction:: mpslib.check_canonical

Operator Representations
----------------------------
.. automodule:: mpo

.. autoclass:: mpo.OpUnit
    :show-inheritance:
    :members: hndim,siteindices,maxsite,as_site,H,get_mathstr,get_data,toMPO

.. autoclass:: mpo.OpUnitI
    :show-inheritance:

.. autoclass:: mpo.OpString
    :show-inheritance:
    :members: get_mathstr,siteindices,maxsite,H,toMPO,query

.. autoclass:: mpo.OpCollection
    :show-inheritance:
    :members: maxsite,H,toMPO,query,filter

.. autoclass:: mpo.MPO
    :show-inheritance:
    :members: hndim,nsite,H,get,set,check_link,use_bm,chlabel,compress

.. autoclass:: mpo.BMPO
    :show-inheritance:
    :members: unuse_bm

MPO Related Functions
------------------------------
.. autofunction:: mpo.WL2MPO
.. autofunction:: mpo.WL2OPC
.. autofunction:: mpolib.opunit_S
.. autofunction:: mpolib.opunit_Sx
.. autofunction:: mpolib.opunit_Sy
.. autofunction:: mpolib.opunit_Sz
.. autofunction:: mpolib.opunit_Sp
.. autofunction:: mpolib.opunit_Sm
.. autofunction:: mpolib.opunit_C
.. autofunction:: mpolib.opunit_c
.. autofunction:: mpolib.opunit_Z
.. autofunction:: mpolib.opunit_cdag
.. autofunction:: mpolib.opunit_N

.. autofunction:: mpolib.insert_Zs

.. autofunction:: mpolib.random_mpo
.. autofunction:: mpolib.random_bmpo
.. autofunction:: mpolib.check_validity_mpo
.. autofunction:: mpolib.mpo_sum

Contraction of MPO and MPS
--------------------------------
.. automodule:: contraction
.. autoclass:: contraction.Contractor
    :show-inheritance:
    :members: lupdate,rupdate,contract2l,evaluate,show

.. autofunction:: get_expect
