===================
Tutorial
===================

Using the BlockMarker
---------------------------
First, I will show a rather simple example, it looks like:

.. literalinclude:: ../../sample.py
   :linenos:

Here, the <SimpleBMG> is a subclass of <BlockMarkerGenerator>,
it is a generator class of block markers, covering frequently use U(1) type good quantum numbers,

    * 'Q' - number of electrons, :math:`{n_\uparrow+n_\downarrow}`.
    * 'M' - 2*spin, :math:`{n_\uparrow-n_\downarrow}`.
    * 'P' - parity Q%2, :math:`{(n_\uparrow+n_\downarrow)\%2}`.
    * 'R' - parity M%4, :math:`{(n_\uparrow-n_\downarrow)\%4}`.

Suppose we have an instance *bmg*, we can use *bmg.bm0*, *bmg.bm1_*(*bmg.bm1* for ordered) to generate block marker for 0 or single site.
And *bmg.update1*, *bmg.join_bms* to reach more than 1 sites. The truncation function *trunc_bm* also plays a key role during update.

The <BlockMarker> instance labels the good quantum number of <Tensor>s.
A <BlockMarker> instance has two main attributes *labels* (the good quantum number) and *Nr* (the block pointer),
i.g. use *slice(Nr[i],Nr[i+1])* to get the potion of bond indicated by the i-th good quantum number.
In many cases, we need the good quantum numbers sorted, so that we can get block diagonal matrices.
We call <BlockMarker> instance *bm* with sorted and grouped labels *compact* <BlockMarker>.
We can use method *bm.compact_form* to achive a compact version of *bm*,
together with a permutation matrix that tell you how to permute the bond indices.

