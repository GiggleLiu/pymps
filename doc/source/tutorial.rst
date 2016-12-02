===================
Tutorial
===================

Using the tensor
---------------------------
First, I will give a brief introduction to tensors,
I will show a rather simple example to explain how it works, it looks like:

.. literalinclude:: ../../mps/sample_tensor.py
   :linenos:

Tensors are always labeled, the labels of a tensor could be a str, or it's subclass <BLabel>,
labels are the clue for any contraction(einsum).
<BLabel> has a member bm(<BlockMarker> instance), when a tensor use <BLabel>s as the labels, we say it is block structured.

The first thing we can do with tensors is to build a MPS.
Here I will show you a piece of code to calculate correlation functions :math:`\langle O(i)O(j)\rangle`.

.. literalinclude:: ../../mps/sample_mps.py
   :linenos:

Each cell of MPS is a 3D tensor, it's axes take the meaning of (llink,site,rlink),
the center axis is the physical site and llink, rlink are the connector to neighboring cells.
The MPS used here is a standard l-canonical one, the l can be changed by canonical move(method <MPS>.canomove).
Compressing can be done through consequtive canonical move with tolerence and maximum bond dimension specified.
On the other side, a lot of utilities can be used to construct operators, 3 elemental types of operators <OpUnit>,<OpString> and <OpCollection> are able to construct a readable form of any operators.
e.g. Sx(0) is an <OpUnit>, Sx(0)*Sx(1)*Sy(3) is an <OpString> and Sx(3)+Sy(2)*Sz(5) is an <OpCollection>, the later the more complex.
function get_expect could be used to cope the expectation problem with all these three.
Besides, we can use <OpCollection>.toMPO method to construct an <MPO> easily, which will be shown below.

.. literalinclude:: ../../mps/sample_bmps.py
   :linenos:

In many cases, the block marker is closely related to good quantum number, in this sample case, it is 'M'.
If we define a direction on each labels(axis) to indicate the flow of good quantum number.
A tensor is 'balanced' if the 'net flow' of quantum number on each non-zero element is 0.
The condition of balance is extremely useful in the definition of MPO and MPS with good quantum number.

In this case, the block markers of cell labels the good quantum number in the left blocks.
If we asign a direction to a cell, it will be (1,1,-1), which means (in,in,out).
We see that all tensors in a <BMPS> or <BMPO> must be balanced! 
This is why we are able to set the block markers automatically with only the knowlege about good quantum number and tensor datas.
