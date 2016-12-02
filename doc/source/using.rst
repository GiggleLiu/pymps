==============
Using the code
==============

The requirements are:

* `Python <http://www.python.org/>`_ 2.6 or higher
* `numpy <http://www.numpy.org/>`_ and `scipy <http://www.scipy.org/>`_
* `matplotlib <http://www.matplotlib.org/>`_ for plotting

* `tba <https://github.com/GiggleLiu/tba/>`_
* `blockmatrix <https://github.com/GiggleLiu/blockmatrix/>`_

It is recommended to use `Anaconda <https://www.continuum.io/downloads/>`_ to install these packages(except tba and blockmatrix).

Download the code using the `Download ZIP
<https://github.com/GiggleLiu/mpslib/archive/master.zip>`_
button on github, or run the following command from a terminal::

    $ wget -O mpslib-master.zip https://github.com/GiggleLiu/mpslib/archive/master.zip

Within a terminal, execute the following to unpack the code::

    $ unzip mpslib-master.zip
    $ cd mpslib-master/
    $ (sudo) python setup.py install

The first program, for instance, can be run by issuing::

    $ cd mps/
    $ python sample_tensor.py
    $ python sample_mps.py
    $ python sample_bmps.py
