

Welcome to ComSeg's documentation!
==================================
ComSeg is an algorithm for single cell spatial RNA profiling for image-based transcriptomic data.

It takes as input  csv files with the spot coordinates and output an anndata object with the genes expression and coordinates of each cell.
It can leverage the information of the cell nuclei to improve the accuracy of the segmentation / RNA profiling.




Getting started
===============

To avoid dependency conflicts, we recommend the the use of a dedicated
`virtual <https://docs.python.org/3.6/library/venv.html>`_ or `conda
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-
environments.html>`_ environment.  In a terminal run the command:

.. code-block:: bash

   $ conda create -n ComSeg_env python=3.8
   $ source activate ComSeg_env
   $ conda install gxx_linux-64 numpy pip R=4.3



Download the package from PyPi
------------------------------

Use the package manager `pip <https://pip.pypa.io/en/stable>`_ to install
ComSeg. In a terminal run the command:

.. code-block:: bash

   $ pip install ComSeg

Clone package from Github
-------------------------

alternatively you can clone the github repository `Github repository <https://github.com/tdefa/ComSeg>`_ and install it manually with the following commands:

.. code-block:: bash

   $ git clone +https://github.com/tdefa/ComSeg_pkg
   $ cd sim-fish
   $ pip install -e .

------------


------------

API reference
*************

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   userguide
   comseg
------------


Index
*************

.. toctree::
   :maxdepth: 2
   :caption: ComSeg:
   index.rst
------------


Support
=======

If you have any question relative to the package, please open an `issue
<https://github.com/tdefa/ComSeg/issues>`_ on Github.

------------

Citation
========

If you exploit this package for your work, please cite:

