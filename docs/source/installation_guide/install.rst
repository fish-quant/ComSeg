


Getting started
===============

To avoid dependency conflicts, we recommend the the use of a dedicated
`virtual <https://docs.python.org/3.6/library/venv.html>`_ or `conda
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-
environments.html>`_ environment.
In a terminal run the command:

.. code-block:: bash

   $ conda create -n ComSeg_env python=3.8
   $ conda activate ComSeg_env
   $ conda install R=4.3

(Optional) If you want to use a normalization step normalization :  then in ``R`` do

.. code-block:: bash

   install.packages("sctransform")
   install.packages("feather")
   install.packages("arrow")

Alternatively, you can install arrow from conda-forge:

.. code-block:: bash

   conda install -c conda-forge r-arrow
   # or conda install -c conda-forge --strict-channel-priority r-arrow

ComSeg optionally uses sctransform to normalize the expression of "RNA communities", if you want to use
this normalization step,
you need to install the R packages. The normalization with sctransform makes the segmentation better.





Download the package from PyPi
------------------------------

Use the package manager `pip <https://pip.pypa.io/en/stable>`_ to install
ComSeg. In a terminal run the command:

.. code-block:: bash

   $ pip install comseg

Clone package from Github
-------------------------

alternatively you can clone the github repository `Github repository <https://github.com/tdefa/ComSeg>`_ and install it manually with the following commands:

.. code-block:: bash

   $ git clone +https://github.com/tdefa/ComSeg_pkg
   $ cd ComSeg_pkg
   $ pip install -e . -r requirements.txt

for this local install if import comseg as follows:

.. code-block:: bash

   import src.comseg as comseg

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
