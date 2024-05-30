.. comseg documentation master file, created by
   sphinx-quickstart on Tue Oct  3 15:19:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ComSeg's documentation!
==================================
ComSeg is an algorithm to perform cell segmentation from RNA point clouds
(single cell spatial RNA profiling) from FISH based spatial transcriptomic data

It takes as input csv files with the spot coordinates and either the cell centroids or the nucleus segmentation masks.
it outputs an anndata object with the genes expression and coordinates of each cell.
It can leverage the information of the cell nuclei to improve the accuracy of the segmentation / RNA profiling.

News
=======
- Update with comseg 0.0.8. Compatibility with SOPA : https://gustaveroussy.github.io/sopa/
- Update with comseg 0.0.7. Computation of the co-expression matrix is now faster.

.. image:: comseg.png

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation_guide
   userguide
   comseg
------------

------------

Support
=======

If you have any questions relative to the package, please open an `issue
<https://github.com/tdefa/ComSeg/issues>`_ on Github.

------------

Citation
========

If you use this package for your work, please cite:

A point cloud segmentation framework for image-based spatial transcriptomic,
Thomas Defard, Hugo Laporte, Mallick Ayan, Soulier Juliette, Sandra Curras-Alonso, Christian Weber, Florian Massip, José-Arturo Londoño-Vallejo, Charles Fouillade, Florian Mueller, Thomas Walter
bioRxiv 2023.12.01.569528; doi: https://doi.org/10.1101/2023.12.01.569528 

