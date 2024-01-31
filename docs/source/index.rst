.. comseg documentation master file, created by
   sphinx-quickstart on Tue Oct  3 15:19:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ComSeg's documentation!
==================================
ComSeg is an algorithm for single cell spatial RNA profiling for image-based transcriptomic data.

It takes as input  csv files with the spot coordinates and output an anndata object with the genes expression and coordinates of each cell.
It can leverage the information of the cell nuclei to improve the accuracy of the segmentation / RNA profiling.



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

If you have any question relative to the package, please open an `issue
<https://github.com/tdefa/ComSeg/issues>`_ on Github.

------------

Citation
========

If you exploit this package for your work, please cite:

A point cloud segmentation framework for image-based spatial transcriptomic,
Thomas Defard, Hugo Laporte, Mallick Ayan, Soulier Juliette, Sandra Curras-Alonso, Christian Weber, Florian Massip, José-Arturo Londoño-Vallejo, Charles Fouillade, Florian Mueller, Thomas Walter
bioRxiv 2023.12.01.569528; doi: https://doi.org/10.1101/2023.12.01.569528 

