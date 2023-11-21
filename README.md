# ComSeg framework

A detail documentation is available Here https://comseg.readthedocs.io/en/latest/userguide.html


# Single cell spatial RNA profiling 

ComSeg is an algorithm for single cell spatial RNA profiling in image-based transcriptomic data.

It takes as input a csv with the spot coordinates and output an anndata 
object with the  enes expression and coordinates of each cell.

## Installation

First, create a dedicated conda environment using Python 3.8

```bash
conda create -n ComSeg python=3.8
conda activate ComSeg
```

To install the latest github version of this library run the following using pip

```bash
pip install git++https://github.com/tdefa/ComSeg
```

or alternatively you can clone the github repository

```bash
git clone +https://github.com/tdefa/ComSeg
cd cnn_framework
pip install -e .
```

A tutorial notebook can be found here : https://comseg.readthedocs.io/en/latest/userguide.html