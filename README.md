#ComSeg framework


# Single cell spatial RNA profiling 

ComSeg is an algorithm for single cell spatial RNA profiling in image-based transcriptomic data.

It takes as input a csv with the spot coordinates and output an anndata object with the  enes expression ond coordinates of each cell.

## Installation

First, create a dedicated conda environment using Python 3.8

```bash
conda create -n ComSeg python=3.8
conda activate ComSeg
```

To install the latest github version of this library run the following using pip

```bash
pip install git++https://github.com/tdefa/ComSeg_pkg
```

or alternatively you can clone the github repository

```bash
git clone +https://github.com/tdefa/ComSeg_pkg
cd cnn_framework
pip install -e .
```

this repository contains a tutorail and a test dataset can be find here : todo