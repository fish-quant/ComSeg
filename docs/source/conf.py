# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'ComSeg'
copyright = '2023, Thomas Defard'
author = 'Thomas Defard'

# The full version, including alpha/beta/rc tags
release = '0.1'

import sys
autodoc_mock_imports = ["scanpy", "pyarrow", "anndata", "pandas", "tqdm",
                        "sklearn", "numpy", "networkx", "anndata", "leidenalg", "scikit-image",
                            "seaborn", "tifffile", "ssam", "scipy", "skimage"]

autodoc_mock_imports += ["anndata","contourpy","cycler","fonttools","h5py","igraph","imageio",
                         "importlib-metadata","importlib-resources","joblib","kiwisolver","lazy_loader","leidenalg",
                         "llvmlite","matplotlib","natsort","networkx","numba","numpy","packaging","pandas","patsy",
                         "Pillow","pyarrow","pynndescent","pyparsing","python-dateutil","python-igraph",
                         "python-louvain","pytz","PyWavelets","scanpy","scikit-image","scikit-learn",
                         "scikit-misc","scipy","seaborn","session-info","six","sparse","ssam",
                         "statsmodels","stdlib-list","tbb","texttable","threadpoolctl","tifffile","tqdm","umap-learn","zipp",]
import os

sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'src')))
sys.path.insert(0, os.path.abspath('../../../'))
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../..'))
sys.path.insert(0, os.path.abspath('../../src/comseg'))
sys.path.insert(0, os.path.abspath('../../src/comseg/utils'))
sys.path.insert(0, os.path.abspath('../../src/comseg/clustering.py'))
sys.path.insert(0, os.path.abspath('../../src/comseg/model.py'))
sys.path.insert(0, os.path.abspath('../../src/comseg/model.py'))


sys.path.insert(0, os.path.abspath('../../src'))
sys.path += ["/home/tom/anaconda3/envs/comseg_v0/lib/python3.8",
'/home/tom/anaconda3/envs/comseg_v0/lib/python3.8/lib-dynload',
'/home/tom/.local/lib/python3.8/site-packages',
'/home/tom/anaconda3/envs/comseg_v0/lib/python3.8/site-packages']
import sphinx_rtd_theme
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    "nbsphinx",
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']