



import numpy as np
from pathlib import Path
from tqdm import tqdm

from sklearn.utils.random import sample_without_replacement
import scipy


################" sc tranform ####################
import os
from tempfile import mkdtemp, TemporaryDirectory
import pyarrow
from packaging import version
import pandas as pd
import subprocess
import sys
import time
import tifffile
from skimage.measure import regionprops

from skimage import measure
#from utils.seg_preprocessing import generate_dico_centroid


def sctransform_from_parameters(fit_params, array_of_vect):

    """
    inspire from the init files of ssam, compute sc transform from th...
    :param fit_params: directly from the r script (gene, 3)
    :param array_of_vect: array shape (nb_of_vect, nb_of_genes)
    :return: the normalize version of array_of_vect
    """

    fit_params = np.array(fit_params).T
    nvec = array_of_vect.shape[0]
    regressor_data = np.ones([nvec, 2])
    regressor_data[:, 1] = np.log10(np.sum(array_of_vect, axis=1))

    mu = np.exp(np.dot(regressor_data, fit_params[1:, :]))
    with np.errstate(divide='ignore', invalid='ignore'):
        res = (array_of_vect - mu) / np.sqrt(mu + mu ** 2 / fit_params[0, :])

    return res


### input preparation




def expression_correlation_from_anndata(anndata,
                                        selected_genes,
                                        distance="pearson",
                                        sample_size=None,
                                        random_state= 4,
                                        condition=None,):


    if condition is not None:
        anndata = anndata[anndata.obs.condition== condition]

    if sample_size is not None:
        random_index = sample_without_replacement(len(anndata),
                                                  sample_size,
                                                  random_state=random_state)
    else:
        random_index = np.arange(len(anndata))

    count_matrix = anndata[random_index, selected_genes].X



    dico_proba_edge = {}
    for gene_source in range(len(selected_genes)): # I compute the same thing twice ...
        dico_proba_edge[selected_genes[gene_source]] = {}

    for gene_source in tqdm(range(len(selected_genes))): # I compute the same thing twice ...
        print(gene_source)
        for gene_target in range(gene_source, len(selected_genes)):
            exp_gene_source = count_matrix[:, gene_source]
            exp_gene_target = count_matrix[:, gene_target]
            if distance == "pearson":
                corr = scipy.stats.pearsonr(exp_gene_source, exp_gene_target)[0]
            elif distance == "spearman":
                corr = scipy.stats.spearmanr(exp_gene_source, exp_gene_target)[0]
            else:
                raise Exception(f'distance {distance} not implemented')
            dico_proba_edge[selected_genes[gene_source]][selected_genes[gene_target]] = corr
            dico_proba_edge[selected_genes[gene_target]][selected_genes[gene_source]] = corr
    return dico_proba_edge


def run_sctransform(data, clip_range=None, verbose=True, debug_path=None, plot_model_pars=False, **kwargs):
    """
    this function is inspired of SSAM  https://github.com/HiDiHlabs/ssam
    Run 'sctransform' R package and returns the normalized matrix and the model parameters.
    Package 'feather' is used for the data exchange between R and Python.
    :param data: N x D ndarray to normlize (N is number of samples, D is number of dimensions).
    :type data: numpy.ndarray
    :param kwargs: Any keyword arguments passed to R function `vst`.
    :returns: A 2-tuple, which contains two pandas.dataframe:
        (1) normalized N x D matrix.
        (2) determined model parameters.
    """

    def _log(m):
        if verbose:
            print(m)

    vst_options = ['%s = "%s"' % (k, v) if type(v) is str else '%s = %s' % (k, v) for k, v in kwargs.items()]
    if len(vst_options) == 0:
        vst_opt_str = ''
    else:
        vst_opt_str = ', ' + ', '.join(vst_options)
    with TemporaryDirectory() as tmpdirname:
        if debug_path:
            tmpdirname = debug_path
        ifn, ofn, pfn, rfn = [os.path.join(tmpdirname, e) for e in
                              ["in.feather", "out.feather", "fit_params.feather", "script.R"]]
        _log("Writing temporary files...")
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data, columns=[str(e) for e in range(data.shape[1])])
        if version.parse(pyarrow.__version__) >= version.parse("1.0.0"):
            df.to_feather(ifn, version=1)
        else:
            df.to_feather(ifn)
        rcmd = 'library(feather); library(sctransform);  library(arrow);  mat <- t(as.matrix(arrow::read_feather("{0}"))); colnames(mat) <- 1:ncol(mat); res <- vst(mat{1}, return_gene_attr=TRUE, return_cell_attr=TRUE); write_feather(as.data.frame(t(res$y)), "{2}"); write_feather(as.data.frame(res$model_pars_fit), "{3}");'.format(
            ifn, vst_opt_str, ofn, pfn)
        if plot_model_pars:
            plot_path = os.path.join(tmpdirname, 'model_pars.png')
            rcmd += 'png(file="%s", width=3600, height=1200, res=300); plot_model_pars(res, show_var=TRUE); dev.off();' % plot_path
        rcmd = rcmd.replace('\\', '\\\\')
        with open(rfn, "w") as f:
            f.write(rcmd)
        _log("Running scTransform via Rscript...")
        proc = subprocess.Popen(["Rscript", rfn], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while not proc.poll():
            c = proc.stdout.read(1)
            if not c:
                break
            if verbose:
                try:
                    sys.stdout.write(c.decode("utf-8"))
                except:
                    pass
            time.sleep(0.0001)
        _log("Reading output files...")
        o, p = pd.read_feather(ofn), pd.read_feather(pfn)
        if plot_model_pars:
            try:
                from matplotlib.image import imread
                import matplotlib.pyplot as plt
                img = imread(plot_path)
                dpi = 80
                fig = plt.figure(figsize=(img.shape[1] / dpi, img.shape[0] / dpi), dpi=dpi)
                plt.imshow(img, interpolation='nearest')
                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.show()
            except:
                print("Warning: plotting failed, perhaps matplotlib is not available?")
        _log("Clipping residuals...")
        if clip_range is None:
            r = np.sqrt(data.shape[0] / 30.0)
            clip_range = (-r, r)
        o.clip(*clip_range)
        return o, p


def select_genes_for_sct(vec = None,
                 genes = None,
                 min_expr = 0.01,
                 min_cell = 5):
    """
    Select the gene where it is possible to apply sctransform
    default value from original vst :https://github.com/satijalab/sctransform/blob/master/R/vst.R
    :param vec:
    :param analysis: need only if vec is None
    :param genes:
    :param min_expr: default 0.01
    :param min_cell: default 5
    :return:
    """
    if type(vec) != type(np.array([])):
        vec = vec.toarray() #sparse object for ex
    bool_index  = np.sum(vec > min_expr, axis=0) >= min_cell
    if bool_index.ndim == 2:
        bool_index = bool_index[0]
    return bool_index, np.array(genes)[bool_index]


def generate_dico_centroid(path_to_mask):
    if 'tif' == str(path_to_mask)[-3:]:
        seg_mask = tifffile.imread(path_to_mask)
    else:
        seg_mask = np.load(path_to_mask)
    seg_mask = seg_mask.astype('int')
    dico_centroid = {}
    props = regionprops(seg_mask)
    for pp in props:
        if len(pp.centroid) == 2:
            dico_centroid[pp.label] = [[0, pp.centroid[0], pp.centroid[1]]]
        else:
            assert len(pp.centroid) == 3
            dico_centroid[pp.label] = [pp.centroid]
    return dico_centroid



def compute_dict_centroid(mask_nuclei,  background=0):
    dict_centroid = {}
    #nuclei_labels = measure.label(mask_nuclei, background=background)
    for pp in measure.regionprops(mask_nuclei):
        if len(pp.centroid) == 2:
            dict_centroid[pp.label] = [[0, pp.centroid[0], pp.centroid[1]]]
        else:
            assert len(pp.centroid) == 3
            dict_centroid[pp.label] = pp.centroid
    return dict_centroid
