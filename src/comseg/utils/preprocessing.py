



import numpy as np
from pathlib import Path
from tqdm import tqdm

from sklearn.utils.random import sample_without_replacement
import scipy

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






