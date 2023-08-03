



import anndata as ad
import numpy as np
import ssam
import pandas as pd


"""
In situ clustering class
take as intput a set of comseg instance
compute in_situclustering
label the community
save parameters to labeled other community with in the pca or euclidian space

compute the norm parameter if need
norm expression matrix

"""

__all__ = ["InSituClustering"]

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



class InSituClustering():

    def __init__(self,
                dict_anndata_img,
                selected_genes,
                concatenate_anndata = True,):
        if concatenate_anndata:
            self.anndata = ad.concat([dict_anndata_img[img_name] for img_name in dict_anndata_img], merge="same")
        else:
            self.dict_anndata_img = dict_anndata_img
        self.selected_genes = selected_genes


    def compute_normalization_parameters(self,
                                         debug_path = None):

        bool_index, new_selected_genes = select_genes_for_sct(
            vec=self.anndata.X,
            genes=self.selected_genes,
            min_expr=0.01,
            min_cell=5)

        self.norm_genes = new_selected_genes
        self.norm_bool_index = bool_index

        row_indice = np.nonzero(np.sum(self.anndata.X[:, bool_index], axis=1) > 0)[0]
        row_indice = np.isin(list(range(len(self.anndata))), row_indice)
        count_matrix = self.anndata.X[:, bool_index].toarray()
        count_matrix = count_matrix[row_indice, :]

        print(f'shape count matrix {count_matrix.shape}')
        np.save("count_matrix", count_matrix)
        count_matrix = pd.DataFrame(count_matrix, columns=[str(e) for e in range(count_matrix.shape[1])])
        print(count_matrix.columns)
        norm_expression_vectors, param_sctransform = ssam.run_sctransform(count_matrix,
                                                                          debug_path=debug_path)

        self.param_sctransform = param_sctransform
        self.genes_to_take = bool_index
        return norm_expression_vectors, param_sctransform


    def cluster_rna_community(self,
                              size_commu_min=args.in_situ_size_commu_min_leiden,
                              n_neighbors=args.in_situ_clustering_n_neighbors,
                              n_comps=args.in_situ_n_comps,
                              n_pcs=args.in_situ_n_pcs,
                              resolution=args.in_situ_resolution,
                              clustering_method=clustering_method,
                              new_selected_genes=new_selected_genes,
                              bool_index=bool_index,
                              param_sctransform=param_sctransform,
                              palette=palette,
                              divide_by_norm=args.divide_by_norm,
                              norm_vector=args.norm_vector,
                              vector_key=args.agregation_mode,  # args.agregation_mode,
                              unorm_vector_key="unorm_vector",
                              n_clusters_kmeans=args.n_clusters_kmeans,
                              max_number_image=len(dico_dico_commu),
                              update_graph=True,
                              divide_by_nb_spots=False, ):

        #### select vector with nb of rna superior to size_commu_min

        bool_index = np.array(self.anndata.obs['nb_rna']) > size_commu_min)

        self.anndata = self.anndata[bool_index, self.genes_to_take]

        norm_sctransform_from_parameters(self.param_sctransform ,  self.anndata[bool_index, self.genes_to_take])

        ####





