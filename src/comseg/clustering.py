

#todo     ### ADD FCT TO MERGE CLUSTER  WITH A PREDEFINE NUMBER OF CLUSTER TO MERGE


import sklearn


import anndata as ad
import numpy as np
import ssam
import pandas as pd
from scipy.sparse import csr_matrix
import scanpy as sc

from .utils.preprocessing import sctransform_from_parameters
import scipy
from matplotlib import pyplot as plt
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
                anndata,
                 selected_genes,
                 normalize = True,):
        self.anndata = anndata  ## contain the expression vector of all commu
        self.selected_genes = selected_genes
        self.normalize = normalize
        self.anndata_cluster = None ## contain the expression vector of the cluster commu ie more than x RNA


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
                              size_commu_min=3,
                              norm_vector = True,

                              ### parameter clustering
                              n_pcs = 15,
                              n_comps =15,
                              clustering_method ="leiden",
                              n_neighbors = 20,
                              resolution = 1,
                              n_clusters_kmeans = 4,
                              palette = None,
                              plot_umap = False):

        #### select vector with nb of rna superior to size_commu_min

        bool_index_row = np.array(self.anndata.obs['nb_rna']) > size_commu_min


        if self.normalize: ## apply sctrasnform
            if not  hasattr(self, "param_sctransform"):
                raise ValueError("You need to compute the normalization parameters with 'compute_normalization_parameters' before clustering ")
            anndata = self.anndata[bool_index_row, self.genes_to_take]
            count_matrix_anndata = sctransform_from_parameters(self.param_sctransform,
                    anndata.X.toarray())
            new_selected_genes = self.norm_genes
        else:
            count_matrix_anndata = self.anndata.X.toarray()
            bool_index_row = [True] * len(self.anndata)
            new_selected_genes = self.selected_genes

        ## norm vector if needed


        adata = ad.AnnData(csr_matrix(count_matrix_anndata))
        adata.var["features"] = new_selected_genes
        adata.var_names = new_selected_genes
        adata.obs["img_name"] = np.array(self.anndata.obs["img_name"])[bool_index_row]
        adata.obs["index_commu"] = np.array(self.anndata.obs["index_commu"])[bool_index_row]
        adata.obs["nb_rna"] = np.array(self.anndata.obs["nb_rna"])[bool_index_row]

        if n_pcs > 0:
            sc.tl.pca(adata, svd_solver='arpack', n_comps=np.min([n_comps, len(new_selected_genes)]))
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

        # sc.tl.umap(adata)
        if clustering_method == "leiden":
            sc.tl.leiden(adata, resolution=resolution)

        elif clustering_method == "louvain":
            sc.tl.louvain(adata, resolution=resolution)

        elif clustering_method == "kmeans":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=0).fit(adata.X)
            adata.obs['kmeans'] = list(np.array(kmeans.labels_).astype(str))
        else:
            raise Exception(f"clustering_method {clustering_method} not implemented")
        if plot_umap:
            try:
                sc.tl.umap(adata)
                sc.pl.umap(adata, color=[clustering_method], palette=palette, legend_loc='on data')
            except Exception as e:
                print(e)
                print("umap not computed")
        print(f"number of cluster {len(np.unique(adata.obs[clustering_method]))}")


        ## add labeled cluster to self anndata

        cluster_label = np.array(['-1'] * len(self.anndata))
        cluster_label[bool_index_row] = adata.obs[clustering_method]


        self.anndata.obs[clustering_method] = cluster_label
        self.anndata_cluster = adata

        return self.anndata

    """n_neighbors=args.in_situ_clustering_n_neighbors,
    n_comps=args.in_situ_n_comps,
    n_pcs=args.in_situ_n_pcs,
    resolution=args.in_situ_resolution,
    clustering_method=clustering_method,
    new_selected_genes=new_selected_genes,
    bool_index=bool_index,
    param_sctransform=param_sctransform,
    palette=palette,
    divide_by_norm=args.divide_by_norm,
    vector_key=args.agregation_mode,  # args.agregation_mode,
    unorm_vector_key="unorm_vector",
    n_clusters_kmeans=args.n_clusters_kmeans,
    max_number_image=len(dico_dico_commu),
    update_graph=True,
    divide_by_nb_spots=False, """

    ### ADD FCT TO MERGE CLUSTER  WITH A PREDEFINE NUMBER OF CLUSTER TO MERGE

    def cluster_centroid(self, ## withput normalization but
                      cluster_column_name = "leiden",
                       agregation_mode = "mean"):
        assert agregation_mode in ["mean", "median"], "arrgration_mode must be 'mean' or 'median'"
        scrna_unique_clusters = np.unique(list(self.anndata_cluster.obs[cluster_column_name]))
        scrna_centroids = []
        for cl in scrna_unique_clusters:
            print(cl)
            if agregation_mode == "median":
                centroid = np.median(self.anndata_cluster[self.anndata_cluster.obs[cluster_column_name] == cl].X, axis=0)
            else:
                assert agregation_mode == "mean"
                centroid = np.mean(self.anndata_cluster[self.anndata_cluster.obs[cluster_column_name] == cl].X, axis=0)
        # print(mean)
            scrna_centroids.append(np.asarray(centroid)[0])


        #eee
        self.scrna_centroids = np.array(scrna_centroids)
        self.scrna_unique_clusters = scrna_unique_clusters

        return self.scrna_centroids, self.scrna_unique_clusters

    def merge_cluster(self, nb_min_cluster = 0,
                      min_merge_correlation = 0.8,
                      cluster_column_name="leiden",
                      plot = True):

        scrna_unique_clusters, scrna_centroids = (list(t) for t in zip(*sorted(
            zip(np.array(self.scrna_unique_clusters).astype(int), self.scrna_centroids))))
        Z = scipy.cluster.hierarchy.linkage(scrna_centroids, metric="cosine")
        if plot:
            fig = plt.figure(figsize=(15, 10))
            dn = scipy.cluster.hierarchy.dendrogram(Z)
            plt.show()

        array_label = np.array(self.anndata_cluster.obs[cluster_column_name]).astype(int)
        nb_cluster = len(np.unique(scrna_unique_clusters))
        for i in range(len(Z[:-1])):
            dist_corr = Z[i, 2]
            if dist_corr > 1 - min_merge_correlation or nb_cluster <= nb_min_cluster:
                break
            print(Z[i].round(2))
            c1_to_merge = Z[i, 0]
            c2_to_merge = Z[i, 1]
            new_cluster = len(scrna_centroids) + i
            array_label[array_label == c1_to_merge] = new_cluster
            array_label[array_label == c2_to_merge] = new_cluster
            nb_cluster -= 1

        ## create a transitional dico
        dico_ori_merge = {}
        dico_merge_ori = {}
        list_ori_leiden = np.array(self.anndata_cluster.obs[cluster_column_name])
        for ind_l in range(len(array_label)):
            dico_ori_merge[list_ori_leiden[ind_l]] = array_label[ind_l]
            dico_merge_ori[array_label[ind_l]] = list_ori_leiden[ind_l]
        dico_ori_merge['-1'] = '-1'
        dico_merge_ori['-1'] = '-1'

        self.dico_merge_ori = dico_merge_ori #todo erase this name
        self.dico_ori_merge = dico_ori_merge #todo erase this name
        column_name = cluster_column_name + '_merged'
        new_list_ori_leiden = []
        for cluster_id in list_ori_leiden:
            new_list_ori_leiden.append(dico_ori_merge[cluster_id])
        self.anndata_cluster.obs[column_name] = new_list_ori_leiden

        ### merge also anndata normal
        list_ori_leiden = np.array(self.anndata.obs[cluster_column_name])
        new_list_ori_leiden = []
        for cluster_id in list_ori_leiden:
            new_list_ori_leiden.append(dico_ori_merge[cluster_id])
        self.anndata.obs[column_name] = new_list_ori_leiden

        return self.anndata_cluster.obs[column_name]

    def classify_by_nn(self,
                        array_of_vect,
                       pca_model,
                       kn_neighb,
                       min_proba=0.5,
                       param_sctransform=None):

        if param_sctransform is None:
            import warnings
            warnings.warn('param_sctransform is none, expression vector are not normalized')

        array_of_vect = np.array(array_of_vect)

        if array_of_vect.ndim == 1:
            array_of_vect = array_of_vect.reshape(1, len(array_of_vect))
        elif array_of_vect.ndim == 3:
            array_of_vect = array_of_vect[:, 0, :]

        if param_sctransform is not None:

            norm_expression_vectors = sctransform_from_parameters(
                np.array(param_sctransform),
                array_of_vect)
        else:
            norm_expression_vectors = array_of_vect

        if pca_model is not None:
            list_boll_nan = np.sum(np.isnan(norm_expression_vectors), axis=1) == 0
            projected_vect = pca_model.transform(norm_expression_vectors[list_boll_nan])
        else:
            list_boll_nan = [True for i in range(len(norm_expression_vectors))]
            projected_vect = norm_expression_vectors
        proba = kn_neighb.predict_proba(projected_vect)
        list_index_cluster_max = np.argmax(proba, axis=1)
        ## apply a nn classifier
        list_pred_rna_seq = []
        decal_varr = 0
        assert np.sum(list_boll_nan) == len(proba)
        assert len(projected_vect) == len(proba)
        for boll_nan_index in range(len(list_boll_nan)):
            if not list_boll_nan[boll_nan_index]:
                decal_varr += 1
                list_pred_rna_seq.append('-1')
                continue
            index_cluster_max = list_index_cluster_max[boll_nan_index - decal_varr]
            if proba[boll_nan_index - decal_varr][index_cluster_max] < min_proba:
                pred_rna_seq = '-1'
            else:
                pred_rna_seq = kn_neighb.classes_[index_cluster_max]
            assert type(pred_rna_seq) == str or type(pred_rna_seq) == np.str_
            list_pred_rna_seq.append(str(pred_rna_seq))
        return norm_expression_vectors, proba, list_pred_rna_seq

    def classify_small_community(self,
        size_commu_max = np.inf,
        size_commu_min_classif = 0,
        key_pred = "leiden_merged",
        unorm_vector_key = "unorm_vector",
        classify_mode = "pca",
        min_correlation = 0,
        min_proba_small_commu = 0,
          ):

        """"
        todo remove small community
        :param size_commu_max:
        :param size_commu_min_classif:
        :param key_pred:
        :param unorm_vector_key:
        :param classify_mode:
        :param min_correlation:
        :param min_proba_small_commu:
        :return:
        """

        if classify_mode == 'pca' or classify_mode == 'euclidien':
            kn_neighb = sklearn.neighbors.KNeighborsClassifier(
                n_neighbors=10,
                weights='distance',
                algorithm='auto',
                leaf_size=30,
                p=2,
                metric='minkowski',  # 'cosine'
                metric_params=None,
                n_jobs=None,
            )
            if classify_mode == 'pca':
                kn_neighb.fit(self.anndata_cluster.obsm['X_pca'],
                              np.array(self.anndata_cluster.obs[key_pred]).astype(str)
                              )

                components_ = self.anndata_cluster.varm['PCs'].T
                pca_model = sklearn.decomposition.PCA(n_components=components_.shape[0])
                pca_model.components_ = components_
                pca_model.mean_ = self.anndata_cluster.X.mean(axis=0)
            else:
                pca_model = None
                kn_neighb.fit(self.anndata_cluster.X,
                              np.array(self.anndata_cluster.obs[key_pred]).astype(str)
                              )

                ## get_index of unclassify community index


            bool_index_unclassified = np.array(self.anndata.obs[key_pred]) == '-1'
            index_unclassified = np.nonzero(bool_index_unclassified)[0]

            ## get unclassified community  expression vector
            unclassified_vector = self.anndata[index_unclassified,
                                  self.genes_to_take].X.toarray()

            ## classify unclassified community
            norm_expression_vectors, proba, list_pred_rna_seq = self.classify_by_nn(
                unclassified_vector,
                pca_model,
                kn_neighb,
                min_proba=min_proba_small_commu,
                param_sctransform=self.param_sctransform)

            list_pred_rna = np.array(self.anndata.obs['leiden_merged'])
            list_pred_rna[index_unclassified] = list_pred_rna_seq
            self.anndata.obs['leiden_merged'] = list_pred_rna

            return self.anndata.obs['leiden_merged']













