import sklearn

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scanpy as sc

import scipy
from matplotlib import pyplot as plt
#from .utils.preprocessing import run_sctransform, select_genes_for_sct
#from .utils.preprocessing import sctransform_from_parameters
from .utils.preprocessing import run_sctransform, select_genes_for_sct
from .utils.preprocessing import sctransform_from_parameters

__all__ = ["InSituClustering"]


class InSituClustering():
    """
    In situ clustering class takes as attribute an anndata object containing the community expression vectors :math:`V_c`
    of RNA partitions/communities from one or many images. This class is in charge of identifying the single cell transcriptomic
    clusters present in the dataset.
    """

    def __init__(self,
                 anndata,
                 selected_genes, ):
        """
        :param anndata: anndata object containing the expression vector of the community.
                The anndata can be the concatenation of several anndata object from different ComSeg instance
        :type anndata: anndata object

        :param selected_genes: list of genes to take into account for the clustering
                the gene list order will define the order of the gene in the expression vector
        :type selected_genes: list[str]
        """

        self.anndata = anndata  ## contain the expression vector of all commu
        self.selected_genes = selected_genes
        self.anndata_cluster = None  ## contain the expression vector of the cluster commu ie more than x RNA
        #self.anndata.var.index = self.anndata.var.index.astype(str)

    def compute_normalization_parameters(self,
                                         debug_path=None):
        """

        Compute the ScTransform normalization parameters from the class attribute anndata

        :param debug_path:
        :return:
        """
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

        #print(f'shape count matrix {count_matrix.shape}')
        np.save("count_matrix", count_matrix)
        count_matrix = pd.DataFrame(count_matrix, columns=[str(e) for e in range(count_matrix.shape[1])])
        #print(count_matrix.columns)
        norm_expression_vectors, param_sctransform = run_sctransform(count_matrix,
                                                                     debug_path=debug_path)

        self.param_sctransform = param_sctransform
        self.genes_to_take = bool_index

        return norm_expression_vectors, param_sctransform

    def cluster_rna_community(self,
                              size_commu_min=3,
                              norm_vector=True,

                              ### parameter clustering
                              n_pcs=15,
                              n_comps=15,
                              clustering_method="leiden",
                              n_neighbors=20,
                              resolution=1,
                              n_clusters_kmeans=4,
                              palette=None,
                              plot_umap=False):

        """
        Cluster the RNA partition/community expression vector to identify the single cell transcriptomic cluster present in the dataset

        :param size_commu_min: minimum number of RNA in a community to be considered for the clustering
        :type size_commu_min: int
        :param norm_vector: if True, the expression vector will be normalized using the scTRANSFORM normalization parameters
        :type norm_vector: bool
        :param n_pcs: number of principal component to compute for the clustering; Lets 0 if no pca
        :type n_pcs: int
        :param n_comps: number of components to compute for the clustering; Lets 0 if no pca
        :type n_comps: int
        :param clustering_method: choose in ["leiden", "kmeans", "louvain"]
        :type clustering_method: str
        :param n_neighbors: number of neighbors similarity graph
        :type n_neighbors: int
        :param resolution: resolution parameter for the leiden/Louvain clustering
        :type resolution: float
        :param n_clusters_kmeans: number of cluster for the kmeans clustering
        :rtype n_clusters_kmeans: int
        :param palette: color palette for the cluster list of (HEX) color
        :type palette: list[str]
        :param plot_umap: if True, plot the umap of the cluster
        :return:
        """

        #### select vector with nb of rna superior to size_commu_min

        bool_index_row = np.array(self.anndata.obs['nb_rna']) > size_commu_min

        if norm_vector:  ## apply sctrasnform
            if not hasattr(self, "param_sctransform"):
                raise ValueError \
                    ("You need to compute the normalization parameters with 'compute_normalization_parameters' before clustering ")
            #self.anndata.obs.index = self.anndata.obs.index.astype(str)
            #self.anndata.var.index = self.anndata.var.index.astype(str)
            anndata = self.anndata[bool_index_row, self.genes_to_take]
            count_matrix_anndata = sctransform_from_parameters(self.param_sctransform,
                                                               anndata.X.toarray())
            new_selected_genes = self.norm_genes
        else:
            count_matrix_anndata = self.anndata[bool_index_row, :].X.toarray()
            #bool_index_row = [True] * len(self.anndata)
            new_selected_genes = self.selected_genes

        ## norm vector if needed

        count_matrix_anndata = np.nan_to_num(count_matrix_anndata)
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

        # add labeled cluster to self anndata

        cluster_label = np.array(['-1'] * len(self.anndata))
        cluster_label[bool_index_row] = adata.obs[clustering_method]

        self.anndata.obs[clustering_method] = cluster_label
        self.anndata_cluster = adata

        return self.anndata

    def get_cluster_centroid(self,  # withput normalization but
                             cluster_column_name="leiden",
                             aggregation_mode="mean"):
        """
        Compute the centroid of each transcriptomic cluster

        :param cluster_column_name:  name of the column containing the cluster label i.e. the method name
        :type cluster_column_name: str
        :param aggregation_mode:  choose in ["mean", "median"]
        :type aggregation_mode: str
        :return: scrna_centroids: list of centroids of each cluster. and list of cluster names
        :rtype: list[np.array], list[str]

        """
        assert aggregation_mode in ["mean", "median"], "arrgration_mode must be 'mean' or 'median'"
        scrna_unique_clusters = np.unique(list(self.anndata_cluster.obs[cluster_column_name]))
        scrna_centroids = []
        for cl in scrna_unique_clusters:
            ## print(cl)
            if aggregation_mode == "median":
                centroid = np.median(self.anndata_cluster[self.anndata_cluster.obs[cluster_column_name] == cl].X,
                                     axis=0)
            else:
                assert aggregation_mode == "mean"
                centroid = np.mean(self.anndata_cluster[self.anndata_cluster.obs[cluster_column_name] == cl].X, axis=0)
            # print(mean)
            scrna_centroids.append(np.asarray(centroid)[0])

        #eee
        self.scrna_centroids = np.array(scrna_centroids)
        self.scrna_unique_clusters = scrna_unique_clusters

        return self.scrna_centroids, self.scrna_unique_clusters

    def merge_cluster(self, nb_min_cluster=0,
                      min_merge_correlation=0.8,
                      cluster_column_name="leiden",
                      plot=True):
        """
        Merge clusters based on the correlation of their centroid

        :param nb_min_cluster:  minimum number of clusters to merge
        :type nb_min_cluster: int
        :param min_merge_correlation: minimum correlation to merge clusters
        :type min_merge_correlation: float
        :param cluster_column_name:  clustering method used
        :type cluster_column_name: str
        :param plot:
        :return:
        """
        from scipy import cluster
        scrna_unique_clusters, scrna_centroids = (list(t) for t in zip(*sorted(
            zip(np.array(self.scrna_unique_clusters).astype(int), self.scrna_centroids))))
        Z = cluster.hierarchy.linkage(scrna_centroids, metric="cosine")
        if plot:
            fig = plt.figure(figsize=(15, 10))
            dn = cluster.hierarchy.dendrogram(Z)
            plt.show()

        array_label = np.array(self.anndata_cluster.obs[cluster_column_name]).astype(int)
        nb_cluster = len(np.unique(scrna_unique_clusters))
        for i in range(len(Z[:-1])):
            dist_corr = Z[i, 2]
            if dist_corr > 1 - min_merge_correlation or nb_cluster <= nb_min_cluster:
                break
            #print(Z[i].round(2))
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

        self.dico_merge_ori = dico_merge_ori  #todo erase this name
        self.dico_ori_merge = dico_ori_merge  #todo erase this name
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
        print(f"number of cluster after merging {len(dico_merge_ori)}")

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
            list_boll_nan = np.sum(np.isnan(norm_expression_vectors), axis=1) == 0
            projected_vect = norm_expression_vectors[list_boll_nan]
        projected_vect = np.array(projected_vect)
        if len(projected_vect) > 0 and np.sum(list_boll_nan) != 0:
            proba = kn_neighb.predict_proba(projected_vect)
            list_index_cluster_max = np.argmax(proba, axis=1)
        else:
            assert np.sum(list_boll_nan) == 0

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
                                 #size_commu_min_classif = 0,
                                 key_pred="leiden_merged",
                                 # unorm_vector_key = "unorm_vector",
                                 classify_mode="pca",
                                 # min_correlation = 0,
                                 min_proba_small_commu=0,
                                 ):

        #todo remove small community
        """
        associate unclassified RNA community expression vector by using a knn classifier
        and the already classify communities

        :param key_pred: leave default
        :param unorm_vector_key: leave default
        :param classify_mode:  choose in 'pca' or 'euclidien'. it either uses the euclidian space or PCA space
        :param min_proba_small_commu: minimum probability to classify a small community based on the KNN classifier
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

            bool_index_unclassified = np.logical_and(np.array(self.anndata.obs[key_pred]) == '-1',
                                                     np.sum(self.anndata.X.toarray(), axis=1) != 0)
            index_unclassified = np.nonzero(bool_index_unclassified)[0]
            ## get unclassified community  expression vector
            unclassified_vector = self.anndata[index_unclassified,
            self.genes_to_take].X.toarray()

            if len(unclassified_vector) > 0 and np.sum(unclassified_vector) > 0:
                ## classify unclassified community
                norm_expression_vectors, proba, list_pred_rna_seq = self.classify_by_nn(
                    array_of_vect=unclassified_vector,
                    pca_model=pca_model,
                    kn_neighb=kn_neighb,
                    min_proba=min_proba_small_commu,
                    param_sctransform=self.param_sctransform
                )

                list_pred_rna = np.array(self.anndata.obs[key_pred])
                list_pred_rna[index_unclassified] = list_pred_rna_seq
                self.anndata.obs[key_pred] = list_pred_rna
            else:
                print("no small vector to classify")

            return self.anndata.obs[key_pred]
