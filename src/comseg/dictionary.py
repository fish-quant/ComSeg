




"""
class set store the graph and anndata of the comseg.rst(s) object
preprocess it (like concatenate anndata) to perform classification
then apply a community classification (in situ clustering class)
"""

#%%


import os
import sys

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
from tqdm import tqdm

import anndata as ad
import clustering
import model
import clustering
#from unused_files.similarity_m import get_simialrity_matrix
#from utils import data_processing as data_processing



from pathlib import Path
import numpy as np

__all__ = ["ComSegDict"]




class ComSegDict():
    """
    As a dataset is often compose of many separated images. It is requiered to create many ComSeg graph of RNAs.
    besides, the in-situ clustering to identify the transcriptomic profile is more informative at the data scale.
    To ease the analysis entire dataset we implement ComSegDict. It is a class that store many ComSeg object
    and allows to perform analysis at the dataset scale.

    This class is implemented as a dictionary of ComSeg graph object
    """

    def __init__(self, dataset=None,
                 mean_cell_diameter= None,
                 clustering_method="louvain_with_prior",
                 prior_keys="in_nucleus",
                 seed=None,
                 super_node_prior_keys="in_nucleus",
                 confidence_level=1,
                 ):

        """
        :param dataset:
        :type dataset: ComSegDataset
        :param mean_cell_diameter: the expected mean cell diameter in µm default is 15µm
        :type mean_cell_diameter: float
        :param clustering_method: choose in ["with_prior",  "louvain"], "with_prior" is our graph partioning / community
                detection method taking into account prior knowledge
        :type clustering_method: str
        :param prior_keys: key of the prior cell assignment in the node attribute dictionary and in the input CSV file
        :type prior_keys: str
        :param seed: (optional) seed for the grpah partioning initialization
        :type seed: int
        :param super_node_prior_keys: key of the prior cell assignment in the node attribute
             and in the input CSV file that is certain. node labeled with the same supernode prior key will be merged.
             prior_keys and super_node_prior_keys can be the different if two landmarks mask prior are available.
             exemple: super_node_prior_keys = "nucleus_landmak", prior_keys = "uncertain_cell_landmark"
        :type super_node_prior_keys: str
        :param confidence_level: confidence level for the prior knowledge (prior_keys) in the range [0,1]. 1 means that the prior knowledge is certain.
        :type confidence_level: float
        """

        self.dataset = dataset
        self.mean_cell_diameter = mean_cell_diameter
        self.clustering_method = clustering_method
        self.prior_keys = prior_keys
        self.seed = seed
        self.super_node_prior_keys = super_node_prior_keys
        self.confidence_level = confidence_level
        self.dict_img_name = {}

        ###
        ##
        ##
        return
    ## create directed grap
    def __setitem__(self, key, item):
        self.dict_img_name[key] = item

    def __getitem__(self, key):
        return self.dict_img_name[key]

    def __repr__(self):
        return repr(self.dict_img_name )

    def __len__(self):
        return len(self.dict_img_name )

    def __delitem__(self, key):
        del self.dict_img_name [key]

    def clear(self):
        return self.dict_img_name.clear()

    def copy(self):
        return self.dict_img_name.copy()

    def has_key(self, k):
        return k in self.dict_img_name

    def update(self, *args, **kwargs):
        return self.dict_img_name.update(*args, **kwargs)

    def keys(self):
        return self.dict_img_name.keys()

    def values(self):
        return self.dict_img_name.values()

    def items(self):
        return self.dict_img_name.items()

    def pop(self, *args):
        return self.dict_img_name.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.dict_img_name , dict_)

    def __contains__(self, item):
        return item in self.dict_img_name

    def __iter__(self):
        return iter(self.dict_img_name)


    def concatenate_anndata(self):
        """
        concatenate all community expression vectors from all
        the ComSeg graphs into a single anndata object

        :return: anndata
        :rtype:  AnnData
        """
        self.global_anndata = ad.concat([self[img].community_anndata for img in self
                                      if str(type(self[img])) == "<class 'comseg.rst.model.ComSeg'>"])
        self.global_anndata.obs["img_name"] =  np.concatenate([[img] * len(self[img].community_anndata) for img in self
                                             if  str(type(self[img])) == "<class 'comseg.rst.model.ComSeg'>"])
        return self.global_anndata


    ### compute in-situ vector


    def compute_community_vector(self,):
        """

        for all the images in the dataset, this function creates a graph of RNAs
        and compute the community vectors

        :param self:
        :return:
        """
        for img_name in tqdm(list(self.dataset)):
            #### GRAPH CREATION
            comseg_m = model.ComSegGraph(selected_genes=self.dataset.selected_genes,
                                         df_spots_label=self.dataset[img_name],
                                         dict_scale=self.dataset.dict_scale,
                                         mean_cell_diameter=self.mean_cell_diameter,  # in micrometer
                                         )
            comseg_m.create_graph(dict_co_expression=self.dataset.dict_co_expression,
                                  )

            #### COMMÛTE COMMUNITY OF RNA

            comseg_m.community_vector(
                clustering_method=self.clustering_method,
                prior_keys=self.prior_keys,
                seed=self.seed,
                super_node_prior_keys=self.super_node_prior_keys,
                confidence_level=self.confidence_level,
            )
            self[img_name] = comseg_m



    ### INSITU CLUSTERING WITH THE CLASS


    def compute_insitu_clustering(self,
                                  size_commu_min=3,
                                  norm_vector=True,
                                  ### parameter clustering
                                  n_pcs=3,
                                  n_comps=3,
                                  clustering_method="leiden",
                                  n_neighbors=20,
                                  resolution=1,
                                  n_clusters_kmeans=4,
                                  palette=None,
                                  nb_min_cluster=0,
                                  min_merge_correlation=0.8,
                                  merge_cluster = True,
                                  ):


        """
        Cluster all together  the RNA partition/community expression vector for all the images in the dataset and
        identify the single cell transcriptomic cluster present in the dataset


        #todo clean the name leiden vs leiden_merged aka clustering_method

        #todo or add the current cleuter name to use in the self so it is reuse in  add_cluster_id_to_graph

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
        :type n_clusters_kmeans: int
        :param palette: color palette for the cluster list of (HEX) color
        :type palette: list[str]
        :param min_merge_correlation: minimum correlation to merge cluster
        :type min_merge_correlation: float
        :return:
        """


        self.clustering_method = clustering_method
        try:
            self.global_anndata
        except:
            self.concatenate_anndata()
        self.in_situ_clustering = clustering.InSituClustering(anndata=self.global_anndata,
                                        selected_genes=self.global_anndata.var_names)
        ### APPLY NORMALIZATION
        self.in_situ_clustering.compute_normalization_parameters()
        self.global_anndata = self.in_situ_clustering.cluster_rna_community(
                                                          size_commu_min=size_commu_min,
                                                          norm_vector=norm_vector,
                                                          ### parameter clustering
                                                          n_pcs=n_pcs,
                                                          n_comps=n_comps,
                                                          clustering_method=clustering_method,
                                                          n_neighbors=n_neighbors,
                                                          resolution=resolution,
                                                          n_clusters_kmeans=n_clusters_kmeans,
                                                          palette=palette
                                                        )
        self.in_situ_clustering.get_cluster_centroid(
            cluster_column_name=clustering_method,
            agregation_mode="mean")

        if merge_cluster:
            self.in_situ_clustering.merge_cluster(nb_min_cluster=nb_min_cluster,
                                            min_merge_correlation=min_merge_correlation,
                                            cluster_column_name=clustering_method,
                                            plot=False)
            self.clustering_method = self.clustering_method + "_merged"

        self.global_anndata = self.in_situ_clustering.anndata
        ### classify small community with cluster_centroid
        self.in_situ_clustering.classify_small_community(
        size_commu_max = np.inf,
        size_commu_min_classif = 0,
        key_pred = self.clustering_method,
        unorm_vector_key = "unorm_vector",
        classify_mode = "pca",
        min_correlation = 0,
        min_proba_small_commu = 0,)



        ## add cluster to community of each images
        for img in self:
            if str(type(self[img])) != "<class 'comseg.rst.model.ComSeg'>":
                continue
            ## get the cluster list and the community index
            cluster_id_list = list(self.global_anndata[self.global_anndata.obs.img_name == img].obs[self.clustering_method])
            community_index_list = list(self.global_anndata[self.global_anndata.obs.img_name == img].obs.index_commu)
            assert community_index_list == list(self[img].community_anndata.obs['index_commu'])
            self[img].community_anndata.obs[self.clustering_method] = cluster_id_list
            ## self[img].community_anndata.obs["cluster"] = self.global_anndata.obs["cluster"].loc[self[img].community_anndata.obs.index]
            ## loop on comseg.rst model images
        return self.global_anndata




    ### ADD FOUND CLUSTER FROM Global anndata to node graph



    def add_cluster_id_to_graph(self,
                                clustering_method = "leiden_merged"):

        """

        Add transcriptional cluster id to each RNA molecule in the graph

        :param self:
        :param clustering_method: clustering method used to get the community (kmeans, leiden_merged, louvain)
        :type clustering_method: str
        :return:
        """
        #todo use the method of the comseg instance
        list_img = [img for img in self if str(type(self[img])) == "<class 'comseg.rst.model.ComSeg'>"]
        for img_name in list_img:
            list_index_commu = list(self[img_name].community_anndata.obs['index_commu'])
            list_cluster_id = list(self[img_name].community_anndata.obs[clustering_method])
            ## get a dico {commu : cluster ID}
            dico_commu_cluster = {}
            for commu_index in range(len(list_index_commu)):
                dico_commu_cluster[list_index_commu[commu_index]] = list_cluster_id[commu_index]
            G = self[img_name].G
            for node in tqdm(G.nodes()):
                if G.nodes[node]['gene'] == 'centroid':
                    continue
                G.nodes[node][clustering_method] = str(dico_commu_cluster[G.nodes[node]['index_commu']])
            self[img_name].G = G
        return self


    ### Add and classify centroid

    def classify_centroid(self,
                      path_dict_cell_centroid,
                      n_neighbors=15,
                      dict_in_pixel=True,
                      max_dist_centroid=None,
                      key_pred="leiden_merged",
                      distance="gaussian",
                      convex_hull_centroid=True,
                        file_extension = "tiff.npy"
                      ):

        """
        Classify cell centroids based on their  centroid neighbors RNA
        label from ``add_cluster_id_to_graph()``


        :param path_dict_cell_centroid: path to the folder containing the centroid dictionary {cell : {z:,y:,x:}}
                        with each centroid dictionary in a file named as the image name and store in a npy format
        :type path_dict_cell_centroid: str
        :param n_neighbors: number of neighbors to consider for the classification of the centroid (default 15)
        :type n_neighbors: int
        :param dict_in_pixel: if True the centroid are in pixel and rescal if False the centroid are in um (default True)
        :type dict_in_pixel: bool
        :param max_dist_centroid: maximum distance to consider for the centroid (default None)
        :type max_dist_centroid: int
        :param key_pred: key of the node attribute containing the cluster id (default "leiden_merged")
        :type key_pred: str
        :param convex_hull_centroid: check that cell centroid is in the convex hull of its RNA neighbors (default True). If not the cell centroid is not classify to avoid artefact misclassification
        :type convex_hull_centroid: bool
        :param file_extension: file extension of the centroid dictionary
        :return:
        """


        for img_name in tqdm(self):
            if str(type(self[img_name])) != "<class 'comseg.rst.model.ComSeg'>":
                continue
            dict_cell_centroid = np.load(Path(path_dict_cell_centroid) / (img_name + file_extension), allow_pickle=True).item()

            self[img_name].classify_centroid(dict_cell_centroid,
                                             n_neighbors=n_neighbors,
                                             dict_in_pixel=dict_in_pixel,
                                             max_dist_centroid=max_dist_centroid,
                                             key_pred=key_pred,
                                             distance=distance,
                                             convex_hull_centroid=convex_hull_centroid,
                                             )


    #### Apply diskjtra


    def associate_rna2landmark(self,
        key_pred="leiden_merged",
        super_node_prior_key='in_nucleus',
        distance='distance',
        max_distance=100):
        """
        Associate RNAs to landmarks based on the both transcriptomic landscape and the
        distance between the RNAs and the centroids of the landmark


        :param key_pred: key of the node attribute containing the cluster id (default "leiden_merged")
        :type key_pred: str
        :param super_node_prior_key:
        :type super_node_prior_key: str
        :param max_distance: maximum distance between a cell centroid and an RNA to be associated (default 100)
        :type max_distance: float
        :return:

        """

        for img_name in tqdm(self):
            if str(type(self[img_name])) != "<class 'comseg.rst.model.ComSeg'>":
                continue

            self[img_name].associate_rna2landmark(
                key_pred=key_pred,
                super_node_prior_key=super_node_prior_key,
                distance=distance,
                max_distance=max_distance)


    #### return an anndata with expresion vector
    ## as obs image name | centroid coordinate |
    # list of rna spots coordinate |
    # list of the corresponding rna species
    ##



    def anndata_from_comseg_result(self,
                                   key_cell_pred='cell_index_pred',
                                   ):

        """
        Return an anndata with the estimated expression vector of each cell in the dataset plus the spot positions.

        :param self:
        :param key_cell_pred: leave it to default
        :return:
        """
        list_image_name = []
        anndata_list = []
        dict_df_spots = {}

        for img_name in tqdm(self):
            if str(type(self[img_name])) != "<class 'comseg.rst.model.ComSeg'>":
                continue
            anndata = self[img_name].get_anndata_from_result(
                key_cell_pred=key_cell_pred)

            anndata_list.append(anndata)
            list_image_name += [img_name] * len(anndata)
            dict_df_spots[img_name] = anndata.uns["df_spots"]

            assert np.array_equal(anndata.var_names, self.dataset.selected_genes), "The anndata var names are not the same as the dataset selected genes"



        self.final_anndata = ad.concat(anndata_list)
        self.final_anndata.obs["image_name"] = list_image_name
        self.final_anndata.var["features"] = self.dataset.selected_genes
        self.final_anndata.var_names = self.dataset.selected_genes
        self.final_anndata.uns["df_spots"] = dict_df_spots
        return self.final_anndata

