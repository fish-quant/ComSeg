# todo add dataset to zenodo
# todo add function to compute centroid of cell in a df format + harmonize with the rest of the code
# update tutorial


#%%


import pandas as pd

from tqdm import tqdm
import anndata as ad
import pickle
from .clustering import InSituClustering
from .model import ComSegGraph
from pathlib import Path
import numpy as np

__all__ = ["ComSegDict"]


class ComSegDict():
    """
    As a dataset is often composed of many separated images. It is required to create many ComSeg graphs of RNAs.
    To ease the analysis of entire dataset, we implement ComSegDict. It is a class that store many ComSeg object
    and allows to perform analysis at the dataset scale.
    This class is implemented as a dictionary of ComSeg graph object
    """

    def __init__(self,
                 dataset=None,
                 mean_cell_diameter=None,
                 community_detection="with_prior",
                 seed=None,
                 prior_name="in_nucleus",
                 ):

        """
        :param dataset:
        :type dataset: ComSegDataset
        :param mean_cell_diameter: the expected mean cell diameter in µm default is 15µm
        :type mean_cell_diameter: float
        :param community_detection: choose in ["with_prior",  "louvain"], "with_prior" is our graph partioning / community
                detection method taking into account prior knowledge
        :type community_detection: str
        :param seed: (optional) seed for the graph partioning initialization
        :type seed: int
        :param prior_name: (optional) Name of the prior cell assignment column the input CSV file. Node with the same prior label will be merged into a super node.
        node with different prior label can not be merged during the modularity optimization.
        :type prior_name: str
        """

        #:param confidence_level: (experimental) confidence level for the prior knowledge (prior_name) in the range [0,1]. 1 means that the prior knowledge is certain.
        #:type confidence_level: float
        self.dataset = dataset
        self.mean_cell_diameter = mean_cell_diameter
        self.community_detection = community_detection
        self.seed = seed
        self.prior_name = prior_name
        self.dict_img_name = {}

        ###
        ##
        ##
        #return

    ## create directed grap
    def __setitem__(self, key, item):
        self.dict_img_name[key] = item

    def __getitem__(self, key):
        return self.dict_img_name[key]

    def __repr__(self):
        return repr(f'ComSegDict {self.dict_img_name}')

    def __len__(self):
        return len(self.dict_img_name)

    def __delitem__(self, key):
        del self.dict_img_name[key]

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
        return self.__cmp__(self.dict_img_name, dict_)

    def __contains__(self, item):
        return item in self.dict_img_name

    def __iter__(self):
        return iter(self.dict_img_name)

    def save(self, file_name):
        """save class as self.name.txt"""
        #assert file_name[-4:] == '.txt', 'file_name must end with .txt'
        with open(file_name, "wb") as file_:
            pickle.dump(self.__dict__, file_, -1)

    def load(self, file_name):
        """try load self.name.txt"""
        self.__dict__ = pickle.load(
            open(file_name, "rb", -1))

    def concatenate_anndata(self):
        """
        concatenate all community expression vectors from all
        the ComSeg graphs into a single anndata object

        :return: anndata
        :rtype:  AnnData
        """
        if len(self) > 1:
            self.global_anndata = ad.concat([self[img].community_anndata for img in self])
            self.global_anndata.obs["img_name"] = np.concatenate \
                ([[img] * len(self[img].community_anndata) for img in self])
        else:
            img_name = list(self.dict_img_name.keys())[0]
            self.global_anndata = self[img_name].community_anndata
            if type(img_name) == type(''):
                self.global_anndata.obs["img_name"] = [img_name] * len(self.global_anndata)
            elif type(img_name) == type([]):
                self.global_anndata.obs["img_name"] = img_name * len(self.global_anndata)
        return self.global_anndata

    ### compute in-situ vector

    def compute_community_vector(self,
                                 k_nearest_neighbors: int = 10):
        """

        for all the images in the dataset, this function creates a graph of RNAs
        and compute the community vectors

        :param self:
        :param k_nearest_neighbors: number of nearest neighbors to consider for the graph creation
        :type k_nearest_neighbors: int
        :return:
        """
        for img_name in tqdm(list(self.dataset)):
            #### GRAPH CREATION
            comseg_m = ComSegGraph(
                selected_genes=self.dataset.selected_genes,
                df_spots_label=self.dataset[img_name],
                dict_scale=self.dataset.dict_scale,
                mean_cell_diameter=self.mean_cell_diameter,  # in micrometer
                dict_co_expression=self.dataset.dict_co_expression,
                k_nearest_neighbors=k_nearest_neighbors,
                gene_column=self.dataset.gene_column,
                prior_name=self.prior_name,
            )
            comseg_m.create_graph()
            self[img_name] = comseg_m

            #### COMMÛTE COMMUNITY OF RNA
            comseg_m.community_vector(
                clustering_method=self.community_detection,
                seed=self.seed,
                prior_name=self.prior_name,
            )
            self[img_name] = comseg_m

    ### INSITU CLUSTERING WITH THE CLASS

    def compute_insitu_clustering(self,
                                  size_commu_min=3,
                                  norm_vector=True,
                                  # parameter clustering
                                  n_pcs=3,
                                  n_comps=3,
                                  clustering_method="leiden",
                                  n_neighbors=20,
                                  resolution=1,
                                  n_clusters_kmeans=4,
                                  palette=None,
                                  nb_min_cluster=0,
                                  min_merge_correlation=0.8,
                                  merge_cluster=True,
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
        :param merge_cluster: if True, the clusters with a correlation > min_merge_correlation will be merged and the clustering method is renamed {clustering_method}_merged
        :param min_merge_correlation: minimum correlation to merge cluster
        :type min_merge_correlation: float
        :return:
        """

        self.clustering_method = clustering_method
        try:
            self.global_anndata
        except:
            self.concatenate_anndata()
        self.in_situ_clustering = InSituClustering(anndata=self.global_anndata,
                                                   selected_genes=self.global_anndata.var_names)
        ### APPLY NORMALIZATION
        if norm_vector:
            self.in_situ_clustering.compute_normalization_parameters()
        else:
            self.in_situ_clustering.param_sctransform = None
            self.in_situ_clustering.genes_to_take = [True] * len(self.in_situ_clustering.selected_genes)

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
            aggregation_mode="mean")

        if merge_cluster:
            self.in_situ_clustering.merge_cluster(nb_min_cluster=nb_min_cluster,
                                                  min_merge_correlation=min_merge_correlation,
                                                  cluster_column_name=clustering_method,
                                                  plot=False)
            self.clustering_method = self.clustering_method + "_merged"

        self.global_anndata = self.in_situ_clustering.anndata
        ### classify small community with cluster_centroid
        if n_pcs > 0:
            classify_mode = "pca"
        else:
            classify_mode = "euclidien"
        self.in_situ_clustering.classify_small_community(
            key_pred=self.clustering_method,
            classify_mode=classify_mode,
            min_proba_small_commu=0, )

        ## add cluster to community of each images
        for img in self:
            ## get the cluster list and the community index
            cluster_id_list = list \
                (self.global_anndata[self.global_anndata.obs.img_name == img].obs[self.clustering_method])
            community_index_list = list(self.global_anndata[self.global_anndata.obs.img_name == img].obs.index_commu)
            assert community_index_list == list(self[img].community_anndata.obs['index_commu'])
            self[img].community_anndata.obs[self.clustering_method] = cluster_id_list
            ## self[img].community_anndata.obs["cluster"] = self.global_anndata.obs["cluster"].loc[self[img].community_anndata.obs.index]
            ## loop on comseg.rst model images
        return self.global_anndata

    ### ADD FOUND CLUSTER FROM Global anndata to node graph

    def add_cluster_id_to_graph(self,
                                clustering_method="leiden_merged"):

        """

        Add transcriptional cluster id to each RNA molecule in the graph

        :param self:
        :param clustering_method: clustering method used to get the community (kmeans, leiden_merged, louvain)
        :type clustering_method: str
        :return:
        """
        #todo use the method of the comseg instance
        list_img = [img for img in self]
        for img_name in list_img:
            list_index_commu = list(self[img_name].community_anndata.obs['index_commu'])
            list_cluster_id = list(self[img_name].community_anndata.obs[clustering_method])

            ## get a dico {commu : cluster ID}
            dico_commu_cluster = {}
            for commu_index in range(len(list_index_commu)):
                dico_commu_cluster[list_index_commu[commu_index]] = list_cluster_id[commu_index]

            self[img_name].add_cluster_id_to_graph(
                dict_cluster_id=dico_commu_cluster,
                clustering_method=clustering_method)

        return self

    ### Add and classify centroid

    def classify_centroid(self,
                          path_cell_centroid=None,
                          n_neighbors=15,
                          dict_in_pixel=True,
                          max_dist_centroid=None,
                          key_pred="leiden_merged",
                          distance="ngb_distance_weights",
                          file_extension="tiff.npy",
                          centroid_csv_key={"x": "x", "y": "y", "z": "z", "cell_index": "cell"}
                          ):

        """
        Classify cell centroids based on their  centroid neighbors RNA
        label from ``add_cluster_id_to_graph()``


        :param path_dict_cell_centroid: If computed already by the ``ComSegDataset`` class from prior Maks leave it None.
        Otherwise : path_dict_cell_centroid is a  Path to the folder containing the centroid dictionary {cell : {z:,y:,x:}} for each image.
                         Each centroid dictionary has to be stored in a file in a npy format,  named as the image name.
        :type path_dict_cell_centroid: str
        :param n_neighbors: number of neighbors to consider for the classification of the centroid (default 15)
        :type n_neighbors: int
        :param dict_in_pixel: if True the centroid are in the same scale than the input csv of spots coorrdinates and rescale with dict_scale if False the centroid are in um (default True)
        :type dict_in_pixel: bool
        :param max_dist_centroid: maximum distance to consider for the centroid (default None)
        :type max_dist_centroid: int
        :param key_pred: key of the node attribute containing the cluster id (default "leiden_merged")
        :type key_pred: str
        :param convex_hull_centroid: check that cell centroid is in the convex hull of its RNA neighbors (default True). If not the cell centroid is not classify to avoid artefact misclassification
        :type convex_hull_centroid: bool
        :param file_extension: file extension of the centroid dictionary
        :type file_extension: str
        :param centroid_csv_key: column name  of the centroid csv file
        :type centroid_csv_key: dict
        :return:
        """

        for img_name in tqdm(self):

            if path_cell_centroid is not None:
                assert self.dataset.dict_centroid is None, "The dict_centroid attribute of the dataset is not None. Please remove it or set it to None."
                if str(file_extension)[-4:] == ".npy":
                    dict_cell_centroid = np.load(Path(path_cell_centroid) / (img_name + file_extension),
                                                 allow_pickle=True).item()
                elif str(file_extension)[-4:] == ".csv":
                    df_centroid = pd.read_csv(Path(path_cell_centroid) / (img_name + file_extension))
                    x_list = list(df_centroid[centroid_csv_key["x"]])
                    y_list = list(df_centroid["y"])
                    if "z" in df_centroid.columns:
                        z_list = list(df_centroid["z"])
                    cell_list = list(df_centroid[centroid_csv_key["cell_index"]])
                    if "z" in df_centroid.columns:
                        dict_cell_centroid = {cell_list[i]: np.array([z_list[i], y_list[i], x_list[i]])
                                              for i in range(len(cell_list))}
                    else:
                        dict_cell_centroid = {cell_list[i]: {"x": x_list[i], "y": y_list[i]} for i in
                                              range(len(cell_list))}
                else:
                    raise ValueError \
                        ("The file extension of path_cell_centroid is not recognized. Please provide a .npy or .csv file")
            else:
                if self.dataset.dict_centroid is None:
                    raise ValueError \
                        ("The dict_centroid attribute of the dataset is None.  Compute the centroid of the cells first. or provide a path to the centroid dataframe or dictionary {cell_index: {z:,y:,x:}} for each image.")
                dict_cell_centroid = self.dataset.dict_centroid[img_name]

            self[img_name].classify_centroid(dict_cell_centroid,
                                             n_neighbors=n_neighbors,
                                             dict_in_pixel=dict_in_pixel,
                                             max_dist_centroid=max_dist_centroid,
                                             key_pred=key_pred,
                                             distance=distance,
                                             )

    #### Apply diskjtra

    def associate_rna2landmark(self,
                               key_pred="leiden_merged",
                               distance='distance',
                               max_cell_radius=100):

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
            print(img_name)
            self[img_name].associate_rna2landmark(
                key_pred=key_pred,
                prior_name=self.prior_name,
                distance=distance,
                max_cell_radius=max_cell_radius)

    def anndata_from_comseg_result(self,
                                   config: dict = None,
                                   min_rna_per_cell=5,
                                   return_polygon=True,
                                   alpha=0.5,
                                   allow_disconnected_polygon=False
                                   ):

        """
        Return an anndata with the estimated expression vector of each cell in the dataset plus the spot positions.

        :param self:
        :param config: dictionary of parameters to overwrite the default parameters, default is None
        :type config: dict
        :param min_rna_per_cell: minimum number of RNA to consider a cell
        :type min_rna_per_cell: int
        :param return_polygon: if True return the polygon of the cells, the polygon are computed using the alphashape library
        :type return_polygon: bool
        :param alpha: alpha parameter to compute the alphashape polygone : https://pypi.org/project/alphashape/.
         alpha is between 0 and 1, 1 correspond to the convex hull of the cell
        :type alpha: float
        :param allow_disconnected_polygon: if True allow disconnected polygon

        :return:
        """
        list_image_name = []
        anndata_list = []
        dict_df_spots = {}
        dict_json_img = {}

        if config is not None:
            print("config dict overwritting the default parameters")
            min_rna_per_cell = config.get("min_rna_per_cell", min_rna_per_cell)
            return_polygon = config.get("return_polygon", return_polygon)
            alpha = config.get("alpha", alpha)
            allow_disconnected_polygon = config.get("allow_disconnected_polygon", allow_disconnected_polygon)

        for img_name in tqdm(self):
            anndata, json_dict = self[img_name].get_anndata_from_result(
                key_cell_pred='cell_index_pred',
                min_rna_per_cell=min_rna_per_cell,
                return_polygon=return_polygon,
                alpha=alpha,
                allow_disconnected_polygon=allow_disconnected_polygon)
            dict_json_img[img_name] = json_dict

            anndata_list.append(anndata)
            list_image_name += [img_name] * len(anndata)
            dict_df_spots[img_name] = anndata.uns["df_spots"]

            assert np.array_equal(anndata.var_names,
                                  self.dataset.selected_genes), "The anndata var names are not the same as the dataset selected genes"

        self.final_anndata = ad.concat(anndata_list)
        self.final_anndata.obs["image_name"] = list_image_name
        self.final_anndata.var["features"] = self.dataset.selected_genes
        self.final_anndata.var_names = self.dataset.selected_genes
        self.final_anndata.uns["df_spots"] = dict_df_spots
        return self.final_anndata, dict_json_img

    def run_all(self,
                config: dict = None,
                k_nearest_neighbors: int = 10,
                max_cell_radius: float = 15,
                ## in situ clutering parameter
                size_commu_min: int = 3,
                norm_vector: bool = False,
                n_pcs: int = 3,
                clustering_method: str = "leiden",
                n_neighbors: int = 20,
                resolution:  float = 1,
                n_clusters_kmeans=4,
                nb_min_cluster: int = 0,
                min_merge_correlation:  float = 0.8,
                ### classify centroid
                path_dataset_folder_centroid: str = None,
                file_extension: str = ".csv",
                ):
        """
        function running all the ComSeg steps: (compute_community_vector(),
        compute_insitu_clustering(), add_cluster_id_to_graph(), classify_centroid(), associate_rna2landmark() )
        :param config: dictionary of parameters to overwrite the default parameters, default is None
        :type config: dict
        :param k_nearest_neighbors: number of nearest neighbors to consider for the KNN graph creation, reduce K to speed computation
        :type k_nearest_neighbors: int
        :param max_cell_radius: maximum distance between a cell centroid and an RNA to be associated
        :type max_cell_radius: float
        :param size_commu_min: minimum number of RNA in a community to be considered for the clustering (default 3)
        :type size_commu_min: int
        :param norm_vector: if True, the expression vector will be normalized using the scTRANSFORM normalization parameters, the normaliztion requires the following R package : sctransform, feather, arrow
        The normalization is important to do on dataset with a high number of gene.
        :type norm_vector: bool
        :param n_pcs: number of principal component to compute for the clustering of the RNA communities expression vector; Lets 0 if no pca
        :type n_pcs: int
        :param clustering_method: choose in ["leiden", "kmeans", "louvain"]
        :type clustering_method: str
        :param n_neighbors: number of neighbors similarity graph of the RNA communities expression vector clustering
        :type n_neighbors: int
        :param resolution:  resolution paramter  for the in-situ-clustering step if louvain or leiden are used
        :type resolution: float
        :param n_clusters_kmeans: number of cluster for the kmeans clustering for ```clustering_method```= "kmeans"
        :type n_clusters_kmeans: int
        :param nb_min_cluster: minimum number of cluster to keep after the merge of the cluster
        :type nb_min_cluster: int
        :param min_merge_correlation: minimum correlation to merge cluster in the in situ clustering
        :type min_merge_correlation: float
        :param path_dataset_folder_centroid: path to the folder containing the centroid in a csv or dictionary {cell : {z:,y:,x:}} for each image, use the same scale than then input csv
        :type path_dataset_folder_centroid: str
        :param file_extension: file extension of the centroid dictionary (.npy) or csv file (.csv)
        :type file_extension: str
        :return:
        """
        if config is not None:
            #print("config dict overwritting the default parameter")
            k_nearest_neighbors = config.get("k_nearest_neighbors", k_nearest_neighbors)
            max_cell_radius = config.get("max_cell_radius", max_cell_radius)
            size_commu_min = config.get("size_commu_min", size_commu_min)
            norm_vector = config.get("norm_vector", norm_vector)
            n_pcs = config.get("n_pcs", n_pcs)
            clustering_method = config.get("clustering_method", clustering_method)
            n_neighbors = config.get("n_neighbors", n_neighbors)
            resolution = config.get("resolution", resolution)
            n_clusters_kmeans = config.get("n_clusters_kmeans", n_clusters_kmeans)
            nb_min_cluster = config.get("nb_min_cluster", nb_min_cluster)
            min_merge_correlation = config.get("min_merge_correlation", min_merge_correlation)
            path_dataset_folder_centroid = config.get("path_dataset_folder_centroid", path_dataset_folder_centroid)
            file_extension = config.get("file_extension", file_extension)

        self.compute_community_vector(k_nearest_neighbors=k_nearest_neighbors)

        self.compute_insitu_clustering(
            size_commu_min=size_commu_min,
            norm_vector=norm_vector,
            ### parameter clustering
            n_pcs=n_pcs,
            n_comps=n_pcs,
            clustering_method=clustering_method,
            n_neighbors=n_neighbors,
            resolution=resolution,
            n_clusters_kmeans=n_clusters_kmeans,
            palette=None,
            nb_min_cluster=nb_min_cluster,
            min_merge_correlation=min_merge_correlation,
        )

        self.add_cluster_id_to_graph(clustering_method="leiden_merged")

        self.classify_centroid(
            path_cell_centroid=path_dataset_folder_centroid,
            n_neighbors=15,
            dict_in_pixel=True,
            key_pred="leiden_merged",
            distance="ngb_distance_weights",
            file_extension=file_extension
        )

        self.associate_rna2landmark(
            key_pred="leiden_merged",
            distance='distance',
            max_cell_radius=max_cell_radius)
