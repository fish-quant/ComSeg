




"""
class set store the graph and anndata of the comseg(s) object
preprocess it (like concatenate anndata) to perform classification
then apply a community classification (in situ clustering class)
"""

#%%


import os
import sys

from tqdm import tqdm

import anndata as ad


sys.path.insert(1, os.getcwd() + "/code/")

#from unused_files.similarity_m import get_simialrity_matrix
#from utils.data_processing import sctransform_from_parameters



from pathlib import Path
import numpy as np

__all__ = ["ComSeg"]




class ComSegDict():

    #todo optimize the memory complexity
    # so far I store many time the same anndata

    def __init__(self):
        return
    ## create directed grap
    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)


    def concatenate_anndata(self):
         self.global_anndata = ad.concat([self[img].community_anndata for img in self
                                          if str(type(self[img])) == "<class 'comseg.model.ComSeg'>"])
         self.global_anndata.obs["img_name"] =  np.concatenate([[img] * len(self[img].community_anndata) for img in self
                                                 if  str(type(self[img])) == "<class 'comseg.model.ComSeg'>"])
         return self.global_anndata



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
        #todo clean the name leiden vs leiden_merged
        # or add the current cleuter name to use in the self
        :param size_commu_min:
        :param norm_vector:
        :param n_pcs:
        :param n_comps:
        :param clustering_method:
        :param n_neighbors:
        :param resolution:
        :param n_clusters_kmeans:
        :param palette:
        :param nb_min_cluster:
        :param min_merge_correlation:
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
        self.in_situ_clustering.cluster_centroid(
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
            if str(type(self[img])) != "<class 'comseg.model.ComSeg'>":
                continue
            ## get the cluster list and the community index
            cluster_id_list = list(self.global_anndata[self.global_anndata.obs.img_name == img].obs[self.clustering_method])
            community_index_list = list(self.global_anndata[self.global_anndata.obs.img_name == img].obs.index_commu)
            assert community_index_list == list(self[img].community_anndata.obs['index_commu'])
            self[img].community_anndata.obs[self.clustering_method] = cluster_id_list
            ## self[img].community_anndata.obs["cluster"] = self.global_anndata.obs["cluster"].loc[self[img].community_anndata.obs.index]
            ## loop on comseg model images
        return self.global_anndata




    ### ADD FOUND CLUSTER FROM Global anndata to node graph



    def add_cluster_id_to_graph(self,
                                clustering_method = "leiden_merged"):

        """
        :param self:
        :return:
        """
        list_img = [img for img in self if str(type(self[img])) == "<class 'comseg.model.ComSeg'>"]
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
        :param self:
        :param dict_cell_centroid:
        :param n_neighbors:
        :param dict_in_pixel:
        :param max_dist_centroid:
        :param key_pred:
        :param distance:
        :param convex_hull_centroid:
        :param prior_keys:
        :return:
        """


        for img_name in tqdm(self):
            if str(type(self[img_name])) != "<class 'comseg.model.ComSeg'>":
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


    #### Aplly diskjtra


    def associate_rna2landmark(self,
        key_pred="leiden_merged",
        super_node_prior_key='in_nucleus',
        distance='distance',
        max_distance=100):
        """

        :param key_pred:
        :param super_node_prior_key:
        :param distance:
        :param max_distance: distanc in um
        :return:
        """

        for img_name in tqdm(self):
            if str(type(self[img_name])) != "<class 'comseg.model.ComSeg'>":
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
        list_image_name = []
        anndata_list = []
        for img_name in tqdm(self):
            if str(type(self[img_name])) != "<class 'comseg.model.ComSeg'>":
                continue
            anndata = self[img_name].get_anndata_from_result(
                key_cell_pred=key_cell_pred)

            anndata_list.append(anndata)
            list_image_name += [img_name] * len(anndata)

        self.final_anndata = ad.concat(anndata_list)
        self.final_anndata.obs["image_name"] = list_image_name

