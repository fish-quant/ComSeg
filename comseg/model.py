


"""
class model, compute the graph
apply community detection
labeled community with the clustering classes
add centroid from the dataset in the graph
apply dikstra to compute the distance between the centroid and the other nodes
return a count matrix of the image
"""

#%%


import os
import sys
from collections import Counter

from tqdm import tqdm
import networkx as nx
import numpy as np
import scipy.sparse as sp

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import anndata as ad


sys.path.insert(1, os.getcwd() + "/code/")

import leidenalg as la
import igraph as ig
#from unused_files.similarity_m import get_simialrity_matrix
#from utils.data_processing import sctransform_from_parameters
from sklearn import metrics
import scipy
import networkx.algorithms.community as nx_comm

import numpy as np

__all__ = ["ComSeg"]

def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density


class ComSeg():

    def __init__(self,
                df_spots_label,
                 selected_genes,
                dict_scale={"x": 0.103, 'y': 0.103, "z": 0.3},  # in micrometer
                mean_cell_diameter=15,  # in micrometer
                max_cell_length=200,  # in micrometer
                k_nearest_neighbors = 40,
                edge_max_length = None,
                eps_min_weight =  0.01,
                resolution =1):
        self.df_spots_label = df_spots_label
        self.dict_scale = dict_scale
        self.k_nearest_neighbors = k_nearest_neighbors
        if edge_max_length is None:
            self.edge_max_length = mean_cell_diameter / 4
        self.eps_min_weight = eps_min_weight
        self.resolution = 1
        self.selected_genes = selected_genes
        self.gene_index_dict = {}
        for gene_id in range(len(selected_genes)):
            self.gene_index_dict[selected_genes[gene_id]] = gene_id

        self.agg_sd =  1
        self.agg_max_dist = mean_cell_diameter/2
    ## create directed graph

    def create_graph(self,
                     #n_neighbors=5, self.k_nearest_neighbors
                     dict_co_expression=None,
                     ):
        try:
            self.df_spots_label = self.df_spots_label.reset_index()
        except Exception as e:
            print(e)

        if "z" in self.df_spots_label.columns:  ## you should do ZYX every where it is error prone, function to chenge : this one rna nuclei assoctioan
            list_coordo_order_no_scaling = np.array([self.df_spots_label.x, self.df_spots_label.y, self.df_spots_label.z]).T
            list_coordo_order = list_coordo_order_no_scaling * np.array([self.dict_scale['x'],
                                                                         self.dict_scale['y'],
                                                                         self.dict_scale["z"]])
        else:
            list_coordo_order_no_scaling = np.array([self.df_spots_label.x, self.df_spots_label.y]).T
            list_coordo_order = list_coordo_order_no_scaling * np.array([self.dict_scale['x'], self.dict_scale['y']])
        dico_list_features = {}
        assert 'gene' in self.df_spots_label.columns
        for feature in self.df_spots_label.columns:
                dico_list_features[feature] = list(self.df_spots_label[feature])
        list_features = list(dico_list_features.keys())
        list_features_order = [(i, {feature: dico_list_features[feature][i] for feature in  list_features}) for i in range(len(self.df_spots_label))]
        dico_features_order = {}
        for node in range(len(list_features_order)):
            dico_features_order[node] = list_features_order[node][1]
        for node in range(len(list_features_order)):
            dico_features_order[node]["nb_mol"] = 1
        print("computing knn")
        nbrs = NearestNeighbors(n_neighbors=self.k_nearest_neighbors,
                                algorithm='ball_tree').fit(list_coordo_order)
        ad = nbrs.kneighbors_graph(list_coordo_order, mode="connectivity") ## can be optimize here
        distance = nbrs.kneighbors_graph(list_coordo_order, mode ='distance')

        ad[distance > self.edge_max_length] = 0
        distance[distance > self.edge_max_length] = 0
        ad.eliminate_zeros()
        distance.eliminate_zeros()

        rows, cols, BOL = sp.find(ad == 1)
        edges_list = list(zip(rows.tolist(), cols.tolist()))
        distance_list = [distance[rows[i], cols[i]] for i in range(len(cols))]
        G = nx.DiGraph()  # oriented graph
        list_features_order = [(k, dico_features_order[k]) for k in dico_features_order]
        G.add_nodes_from(list_features_order)
        print("adding edges")
        for edges_index in tqdm(range(len(edges_list))):
            edges = edges_list[edges_index]
            gene_source = G.nodes[edges[0]]['gene']
            gene_target = G.nodes[edges[1]]['gene']
            G.add_edge(edges[0], edges[1])
            weight = np.max(dict_co_expression[gene_source][gene_target], 0) + self.eps_min_weight   ##
            relative_weight = dict_co_expression[gene_source][gene_target]
            G[edges[0]][edges[1]]["weight"] = weight
            G[edges[0]][edges[1]]["relative_weight"] = relative_weight
            G[edges[0]][edges[1]]["distance"] = distance_list[edges_index]
            G[edges[0]][edges[1]]["gaussian"] =  normal_dist(distance_list[edges_index], mean=0, sd =1)

        self.G = G
        self.list_features_order = np.array(list_features_order) ## for later used ?
        self.list_coordo_order = list_coordo_order ## for later used ?
        self.list_coordo_order_no_scaling = list_coordo_order_no_scaling ## for later used ?
        return G


    ## get community detection vector

    def community_vector(self,
                         clustering_method="louvain_with_prior",
                         weights_name="weight",

                         prior_keys="prior",
                         seed=None,
                         super_node_prior_keys="prior",
                         confidence_level=1,
                         # param for multigrpah leiden
                         ):


        nb_egde_total = len(self.G.edges())

        ### if prior create new graph + matching super-node dico
        if super_node_prior_keys is not None:
            print(f'creation of  super node with {super_node_prior_keys}')
            partition = []
            assert clustering_method in ["louvain_with_prior"]
            list_nodes = np.array([index for index, data in self.G.nodes(data=True)])

            array_super_node_prior = np.array([data[super_node_prior_keys] for index, data in self.G.nodes(data=True)])
            unique_super_node_prior = np.unique(array_super_node_prior)
            if 0 in unique_super_node_prior:
                assert unique_super_node_prior[0] == 0
                unique_super_node_prior = unique_super_node_prior[1:]
                list_nodes[array_super_node_prior == 0]
                partition += [{u} for u in list_nodes[array_super_node_prior == 0]]
            for super_node in unique_super_node_prior:
                list_nodes[array_super_node_prior == super_node]
                partition += [set(list_nodes[array_super_node_prior == super_node])]
        else:
            partition = None


        assert nx.is_directed(self.G)
        if clustering_method == "louvain":
            comm = nx_comm.louvain_communities(self.G.to_undirected(reciprocal=False),
                                               weight=weights_name,
                                               resolution=self.resolution,
                                               seed=seed)


        if clustering_method == "louvain_with_prior":
            from .utils import custom_louvain
            comm, final_graph = custom_louvain.louvain_communities(
                    G=self.G.to_undirected(reciprocal=False),
                    weight=weights_name,
                    resolution=self.resolution,
                    threshold=0.0000001,
                    seed=seed,
                    partition=partition,
                    prior_key=prior_keys,
                    confidence_level=confidence_level)

        list_expression_vectors = []
        list_coordinates = []
        list_node_index = []
        list_prior = []
        for index_commu in tqdm(range(len(comm))):
            cluster_coordinate = []
            expression_vector = np.bincount([self.gene_index_dict[self.G.nodes[ind_node]["gene"]] for ind_node in comm[index_commu]],
                                            minlength = len(self.gene_index_dict))
            for node in comm[index_commu]:
                if self.G.nodes[node]['gene'] == "centroid":
                        continue
                self.G.nodes[node]["index_commu"] =  index_commu
                if clustering_method == "louvain_with_prior":
                    self.G.nodes[node]["index_commu_in_nucleus"] =  final_graph.nodes[index_commu]['prior_index']
                if "z" in self.G.nodes[0]:
                    cluster_coordinate.append([self.G.nodes[node]['x'],
                                               self.G.nodes[node]['y'],
                                               self.G.nodes[node]['z']])
                else:
                    cluster_coordinate.append([self.G.nodes[node]['x'], self.G.nodes[node]['y']])

            ### transform it as an andata object

            #" count matrix
            #obs list_coord ,list  node index , prior
            list_expression_vectors.append(expression_vector)
            list_coordinates.append(cluster_coordinate)
            list_node_index.append(comm[index_commu])
            list_prior.append(final_graph.nodes[index_commu]['prior_index'])

        count_matrix_anndata = np.array(list_expression_vectors)


        anndata = ad.AnnData(csr_matrix(count_matrix_anndata))
        anndata.var["features"] = self.selected_genes
        anndata.var_names = self.selected_genes
        anndata.obs["list_coord"] =  list_coordinates
        anndata.obs["node_index"] = list_node_index
        anndata.obs["prior"] = list_prior
        anndata.obs["index_commu"] = range(len(comm))


        assert nb_egde_total == len(self.G.edges()) # check it is still the same graph
        self.community_anndata = anndata
        self.estimation_density_vec(
                               # max_dist = 5,
                               key_word="kernel_vector",
                               remove_self_node=True,
                               norm_gauss=True)

        return self.community_anndata



    def estimation_density_vec(self,
                               #max_dist = 5,
                                key_word = "kernel_vector",
                               remove_self_node = True,
                               norm_gauss = True):
        import numpy as np
        import scipy.spatial as spatial
        def normal_dist(x, mean, sd, norm_gauss=False):  # copy from knn_to_count
            if norm_gauss:
                prob_density = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
            else:
                prob_density = np.exp(-0.5 * ((x - mean) / sd) ** 2)
            return prob_density
        point_tree = spatial.cKDTree(self.list_coordo_order)
        list_nn = point_tree.query_ball_point(self.list_coordo_order, self.agg_max_dist)
        for node_index, node_data in tqdm(self.G.nodes(data=True)): ## create a kernel density estimation for each node
            if node_data["gene"] == 'centroid':
                continue
            if remove_self_node:
                list_nn[node_index].remove(node_index)
            list_nn_node_index = list_nn[node_index]
            array_distance = spatial.distance.cdist([self.list_coordo_order[node_index]],
                                                    self.list_coordo_order[list_nn_node_index],
                                                    'euclidean')
            array_normal_distance = normal_dist(array_distance[0],
                                                mean=0,
                                                sd=self.agg_sd,
                                                norm_gauss = norm_gauss)
            ## add code to remove centroid but there is no centroid in list coordo ?
            nn_expression_vector = np.bincount([self.gene_index_dict[self.G.nodes[node_nn]["gene"]] for node_nn in list_nn_node_index],
                                               weights=array_normal_distance,
                                               minlength=len(self.gene_index_dict))
            self.G.nodes[node_index][key_word] = nn_expression_vector
        ## to factorize with community_nn_message_passing_agg
        list_expression_vectors = []
        for comm_index in range(len(self.community_anndata.obs["index_commu"])):
            nn_expression_vector = np.zeros(len(self.gene_index_dict))
            for node in self.community_anndata.obs["node_index"][comm_index]:
                nn_expression_vector += self.G.nodes[node][key_word]
            list_expression_vectors.append(nn_expression_vector)
        count_matrix_anndata = np.array(list_expression_vectors)
        anndata = ad.AnnData(csr_matrix(count_matrix_anndata))
        anndata.var["features"] = self.selected_genes
        anndata.var_names = self.selected_genes
        anndata.obs["list_coord"] =  self.community_anndata.obs["list_coord"]
        anndata.obs["node_index"] = self.community_anndata.obs["prior"]
        anndata.obs["prior"] = self.community_anndata.obs["prior"]
        anndata.obs["index_commu"] = self.community_anndata.obs["index_commu"]
        anndata.obs["nb_rna"] = np.asarray((np.sum(self.community_anndata.X, axis = 1).astype(int)))

        self.community_anndata = anndata
        return anndata
    ##
