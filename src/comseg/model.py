"""
class model, compute the graph
apply community detection
labeled community with the clustering classes
add centroid from the dataset in the graph
apply dikstra to compute the distance between the centroid and the other nodes
return a count matrix of the image
"""

#%%

from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import anndata as ad
import pandas as pd
import alphashape

import networkx.algorithms.community as nx_comm
import numpy as np
from sklearn.utils.extmath import weighted_mode
from scipy.spatial import ConvexHull, Delaunay
from .utils import custom_louvain  ## for dev mode

__all__ = ["ComSegGraph"]

import shapely


#def normal_dist(x, mean, sd):
#    prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
#    return prob_density


class ComSegGraph():
    """
    Class to the generate the graph of RNA spots from a CSV file/image
    this class is in charge of :

    #. create the graph
    #. apply community detection / graph partitioning
    #. Compute the community expression vector :math:`V_c`
    #. add to the communities the labels/cell types computed by the clustering of :math:`V_c` the by the :func:`InSituClustering` class
    #. add the centroid of the cells in the graph
    #. associate RNAs to cell
    #. compute the cell-by-gene matrix of the input sample
    """

    def __init__(self,
                 df_spots_label,
                 selected_genes,
                 dict_co_expression,
                 dict_scale={"x": 0.103, 'y': 0.103, "z": 0.3},  # in micrometer
                 mean_cell_diameter=15,  # in micrometer
                 k_nearest_neighbors=10,
                 edge_max_length=None,
                 gene_column="gene",
                 prior_name="in_nucleus",
                 ):

        """
        :param df_spots_label: dataframe of the spots with column x,y,z, gene and optionally the prior label column
        :type df_spots_label: pd.DataFrame
        :param selected_genes: list of genes to consider
        :type selected_genes: list[str]
        :param dict_scale: dictionary containing the pixel/voxel size of the images in µm default is {"x": 0.103, 'y': 0.103, "z": 0.3}
        :type dict_scale: dict
        :param mean_cell_diameter: the expected mean cell diameter in µm default is 15µm
        :type mean_cell_diameter: float
        :param k_nearest_neighbors: number of nearest neighbors to consider for the graph construction default is 10
        :type k_nearest_neighbors: int
        :param edge_max_length: default is mean_cell_diameter / 4
        :type edge_max_length: float
        """

        self.df_spots_label = df_spots_label
        self.dict_co_expression = dict_co_expression
        self.dict_scale = dict_scale
        self.k_nearest_neighbors = k_nearest_neighbors
        if edge_max_length is None:
            self.edge_max_length = mean_cell_diameter / 4
        self.resolution = 1
        self.selected_genes = selected_genes
        self.gene_index_dict = {}
        for gene_id in range(len(selected_genes)):
            self.gene_index_dict[selected_genes[gene_id]] = gene_id
        self.agg_sd = 1
        self.agg_max_dist = mean_cell_diameter / 2
        self.dico_xyz_index = {"x": 2, "y": 1, "z": 0}
        self.mean_cell_diameter = mean_cell_diameter
        self.gene_column = gene_column
        self.prior_name = prior_name

    ## create directed graph

    def create_graph(self):
        """
        create the graph of the RNA nodes, all the graph generation parameters are set in the __init__() function

        :return: a graph of the RNA spots
        :rtype: nx.Graph
        """

        try:
            self.df_spots_label = self.df_spots_label.reset_index()
        except Exception as e:
            print(e)

        if "z" in self.df_spots_label.columns:  ## you should do ZYX every where it is error prone, function to chenge : this one rna nuclei assoctioan
            list_coordo_order_no_scaling = np.array \
                ([self.df_spots_label.z, self.df_spots_label.y, self.df_spots_label.x]).T
            list_coordo_order = list_coordo_order_no_scaling * np.array([self.dict_scale['z'],
                                                                         self.dict_scale['y'],
                                                                         self.dict_scale["x"]])
        else:
            list_coordo_order_no_scaling = np.array([self.df_spots_label.x, self.df_spots_label.y]).T
            list_coordo_order = list_coordo_order_no_scaling * np.array([self.dict_scale['y'], self.dict_scale['x']])
        dico_list_features = {}
        assert self.gene_column in self.df_spots_label.columns, f"the column {self.gene_column} is not in the dataframe"
        for feature in self.df_spots_label.columns:
            dico_list_features[feature] = list(self.df_spots_label[feature])
        list_features = list(dico_list_features.keys())
        list_features_order = [(i, {feature: dico_list_features[feature][i] for feature in list_features}) for i in
                               range(len(self.df_spots_label))]
        dico_features_order = {}
        for node in range(len(list_features_order)):
            dico_features_order[node] = list_features_order[node][1]
        for node in range(len(list_features_order)):
            dico_features_order[node]["nb_mol"] = 1
        nbrs = NearestNeighbors(n_neighbors=self.k_nearest_neighbors,
                                algorithm='ball_tree').fit(list_coordo_order)
        ad = nbrs.kneighbors_graph(list_coordo_order, mode="connectivity")  ## can be optimize here
        distance = nbrs.kneighbors_graph(list_coordo_order, mode='distance')

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
        for edges_index in range(len(edges_list)):
            edges = edges_list[edges_index]
            gene_source = G.nodes[edges[0]][self.gene_column]
            gene_target = G.nodes[edges[1]][self.gene_column]
            G.add_edge(edges[0], edges[1])
            weight = self.dict_co_expression[gene_source][gene_target]
            G[edges[0]][edges[1]]["weight"] = weight
            G[edges[0]][edges[1]]["distance"] = distance_list[edges_index]
            #G[edges[0]][edges[1]]["gaussian"] = normal_dist(distance_list[edges_index], mean=0, sd=1)

        self.G = G
        self.list_features_order = np.array(list_features_order)  ## for later used ?
        self.list_coordo_order = list_coordo_order  # for later used ?
        self.list_coordo_order_no_scaling = list_coordo_order_no_scaling  ## for later used ?
        return G

    ## get community detection vector

    def community_vector(self,
                         clustering_method="with_prior",
                         seed=None,
                         prior_name="in_nucleus",
                         ):

        """
        Partition the graph into communities/sets of RNAs and computes and stores the "community expression vector"
         in the ``community_anndata`` class attribute
        :param clustering_method: choose in ["with_prior",  "louvain"]. "with_prior" is our graph partitioning / community
                detection method modify from Louvain, taking into account prior knowledge from landmarks like nuclei or cell mask.
        :type clustering_method: str
        :param seed: (optional) seed for the graph partitioning initialization
        :type seed: int
        :param prior_name: (optional) Name of the prior cell assignment column the input CSV file. Node with the same prior label will be merged into a super node.
        node with different prior label can not be merged during the modularity optimization.
        :type prior_name: str
        :return: a graph with a new node attribute "community" with the community detection vector
        :rtype: nx.Graph
        """
        confidence_level = 1  ##### experimental parameter

        nb_egde_total = len(self.G.edges())
        if prior_name is not None:
            partition = []
            assert clustering_method in ["with_prior"]
            list_nodes = np.array([index for index, data in self.G.nodes(data=True)])
            array_super_node_prior = np.array([data[prior_name] for index, data in self.G.nodes(data=True)])
            unique_super_node_prior = np.unique(array_super_node_prior)
            if 0 in unique_super_node_prior:
                assert unique_super_node_prior[0] == 0
                unique_super_node_prior = unique_super_node_prior[1:]
                partition += [{u} for u in list_nodes[array_super_node_prior == 0]]
            for super_node in unique_super_node_prior:
                partition += [set(list_nodes[array_super_node_prior == super_node])]
        else:
            partition = None

        assert nx.is_directed(self.G)
        G = self.G.copy()
        G.remove_edges_from([(d[0], d[1]) for d in list(G.edges(data=True)) if d[2]["weight"] < 0])
        if clustering_method == "louvain":
            comm = nx_comm.louvain_communities(G.to_undirected(reciprocal=False),
                                               weight="weight",
                                               resolution=self.resolution,
                                               seed=seed)

        if clustering_method == "with_prior":
            comm, final_graph = custom_louvain.louvain_communities(
                G=G.to_undirected(reciprocal=False),
                weight="weight",
                resolution=self.resolution,
                threshold=0.0000001,
                seed=seed,
                partition=partition,
                prior_key=prior_name,
                confidence_level=confidence_level)

        list_expression_vectors = []
        list_coordinates = []
        list_node_index = []
        list_prior = []

        for index_commu in range(len(comm)):
            cluster_coordinate = []
            expression_vector = np.bincount \
                ([self.gene_index_dict[self.G.nodes[ind_node][self.gene_column]] for ind_node in comm[index_commu]],
                 minlength=len(self.gene_index_dict))
            for node in comm[index_commu]:
                if self.G.nodes[node][self.gene_column] == "centroid":
                    continue
                self.G.nodes[node]["index_commu"] = index_commu
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
        anndata.obs["list_coord"] = list_coordinates
        anndata.obs["node_index"] = list_node_index
        anndata.obs["prior"] = list_prior
        anndata.obs["index_commu"] = range(len(comm))

        assert nb_egde_total == len(self.G.edges())  # check it is still the same graph
        self.community_anndata = anndata
        self._estimation_density_vec(
            key_word="kernel_vector",
            remove_self_node=True,
            norm_gauss=False,
            distance_weight_mode="linear")

        return self.community_anndata

    def add_cluster_id_to_graph(self,
                                dict_cluster_id,
                                clustering_method="leiden_merged"):
        """

        add transcriptional cluster id to each RNA molecule in the graph

        :param dict_cluster_id: dict {index_commu : cluster_id}
        :type dict_cluster_id: dict
        :param clustering_method: clustering method used to get the community
        :type clustering_method: str
        :return:
        :rtype: nx.Graph
        """

        list_index_commu = list(self.community_anndata.obs['index_commu'])
        dico_commu_cluster = {}
        for index_commu in dict_cluster_id:
            dico_commu_cluster[list_index_commu[index_commu]] = dict_cluster_id[index_commu]
        G = self.G
        for node in tqdm(G.nodes()):
            if G.nodes[node][self.gene_column] == 'centroid':
                continue
            G.nodes[node][clustering_method] = str(dico_commu_cluster[G.nodes[node]['index_commu']])
        self.G = G

    def _estimation_density_vec(self,
                                #max_dist = 5,
                                key_word="kernel_vector",
                                remove_self_node=True,
                                norm_gauss=False,
                                distance_weight_mode="exp"):
        import numpy as np
        import scipy.spatial as spatial
        def normal_dist(x, mean, sd, norm_gauss=False):  # copy from knn_to_count
            if norm_gauss:
                #prob_density = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
                raise Exception("not implemented")
            else:
                prob_density = np.exp(-0.5 * ((x - mean) / sd) ** 2)
            return prob_density

        point_tree = spatial.cKDTree(self.list_coordo_order)
        if distance_weight_mode == "exp":
            list_nn = point_tree.query_ball_point(self.list_coordo_order, self.agg_max_dist)
        if distance_weight_mode == "linear":
            list_nn = point_tree.query_ball_point(self.list_coordo_order, self.edge_max_length)
        for node_index, node_data in self.G.nodes(data=True):  ## create a kernel density estimation for each node
            if distance_weight_mode == "None":
                nn_expression_vector = np.zeros(len(self.selected_genes))
                nn_expression_vector[self.gene_index_dict[self.G.nodes[node_index][self.gene_column]]] = 1
                self.G.nodes[node_index][key_word] = np.ones(len(self.selected_genes))
                continue
            if node_data[self.gene_column] == 'centroid':
                continue
            if remove_self_node:
                list_nn[node_index].remove(node_index)
            list_nn_node_index = list_nn[node_index]
            array_distance = spatial.distance.cdist([self.list_coordo_order[node_index]],
                                                    self.list_coordo_order[list_nn_node_index],
                                                    'euclidean')

            if distance_weight_mode == "exp":
                distance_weights = normal_dist(array_distance[0],
                                               mean=0,
                                               sd=self.agg_sd,
                                               norm_gauss=norm_gauss)
            if distance_weight_mode == "linear":
                distance_weights = (self.edge_max_length - array_distance[0]) / self.edge_max_length
            ## add code to remove centroid but there is no centroid in list coordo ?
            nn_expression_vector = np.bincount \
                ([self.gene_index_dict[self.G.nodes[node_nn][self.gene_column]] for node_nn in list_nn_node_index],
                 weights=distance_weights,
                 minlength=len(self.gene_index_dict))
            self.G.nodes[node_index][key_word] = nn_expression_vector
        ## to factorize with community_nn_message_passing_agg
        list_expression_vectors = []
        for comm_index in range(len(self.community_anndata.obs["index_commu"])):
            nn_expression_vector = np.zeros(len(self.gene_index_dict))
            for node in self.community_anndata.obs["node_index"][comm_index]:
                nn_expression_vector += self.G.nodes[node][key_word]
            nn_expression_vector = nn_expression_vector / len(self.community_anndata.obs["node_index"][comm_index])
            list_expression_vectors.append(nn_expression_vector)
        count_matrix_anndata = np.array(list_expression_vectors)
        anndata = ad.AnnData(csr_matrix(count_matrix_anndata))
        anndata.var["features"] = self.selected_genes
        anndata.var_names = self.selected_genes
        anndata.obs["list_coord"] = self.community_anndata.obs["list_coord"]
        anndata.obs["node_index"] = self.community_anndata.obs["node_index"]
        anndata.obs["prior"] = self.community_anndata.obs["prior"]
        anndata.obs["index_commu"] = self.community_anndata.obs["index_commu"]
        anndata.obs["nb_rna"] = np.asarray((np.sum(self.community_anndata.X, axis=1).astype(int)))

        self.community_anndata = anndata
        return anndata

    ### add centroids to the graph
    def add_centroids(self, dict_cell_centroid):

        """
        add centroids to the graph
        :param dict_cell_centroid: dict of centroid coordinate  {cell : {z:,y:,x:}}
        :type dict_cell_centroid: dict
        :return:
        """
        for cell, centroid in dict_cell_centroid.items():
            centroid = np.array(centroid)
            if centroid.ndim == 2:
                centroid = np.mean(centroid, axis=0)
            assert centroid.ndim == 1
            self.G.add_node(len(self.G) - 1 + int(cell),
                            # gene = "centroid",
                            cell=cell,
                            in_nucleus=cell,
                            z=centroid[self.dico_xyz_index["z"]],
                            y=centroid[self.dico_xyz_index["y"]],
                            x=centroid[self.dico_xyz_index["x"]])
            self.G.nodes[len(self.G.nodes) - 1][self.gene_column] = "centroid"
            self.dict_cell_centroid = dict_cell_centroid
        return self.G

    ### add classify it
    def classify_centroid(self,
                          dict_cell_centroid,
                          n_neighbors=15,
                          dict_in_pixel=True,
                          max_dist_centroid=None,
                          key_pred="leiden_merged",
                          distance="ngb_distance_weights",
                          ):
        """
        classify cells centroid based on their  neighbor RNAs labels from ``add_cluster_id_to_graph()``

        :param dict_cell_centroid: dict of centroid coordinate  {cell : {z:,y:,x:}}
        :type dict_cell_centroid: dict
        :param n_neighbors: number of neighbors to consider for the classification of the centroid (default 15)
        :type n_neighbors: int
        :param dict_in_pixel: if True the centroid are in pixel and rescals if False the centroid are in um (default True)
        :type dict_in_pixel: bool
        :param max_dist_centroid: maximum distance to consider for the centroid, if None it is set to mean_cell_diameter / 2
        :type max_dist_centroid: int
        :param key_pred: key-name of the node attribute containing the cluster id (default "leiden_merged")
        :type key_pred: str
        :param distance: leave it to "ngb_distance_weights" (default "ngb_distance_weights")
        :type distance: str

        :return: self.G
        :rtype: nx.Graph
        """

        self.dict_cell_centroid = dict_cell_centroid

        if max_dist_centroid is None:
            max_dist_centroid = self.mean_cell_diameter / 2
        nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                                algorithm='ball_tree').fit(self.list_coordo_order)
        if dict_in_pixel:
            list_coordo_order_nuc_centroid_no_scaling = []
            list_coordo_order_nuc_centroid = []
            for nuc in self.dict_cell_centroid:
                if len(self.dict_cell_centroid[nuc]) == 1:
                    centroid_pix = np.array(
                        self.dict_cell_centroid[nuc][0])  ## to clean, in case of old version of dict
                else:
                    assert len(self.dict_cell_centroid[nuc]) == len(self.list_coordo_order[0])
                    centroid_pix = np.array(self.dict_cell_centroid[nuc])
                centroid_um = centroid_pix * np.array([self.dict_scale['z'],
                                                       self.dict_scale['y'],
                                                       self.dict_scale["x"]])
                list_coordo_order_nuc_centroid_no_scaling.append(centroid_pix)
                list_coordo_order_nuc_centroid.append(centroid_um)
        else:
            list_coordo_order_nuc_centroid = []
            for nuc in self.dict_cell_centroid:
                assert len(self.dict_cell_centroid[nuc]) == len(self.list_coordo_order[0])
                centroid_um = self.dict_cell_centroid[nuc]
                list_coordo_order_nuc_centroid.append(centroid_um)

        ad_nuc_centroid = nbrs.kneighbors_graph(list_coordo_order_nuc_centroid)  ## can be optimize here
        distance_nuc_centroid = nbrs.kneighbors_graph(list_coordo_order_nuc_centroid, mode='distance')

        dico_nuclei_centroid = {}
        for nuc_index in range(len(self.dict_cell_centroid)):

            nuc = list(self.dict_cell_centroid.keys())[nuc_index]
            centroid = self.dict_cell_centroid[nuc]
            centroid = np.array(centroid)
            if centroid.ndim == 2:
                centroid = np.mean(centroid, axis=0)

            dico_nuclei_centroid[nuc] = {}
            dico_nuclei_centroid[nuc]['z'] = centroid[self.dico_xyz_index["z"]]
            dico_nuclei_centroid[nuc]['y'] = centroid[self.dico_xyz_index["y"]]
            dico_nuclei_centroid[nuc]['x'] = centroid[self.dico_xyz_index["x"]]

            dico_nuclei_centroid[nuc]["type_list"] = []
            dico_nuclei_centroid[nuc]["gr_type_list"] = []
            dico_nuclei_centroid[nuc]["ngb_distance"] = []
            dico_nuclei_centroid[nuc]["ngb_gr_cell"] = []
            dico_nuclei_centroid[nuc]["ngb_distance_weights"] = []
            dico_nuclei_centroid[nuc]["gaussian"] = []
            dico_nuclei_centroid[nuc][self.gene_column] = "centroid"
            dico_nuclei_centroid[nuc]["cell"] = nuc
            dico_nuclei_centroid[nuc]["in_nucleus"] = nuc
            dico_nuclei_centroid[nuc]["index_commu_in_nucleus"] = nuc
            type_list = []
            array_index_nn = np.nonzero(ad_nuc_centroid[nuc_index].toarray())[1]
            index_type_list = []
            for index_nn_centroid in array_index_nn:
                if max_dist_centroid is not None:
                    if distance_nuc_centroid[nuc_index, index_nn_centroid] > max_dist_centroid:
                        continue
                index_type_list.append(index_nn_centroid)
                type_list.append(self.G.nodes[index_nn_centroid])
                dico_nuclei_centroid[nuc]["type_list"].append(self.G.nodes[index_nn_centroid][key_pred])
                dico_nuclei_centroid[nuc]["ngb_distance"].append(
                    distance_nuc_centroid[nuc_index, index_nn_centroid])
                dico_nuclei_centroid[nuc]["ngb_distance_weights"].append(
                    max_dist_centroid - distance_nuc_centroid[nuc_index, index_nn_centroid])
                dico_nuclei_centroid[nuc]['nn_graph_indice'] = array_index_nn

            type_list = np.array(dico_nuclei_centroid[nuc]["type_list"])
            weights_list = np.array(dico_nuclei_centroid[nuc][distance])[type_list != None]
            type_list = type_list[type_list != None]
            pred_cluster = weighted_mode(a=type_list,
                                         w=weights_list)[0][0] if len(
                type_list) > 0 else "unknown"
            dico_nuclei_centroid[nuc][key_pred] = pred_cluster

            ### add new attribute to node centroid

            node_index = len(self.list_coordo_order) - 1 + nuc
            self.G.add_nodes_from([(node_index, dico_nuclei_centroid[nuc])])
            for ii in array_index_nn:
                self.G.add_edge(node_index, ii)
                if nuc == self.G.nodes[ii][self.prior_name]:
                    self.G.add_edge(node_index, ii, distance=0)
                else:
                    self.G.add_edge(node_index, ii, distance=distance_nuc_centroid[nuc_index, ii])

        return self.G

    ###### RNA landmark  association ######

    def associate_rna2landmark(self,
                               key_pred="leiden_merged",
                               prior_name='in_nucleus',
                               distance='distance',
                               max_cell_radius=100):

        """

        Associate RNA to cell based on the both transcriptomic landscape and the
        distance between the RNAs and the centroid of the cell

        :param key_pred: key of the node attribute containing the cluster id (default "leiden_merged")
        :type key_pred: str
        :param prior_name: prior_name attribut used in community_vector
        :type prior_name: str
        :param max_distance: maximum distance between a cell centroid and an RNA to be associated (default 100)
        :type max_distance: float
        :return: self.G
        :rtype nx.Graph
        """
        ## merge supernodes belonging to the nuclei landmark
        from .utils.utils_graph import _gen_graph
        G_merge = _gen_graph(graph=self.G.copy(),
                             super_node_prior_key=prior_name,
                             distance=distance,
                             key_pred=key_pred,
                             )
        G_merge, dico_expression_m_merge = self._associate_rna2landmark(
            G=G_merge,
            distance=distance,
            max_cell_radius=max_cell_radius,
        )
        for super_node_index in G_merge.nodes():
            set_super_nodes = G_merge.nodes[super_node_index]['nodes']
            for simple_nodes in set_super_nodes:
                self.G.nodes[simple_nodes]["cell_index_pred"] = G_merge.nodes[super_node_index]["cell_index_pred"]
                self.G.nodes[simple_nodes]["distance2centroid"] = G_merge.nodes[super_node_index]["distance2centroid"]
        return self.G

    def _associate_rna2landmark(self,
                                G,
                                distance='distance',
                                max_cell_radius=100,  ## in theunit of the graph distance ie um
                                ):

        """

        v3 same than v1 but use the prior key instead of the centroid
        Parameters
        ----------
        G :
        df_spots_label :
        scrna_unique_clusters : scrna_unique_clusters from in situ clustering
        Returns
        ----------
        """

        #print(f'max distance is {max_distance}')

        nn_find = 0
        find = 0
        dico_expression_m = {}  ## store {nuc: nodes}
        nb_centroid = 0
        scrna_unique_clusters = []
        for node_index in G.nodes():
            if G.nodes[node_index]["key_pred"] is None:
                raise ValueError("no pred, should be at least -1")
            scrna_unique_clusters.append(G.nodes[node_index]["key_pred"])
        scrna_unique_clusters = np.unique(scrna_unique_clusters)
        for celltype in scrna_unique_clusters:
            ## get the node of celltype
            list_nodes_index = [node_index for node_index in G.nodes() if
                                G.nodes[node_index][
                                    "key_pred"] == celltype]  # or celltype in G.nodes[node_index][key_pred]] à quoi sert cette ligne
            subgraph = G.subgraph(list_nodes_index).copy().to_undirected()
            # print(top)
            centroid_list = [n for n, y in subgraph.nodes(data=True) if y["super_node_prior_key"] != 0]
            nb_centroid += len(centroid_list)
            for cc in list(nx.connected_components(subgraph)):
                # print(f'nb node in cc {len(cc)}')
                if len(set(centroid_list).intersection(cc)) == 1:
                    nucleus_node = list(set(centroid_list).intersection(cc))[0]
                    length, path = nx.single_source_dijkstra(G.subgraph(cc).to_undirected(),
                                                             nucleus_node,
                                                             weight=distance,
                                                             cutoff=max_cell_radius)

                    dico_expression_m[nucleus_node] = [[node, length[node]] for node in length.keys()]
                    find += len(list(length.keys()))
                if len(set(centroid_list).intersection(cc)) > 1:
                    # break
                    list_nuclei = list(set(centroid_list).intersection(cc))
                    dico_length = {}  # {centroid : length list}
                    dico_shortest_path = {}
                    # print(f'list_nuclei len  {len(list_nuclei)}')

                    for nucleus_node in list_nuclei:
                        dico_expression_m[nucleus_node] = []
                        length, path = nx.single_source_dijkstra(G.subgraph(cc).to_undirected(),
                                                                 nucleus_node,
                                                                 weight=distance,
                                                                 cutoff=max_cell_radius)
                        dico_length[nucleus_node] = length
                        dico_shortest_path[nucleus_node] = path
                    dico_nodes_centroid_distance = {}
                    for node in cc:
                        dico_nodes_centroid_distance[node] = {}
                    for nucleus_node in list_nuclei:
                        for node in dico_length[nucleus_node]:
                            if node in dico_length[nucleus_node]:
                                dico_nodes_centroid_distance[node][nucleus_node] = dico_length[nucleus_node][node]
                    for node in cc:
                        try:
                            if len(dico_nodes_centroid_distance[node]) > 0:
                                nearest_c = min(dico_nodes_centroid_distance[node],
                                                key=dico_nodes_centroid_distance[node].get)
                                distance2centroid = dico_nodes_centroid_distance[node][nearest_c]
                                dico_expression_m[nearest_c].append((node, distance2centroid))
                                find += 1
                        except ValueError as e:
                            print(e)
                            raise ValueError(f"node {node} not find in dikjtra")
        ## add new_label to the grpa
        for node_all in G.nodes():
            G.nodes[node_all]["cell_index_pred"] = 0
            G.nodes[node_all]["distance2centroid"] = np.inf
        for centroid in dico_expression_m:
            nuc_index = G.nodes[centroid]["super_node_prior_key"]
            for value_exp in dico_expression_m[centroid]:
                node = value_exp[0]
                distance2centroid = value_exp[1]
                G.nodes[node]["cell_index_pred"] = nuc_index
                G.nodes[node]["distance2centroid"] = distance2centroid

        return G, dico_expression_m

    ##### generate an anndata from the graph

    #### return an anndata with expresion vector
    ## as obs image name | centroid coordinate |
    # list of rna spots coordinate |
    # list of the corresponding rna species
    #

    def get_anndata_from_result(
            self,
            key_cell_pred='cell_index_pred',
            return_polygon=False,
            alpha=0.5,
            min_rna_per_cell=5,
            allow_disconnected_polygon=False
    ):
        """
        Generate an anndata storing the estimated expression vector and their spots coordinates

        :param key_cell_pred:  key of the cell prediction in the graph (default cell_index_pred)
        :type key_cell_pred: str
        :param return_polygon: if True return the polygon of the cells
        :type return_polygon: bool
        :param alpha: alpha parameter to compute the alphashape polygone : https://pypi.org/project/alphashape/.
         alpha is between 0 and 1, 1 correspond to the convex hull of the cell
        :type alpha: float
        :param min_rna_per_cell: minimum number of RNA to consider a cell
        :type min_rna_per_cell: int
        :param allow_disconnected_polygon: if True allow disconnected polygon
        :return:
        """

        dict_cell_genes = {}
        dict_cell_genes_coordinate = {}
        dict_cell_genes_name = {}
        dict_cell_genes_mol_index = {}
        dict_cell_dist2centroid = {}

        for cell in self.dict_cell_centroid:
            dict_cell_genes[cell] = []
            dict_cell_genes_coordinate[cell] = []
            dict_cell_genes_name[cell] = []
            dict_cell_genes_mol_index[cell] = []
            dict_cell_dist2centroid[cell] = []

        for node_index, node_data in self.G.nodes(data=True):
            if node_data[self.gene_column] != 'centroid' and node_data[key_cell_pred] != 0:
                if node_data[
                    key_cell_pred] in dict_cell_genes:  ## check that the centroid is indeed in the crop and not only the prior
                    dict_cell_genes[node_data[key_cell_pred]].append(node_data[self.gene_column])
                    dict_cell_genes_coordinate[node_data[key_cell_pred]].append([node_data["z"],
                                                                                 node_data['y'],
                                                                                 node_data['x']])
                    dict_cell_dist2centroid[node_data[key_cell_pred]].append(node_data["distance2centroid"])
                    dict_cell_genes_mol_index[node_data[key_cell_pred]].append(node_data['index'])

        list_expression_vectors = []
        list_cell_id = []
        list_cell_genes_coordinate = []
        list_genes_name = []
        list_gene_index = []
        list_cell_centroid = []
        list_gene_dist2centroid = []
        for cell in dict_cell_genes:
            expression_vector = np.bincount(
                [self.gene_index_dict[gene] for gene in dict_cell_genes[cell]]
                , minlength=len(self.gene_index_dict))
            if np.sum(expression_vector) < min_rna_per_cell:
                continue
            list_expression_vectors.append(expression_vector)
            list_cell_id.append(cell)
            list_cell_genes_coordinate.append(
                np.array(dict_cell_genes_coordinate[cell])
            )
            list_genes_name.append(dict_cell_genes[cell])
            list_gene_index.append(dict_cell_genes_mol_index[cell])
            list_gene_dist2centroid.append(dict_cell_dist2centroid[cell])
            centroid = self.dict_cell_centroid[cell]
            centroid = np.array(centroid)
            if centroid.ndim == 2:
                centroid = np.mean(centroid, axis=0)
            list_cell_centroid += [tuple(centroid)]

        anndata = ad.AnnData(np.array(list_expression_vectors))
        anndata.var["features"] = self.selected_genes
        anndata.var_names = self.selected_genes
        anndata.var["Name"] = self.selected_genes
        #
        random_string = str(np.random.randint(0, 10000))

        anndata.obs["CellID"] = list_cell_id
        anndata.obs["Name"] = [random_string + "_" + str(i) for i in list_cell_id]
        anndata.obs.index = [random_string + "_" + str(i) for i in list_cell_id]
        anndata.obs_names = [random_string + "_" + str(i) for i in list_cell_id]
        if len(list_cell_centroid[0]) == 2:
            x_list_centroid = [x[1] for x in list_cell_centroid]
            y_list_centroid = [x[0] for x in list_cell_centroid]
            anndata.obs["centroid_y"] = y_list_centroid
            anndata.obs["centroid_x"] = x_list_centroid
        else:
            assert len(list_cell_centroid[0]) == 3
            x_list_centroid = [x[2] for x in list_cell_centroid]
            y_list_centroid = [x[1] for x in list_cell_centroid]
            z_list_centroid = [x[0] for x in list_cell_centroid]
            anndata.obs["centroid_z"] = z_list_centroid
            anndata.obs["centroid_y"] = y_list_centroid
            anndata.obs["centroid_x"] = x_list_centroid

        #anndata.obs["spots_coordinates"] = np.array(list_cell_genes_coordinate)
        #anndata.obs["genes"] = np.array(list_genes_name)

        ### create csv file
        column_name = list(self.G.nodes[list(self.G.nodes)[0]].keys())
        if "kernel_vector" in column_name:
            column_name.remove("kernel_vector")
        dict_dataframe = {}
        for column in column_name:
            dict_dataframe[column] = []
        for node_index, node_data in self.G.nodes(data=True):
            if node_data[self.gene_column] != 'centroid':
                for column in column_name:
                    dict_dataframe[column].append(node_data[column])
        df_spots = pd.DataFrame(dict_dataframe)

        ## rename colum index by original index node
        df_spots.rename(columns={"index": "node_index"}, inplace=True)

        anndata.uns["df_spots"] = df_spots
        self.final_anndata = anndata

        if not return_polygon:
            return anndata, {}
        assert min_rna_per_cell > 0, "min_rna_per_cell should be > 0"

        list_polygon = []
        dict_polygon = {}
        for cell_index in range(len(list_cell_id)):
            csv_list_y = []
            csv_list_x = []
            if len(list_genes_name[cell_index]) >= min_rna_per_cell:
                csv_list_y += list(np.array(list_cell_genes_coordinate[cell_index])[:, 1])
                csv_list_x += list(np.array(list_cell_genes_coordinate[cell_index])[:, 2])
                list_point = list(zip(csv_list_x, csv_list_y))
                alpha_shape = alphashape.alphashape(list_point, alpha)
                list_polygon.append(alpha_shape)
                dict_polygon[list_cell_id[cell_index]] = alpha_shape

        ### ensure that there is no overlap between the polygons of list_polygon
        for i in tqdm(list(range(len(list_polygon)))):
            for j in range(i + 1, len(list_polygon)):
                if list_polygon[i].intersects(list_polygon[j]):
                    list_polygon[i] = list_polygon[i].difference(list_polygon[j])
        list_keys = list(dict_polygon.keys())
        for i in range(len(list_keys)):
            dict_polygon[list_keys[i]] = list_polygon[i]
        self.list_polygon = list_polygon
        self.dict_polygon = dict_polygon

        ## Create the json
        json_dict = {"geometries": []}
        for cell in dict_polygon:
            geometry = dict_polygon[cell]
            if isinstance(geometry, shapely.geometry.Polygon):
                json_dict["geometries"].append({
                    "type": "Polygon",
                    "coordinates": [list(geometry.exterior.coords)],
                    "cell": cell
                })
            elif isinstance(geometry, shapely.geometry.MultiPolygon):
                for poly in list(geometry.geoms):
                    if allow_disconnected_polygon:
                        json_dict["geometries"].append({
                            "type": "Polygon",
                            "coordinates": [list(poly.exterior.coords)],
                            "cell": cell
                        })
                    else:
                        raise ValueError(
                            "disconnected polygons are not allowed, change allow_disconnected_polygon=True")
            else:
                continue
        return anndata, json_dict
