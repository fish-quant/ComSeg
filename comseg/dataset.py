

## todod
## clean count_matrix_in_situ_from_knn and recheck
## clean count_matrix_in_situ_from_knn
## how to handle the arg "selected_gene"


import os
import sys
import networkx as nx
import numpy as np

import scipy
import scipy.sparse as sp
import tifffile

import pandas as pd
from scipy.stats import hypergeom
from sklearn.neighbors import NearestNeighbors
sys.path.insert(1, os.getcwd() + "/code/")
from pathlib import Path
from sklearn.neighbors import radius_neighbors_graph
#from processing_seg import full_df


### dataset class


## take as input the path to the dataset folder of dataframes

## comput if possible the prior and save save it in the dataset folder path to mask

## compute the co-expression matrix

### compute the norm parameter


class ComSegDataset():

    def __init__(self,
                 selected_genes,
        path_dataset_folder = None,
        path_to_mask_prior = None,
        mask_file_extension = ".tiff",
                    ):
        self.path_dataset_folder = Path(path_dataset_folder)
        self.path_to_mask_prior = Path(path_to_mask_prior)
        self.mask_file_extension = mask_file_extension

        ## initilatise dicotnary image name : path to image

        self.path_image_dict = {}

        for image_path_df in self.path_dataset_folder.glob('*.csv'):
            print(f"add {image_path_df.stem}")
            self.path_image_dict[image_path_df.stem] = image_path_df
        self.list_index = list(self.path_image_dict.keys())

        self.selected_genes = selected_genes

    def __getitem__(self, key):
        if isinstance(key, int):
            return pd.read_csv(self.path_image_dict[self.list_index[key]])
        elif isinstance(key, str):
            return pd.read_csv(self.path_image_dict[key])
        return self.path_dataset_folder[key]

    def __len__(self):
        return len(list(self.path_dataset_folder.glob('*.csv')))

    def __iter__(self):
        for x in self.list_index :
            yield x


    #### fct adding prior

    def add_prior_from_mask(self):

        for image_path_df in self.path_dataset_folder.glob('*.csv'):
            print(f"add prior to {image_path_df.stem}")
            df_spots = pd.read_csv(image_path_df)

            assert (self.path_to_mask_prior / (image_path_df.stem + self.mask_file_extension)  ).exists(), f"no mask prior found for {image_path_df.stem}"

            if 'tif' in self.mask_file_extension :
                mask = tifffile.imread(self.path_to_mask_prior / (image_path_df.stem + self.mask_file_extension) )

            elif 'npy' in self.mask_file_extension :
                mask = np.load(self.path_to_mask_prior / (image_path_df.stem + self.mask_file_extension) )
            else:
                raise ValueError("mask file extension not supported")
            x_list = list(df_spots.x)
            y_list = list(df_spots.y)
            z_list = list(df_spots.z)
            prior_list = []
            for ix in range(len(z_list)):
                nuc_index_prior = mask[z_list[ix], y_list[ix], x_list[ix]]
                prior_list.append(nuc_index_prior)
            df_spots['prior'] = prior_list
            df_spots.to_csv(image_path_df,  index=False)



    ### compute the co-expression matrix

    def count_matrix_in_situ_from_knn(self,
                                      df_spots_label,
                                      dico_scale={"x": 0.103, 'y': 0.103, "z": 0.3},  # in micrometer
                                      n_neighbors=5,
                                      radius=5,
                                      mode="nearest_nn_radius",
                                      directed=True):
        """
        compute the in situ count matrix indexed by selected_genes using the directed graph
        take either as input dico_simulation (internal) or df_spots_label (more general)
        Args:
            dico_simulation ():
            df_spots_label (dataframe): with the columm 'x', 'y', 'z', 'gene',
            selected_genes ():
            scale ():
            scaling_factor ():
            n_neighbors ():
            radius ():
            mode ():
            directed ():
        Returns:
        """

        # todo add a double condition
        assert mode in ["nearest_nn", "radius", "nearest_nn_radius", "dist_weighted"]
        gene_index_dico = {}
        for gene_id in range(len(self.selected_genes)):
            gene_index_dico[self.selected_genes[gene_id]] = gene_id #todo gene_index_dico to add in self
        print(len(gene_index_dico))
        ## this part should be factorize with create_directed_nn_graph
        try:
            df_spots_label = df_spots_label.reset_index()
        except Exception as e:
            print(e)
        if "z" in df_spots_label.columns:
            list_coordo_order_no_scaling = np.array([df_spots_label.x, df_spots_label.y, df_spots_label.z]).T
            list_coordo_order = list_coordo_order_no_scaling * np.array(
                [dico_scale['x'], dico_scale['y'], dico_scale["z"]])

        else:
            list_coordo_order_no_scaling = np.array([df_spots_label.x, df_spots_label.y]).T
            list_coordo_order = list_coordo_order_no_scaling * np.array([dico_scale['x'], dico_scale['y']])

        dico_list_features = {}
        assert 'gene' in df_spots_label.columns
        for feature in df_spots_label.columns:
            dico_list_features[feature] = list(df_spots_label[feature])
        list_features = list(dico_list_features.keys())
        list_features_order = [(i, {feature: dico_list_features[feature][i] for feature in list_features}) for i in
                               range(len(df_spots_label))]
        array_gene_indexed = np.array([dico_list_features['gene'][i] for i in range(len(df_spots_label))])

        if mode == "nearest_nn":
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(list_coordo_order)
            ad = nbrs.kneighbors_graph(list_coordo_order)  ## can be optimize here

        elif mode == "radius":
            ad = radius_neighbors_graph(list_coordo_order, radius, mode='connectivity', include_self=True)
        elif mode == "nearest_nn_radius" or mode == "dist_weighted":
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(list_coordo_order)
            ad = nbrs.kneighbors_graph(list_coordo_order)  ## can be optimize here
            distance = nbrs.kneighbors_graph(list_coordo_order, mode='distance')
            ad[distance > radius] = 0
            ad.eliminate_zeros()
        else:
            raise Exception(f" {mode} Not implemented")
        rows, cols, BOL = sp.find(ad == 1)
        edges = list(zip(rows.tolist(), cols.tolist()))

        if directed:
            G = nx.DiGraph()  # oriented graph
            G.add_nodes_from(list_features_order)
            if mode == "dist_weighted":
                weighted_edges = [(e[0], e[1], distance[e[0], e[1]]) for e in edges]
                G.add_weighted_edges_from(weighted_edges)
            else:
                G.add_edges_from(edges)

            list_expression_vec = []
            for node in list(G.nodes()):
                vectors_gene = list(array_gene_indexed[list(G.successors(node))])
                if mode == "dist_weighted":
                    vector_distance = np.array([G[node][suc]['weight'] for suc in G.successors(node)])
                # vector_distance
                expression_vector = np.zeros(len(self.selected_genes))
                for str_gene_index in range(len(vectors_gene)):
                    str_gene = vectors_gene[str_gene_index]
                    if mode == "dist_weighted":
                        expression_vector[gene_index_dico[str_gene]] += 1 * vector_distance[str_gene_index]
                    else:
                        expression_vector[gene_index_dico[str_gene]] += 1
                # print(np.sum(expression_vector))
                list_expression_vec.append(expression_vector)
        else:
            raise Exception('not implemented')
            # G = nx.Graph()
        count_matrix = np.array(list_expression_vec)
        return count_matrix

    def get_dico_proba_edge_in_situ(self, count_matrix,
                                    distance="pearson",
                                    negative_k=1):
        """
        compute
        dico_proba_edge { gene1: {gene2 : pearson correlation }
        Parameters
        ----------
        count_matrix :
        selected_genes :
        Returns
        -------
        dico_proba_edge { gene1: {gene2 : correlation }
        """
        import math
        assert distance in ["spearman", "pearson"]
        from tqdm import tqdm
        dico_proba_edge = {}
        for gene_source in range(len(self.selected_genes)):  # I compute the same thing twice ...
            dico_proba_edge[self.selected_genes[gene_source]] = {}

        for gene_source in tqdm(range(len(self.selected_genes))):  # I compute the same thing twice ...
            #print(gene_source)
            for gene_target in range(gene_source, len(self.selected_genes)):
                exp_gene_source = count_matrix[:, gene_source]
                exp_gene_target = count_matrix[:, gene_target]
                if distance == "pearson":
                    corr = scipy.stats.pearsonr(exp_gene_source, exp_gene_target)[0]
                elif distance == "spearman":
                    corr = scipy.stats.spearmanr(exp_gene_source, exp_gene_target)[0]

                else:
                    raise Exception(f'distance {distance} not implemented')
                if math.isnan(corr):
                    corr = -1
                if corr < 0:
                    corr = corr * negative_k

                dico_proba_edge[self.selected_genes[gene_source]][self.selected_genes[gene_target]] = corr
                dico_proba_edge[self.selected_genes[gene_target]][self.selected_genes[gene_source]] = corr
        return dico_proba_edge

    def compute_in_situ_edge(
            self,
            dico_df_spots_label=None,

            dico_scale={"x": 0.103, 'y': 0.103, "z": 0.3},  # in micrometer
            images_subset = None,
            mode="nearest_nn_radius",
            n_neighbors=25,
            radius=1,  # in micormeter
            distance="pearson",
            per_images=False,
            sampling=False,
            sampling_size=100000):

        #print("adapt to when I prune")

        """
        compute the count matrix of each images with count_matrix_in_situ_from_knn()
         then use them all to compute the edge weigth dico with get_dico_proba_edge_in_situ()
           dico_proba_edge { gene1: {gene2 : correlation }
        Parameters
        ----------
        path_simu_only :
        selected_genes :
        max_number_image :
        mode :
        n_neighbors :
        radius :
        scale :
        Returns
        -------
        dico_proba_edge { gene1: {gene2 : correlation }
        """

        dico_proba_edge = {}
        assert mode in ["nearest_nn", "radius", "nearest_nn_radius", 'dist_weighted']
        assert dico_df_spots_label is None

        list_of_count_matrix = []
        for image_name in self.path_image_dict.keys():

            if images_subset is not None: ## select only intersting images if needed
                if image_name not in images_subset:
                    continue
            df_spots_label = pd.read_csv(self.path_image_dict[image_name])
            #print(df_spots_label)


            print("image name : ", image_name)
            count_matrix = self.count_matrix_in_situ_from_knn(df_spots_label=df_spots_label,  # df_spots_label,
                                                         dico_scale=dico_scale,  # in micrometer
                                                         n_neighbors=n_neighbors,
                                                         radius=radius,
                                                         mode=mode,
                                                         directed=True,)
            list_of_count_matrix.append(count_matrix)
        if sampling:
            print("count_matrix.shape", count_matrix.shape)

            print(f"sampling {sampling} vectors")
            count_matrix = np.concatenate(list_of_count_matrix, axis=0)
            count_matrix = count_matrix[np.random.choice(count_matrix.shape[0], sampling_size, replace=False), :]
            print("count_matrix.shape", count_matrix.shape)

        if not per_images:
            dico_proba_edge = self.get_dico_proba_edge_in_situ(count_matrix=count_matrix,
                                                          distance=distance,
                                                          )

        self.dict_co_expression = dico_proba_edge

        return dico_proba_edge, count_matrix

    ### fct adding co-expression matrix


if __name__ == "__main__":
    list_marker_ns =  ['Atp6v0d2', 'Abcg1',# AM
             'Rtkn2',  'Igfbp2', #AT1
             'Sftpc','Cxcl15', #AT2,
            'Cd79a', #B_cells
             'Ms4a2', 'Fcer1a', #Basophils
             'Ccdc153', #Ciliated
             'Scgb3a2', 'Scgb1a1',#Club
             'Cst3',#DC
             'Cdh5', 'Clec14a',  #EC
             'Inmt', 'Pcolce2', # Fibroblasts
             'C1qc', 'C1qa', 'C1qb', # 'C3ar1', #IM
             'Upk3b',# Mesotheliocytes
             'Ifitm6','Plac8',# Monocytes
            'Ms4a4b', 'Ccl5', 'Hcst', # NK_T_cells
             'Gzma', 'Ncr1',# NK_cells
             'S100a9',# Neutrophils
             'Mmrn1',#Platelets
           'Acta2','Myh11', # SMC
             'Cd3g', 'Cd3d' #T_cells
             ]

    dataset = ComSegDataset(selected_genes = list_marker_ns,
        path_dataset_folder = "/media/tom/T7/simulation/test_set/dataframes",
                    path_to_mask_prior = "/media/tom/T7/simulation/test_set/mask",
                    )
    dataset.add_prior_from_mask()

    dataset.compute_in_situ_edge(dico_scale={"x": 0.103, 'y': 0.103, "z": 0.3},  # in micrometer
            selected_genes=list_marker_ns,
            images_subset = None,
            mode="nearest_nn_radius",
            n_neighbors=25,
            radius=1,  # in micormeter
            distance="pearson",
            per_images=False,
            sampling=False,
            sampling_size=100000)
