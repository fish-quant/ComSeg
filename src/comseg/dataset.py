import numpy as np

import scipy
import scipy.sparse as sp
import tifffile

import pandas as pd
from scipy.stats import hypergeom
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
#from processing_seg import full_df
from tqdm import tqdm
### dataset class
from .utils.preprocessing import compute_dict_centroid


class ComSegDataset():
    """
    this class is in charge of :

    1) loading the CSV input
    2) computation of the co-expression matrix at the dataset scale
    3) add prior knowledge if available

    The dataset class can be used like a dictionary of where the keys are the csv file names and the values are the csv

    """

    def __init__(self,
                 path_dataset_folder,
                 path_to_mask_prior=None,
                 mask_file_extension=".tiff",
                 dict_scale={"x": 0.103, 'y': 0.103, "z": 0.3},
                 mean_cell_diameter=15,
                 gene_column="gene",
                 image_csv_files: list = None,
                 centroid_csv_files: list = None,
                 path_cell_centroid=None,
                 centroid_csv_key={"x": "x", "y": "y", "z": "z", "cell_index": "cell"}
                 ):

        """
        :param path_dataset_folder: path to the folder containing the csv files
        :type path_dataset_folder: str
        :param path_to_mask_prior: path to the folder containing the mask priors. They must have the same name as the corresponding csv files
        :type path_to_mask_prior:  str
        :param mask_file_extension: file extension of the mask priors
        :default mask_file_extension: ".tiff"
        :param dict_scale: dictionary containing the pixel/voxel size of the images in µm default is {"x": 0.103, 'y': 0.103, "z": 0.3}
        :type dict_scale: dict
        :param mean_cell_diameter: the expected mean cell diameter in µm default is 15µm
        :type mean_cell_diameter: float
        :param gene_column: name of the column containing the gene name in the csv files
        :type gene_column: str
        :param image_names_csv_file: list of image csv file name to consider in the dataset if None consider all the csv files in the folder
        :type image_names_csv_file: list
        :param centroid_name: list of the centroid csv file name , as to be in the same order as the image_names_csv_file
        :type centroid_name: list
        """

        self.path_dataset_folder = Path(path_dataset_folder)
        if path_to_mask_prior is not None:
            self.path_to_mask_prior = Path(path_to_mask_prior)
        else:
            self.path_to_mask_prior = None
        self.mask_file_extension = mask_file_extension
        self.dict_scale = dict_scale
        self.mean_cell_diameter = mean_cell_diameter
        self.gene_column = gene_column
        self.path_image_dict = {}
        unique_gene = []
        self.image_csv_files = image_csv_files
        if image_csv_files is not None:
            for image_name in image_csv_files:
                image_path_df = Path(path_dataset_folder) / (image_name)
                ## check if the csv file exists
                assert image_path_df.exists(), f"{image_name} not found in the dataset folder {path_dataset_folder}"
                self.path_image_dict[image_path_df.stem] = self.path_dataset_folder / (image_name)
                unique_gene += list(pd.read_csv(self.path_image_dict[image_path_df.stem])[self.gene_column].unique())
        else:
            for image_path_df in self.path_dataset_folder.glob(f'*.csv'):
                print(f"add {image_path_df.stem}")
                self.path_image_dict[image_path_df.stem] = image_path_df
                unique_gene += list(pd.read_csv(self.path_image_dict[image_path_df.stem])[self.gene_column].unique())
        if len(self.path_image_dict) == 0:
            raise ValueError(f"no csv file found in the dataset folder {self.path_dataset_folder}")
        self.list_index = list(self.path_image_dict.keys())
        self.selected_genes = np.unique(unique_gene)
        self.centroid_csv_files = centroid_csv_files
        if self.centroid_csv_files is not None:
            assert len(self.centroid_csv_files) == len(
                self.list_index), "centroid_file_name should have the same length as image_names_csv_file"
            self.dict_centroid = {}
            for i in range(len(self.centroid_csv_files)):
                self.dict_centroid[self.list_index[i]] = self.centroid_csv_files[i]

                df_centroid = pd.read_csv(Path(path_cell_centroid) / (self.centroid_csv_files[i]))
                x_list = list(df_centroid[centroid_csv_key["x"]])
                y_list = list(df_centroid["y"])
                if centroid_csv_key["z"] in df_centroid.columns:
                    z_list = list(df_centroid["z"])
                cell_list = list(df_centroid[centroid_csv_key["cell_index"]])
                if centroid_csv_key["z"] in df_centroid.columns:
                    self.dict_centroid[self.list_index[i]] = {cell_list[i]: np.array([z_list[i], y_list[i], x_list[i]])
                                                              for i in range(len(cell_list))}
                else:
                    self.dict_centroid[self.list_index[i]] = {cell_list[i]: {"x": x_list[i], "y": y_list[i]} for i in
                                                              range(len(cell_list))}
                self.dict_centroid[self.list_index[i]]

    def __getitem__(self, key):
        if isinstance(key, int):
            return pd.read_csv(self.path_image_dict[self.list_index[key]])
        elif isinstance(key, str):
            return pd.read_csv(self.path_image_dict[key])
        return self.path_dataset_folder[key]

    def __len__(self):
        return len(list(self.path_dataset_folder.glob('*.csv')))

    def __iter__(self):
        for x in self.list_index:
            yield x

    def __repr__(self):
        return repr(f'dataset comseg  {self.list_index}')

    def keys(self):
        return self.list_index

    def convert_spots_coord_in_pixel(self, overwrite=True):
        """
        convert the coordinates of the spots in the csv files from µm to pixels using the dict_scale attribute of the dataset object

        :param overwrite: if True, overwrite the csv files with the new coordinates in pixels else save the new csv files with the suffix "_pixel"
        :type bool
        :return:
        """
        for image_path_df in self.path_dataset_folder.glob('*.csv'):
            print(f"to pixel coordinates {image_path_df.stem}")
            df_spots = pd.read_csv(image_path_df)

            df_spots["y"] = (df_spots["y"] / self.dict_scale["y"]).astype(int)
            df_spots["x"] = (df_spots["x"] / self.dict_scale["x"]).astype(int)
            if "z" in df_spots.columns:
                df_spots["z"] = (df_spots["z"] / self.dict_scale["z"]).astype(int)
            if overwrite:
                df_spots.to_csv(image_path_df, index=False)
            else:
                print(f"nex csv save as {image_path_df.parent / (image_path_df.stem + '_pixel.csv')}")
                df_spots.to_csv(image_path_df.parent / (image_path_df.stem + '_pixel.csv'), index=False)

    #### fct adding prior

    def add_prior_from_mask(self,
                            config=None,
                            prior_name='in_nucleus',
                            overwrite=False,
                            compute_centroid=True,
                            regex_df="*.csv"
                            ):

        """

        This function add prior knowledge to the dataset. It adds a column in the csv files indicating prior label of each spot.
        It takes the positition of each spot and add the corresponding value of the segmentation mask prior (.tiff) at this position.

        :param prior_name: name of the column to add in the csv files containing the prior label of each spot
        :type str
        :param overwrite: if True, overwrite the prior_name column if it already exists
        :type bool
        :param compute_centroid : if True, compute the centroid of each cell/nucleus in segmentation mask for to use it for RNA-cell association
        :type bool
        :return: None
        """

        if config is not None:
            print("config dict overwritting default parameters")
            prior_name = config.get("prior_name", prior_name)
            overwrite = config.get("overwrite", overwrite)
            compute_centroid = config.get("compute_centroid", compute_centroid)
            regex_df = config.get("regex_df", regex_df)

        if compute_centroid:
            self.dict_centroid = {}
        for image_path_df in self.path_dataset_folder.glob(regex_df):
            print(f"add prior to {image_path_df.stem}")
            df_spots = pd.read_csv(image_path_df)

            assert (self.path_to_mask_prior / (
                        image_path_df.stem + self.mask_file_extension)).exists(), f"no mask prior found for {image_path_df.stem}"

            if 'tif' in self.mask_file_extension[-4:]:
                mask = tifffile.imread(self.path_to_mask_prior / (image_path_df.stem + self.mask_file_extension))
            elif 'npy' in self.mask_file_extension[-4:]:
                mask = np.load(self.path_to_mask_prior / (image_path_df.stem + self.mask_file_extension))
            else:
                raise ValueError("mask file extension not supported please use image_name.npy or image_name.tif(f)")

            if "z" in df_spots.columns and len(mask.shape) == 3:
                pixel_coordinates = (df_spots[["z", "y", "x"]]).astype(int).values
                prior_list = [mask[z, y, x] for (z, y, x) in pixel_coordinates]

            else:
                pixel_coordinates = (df_spots[["y", "x"]]).astype(int).values
                prior_list = [mask[y, x] for (y, x) in pixel_coordinates]

            if prior_name in df_spots.columns and overwrite == False:
                raise Exception(f"prior_name {prior_name} already in df_spots and overwrite is False")
            df_spots[prior_name] = prior_list
            df_spots.to_csv(image_path_df, index=False)
            print(f"prior added to {image_path_df.stem} and saved in csv file")

            if compute_centroid:
                dict_centroid = compute_dict_centroid(mask_nuclei=mask,
                                                      background=0)

                self.dict_centroid[image_path_df.stem] = dict_centroid
                print(f"dict_centroid added for {image_path_df.stem} ")

    ### compute the co-expression matrix

    def count_matrix_in_situ_from_knn(self,
                                      df_spots_label,
                                      n_neighbors=5,
                                      radius=None,
                                      remove_self_node=False,
                                      sampling=True,
                                      sampling_size=10000
                                      ):

        """
        Compute the co-expression score matrix for the RNA spatial distribution

        :param df_spots_label:  dataframe with the columns x,y,z,gene. the coordinates are rescaled in µm by dict_scale attribute of the dataset object
        :type df_spots_label: pd.DataFrame
        :param n_neighbors: maximum number of neighbors default is 40
        :type n_neighbors: int
        :param radius: maximum radius of neighbors. It should be set proportionnaly to expected cell size, default is radius =  mean_cell_diameter / 2
        :param sampling : if True, sample the dataset to compute the correlation
        :type sampling: bool
        :param sampling_size: if sampling is True : number of proximity weighted expression vector to sample
        :type sampling_size: int
        :return: count_matrix of shape (N_rna,  n_genes) where n_genes is the number of unique genes in df_spots_label
        each row is an 'RNA expression vector' summarizing local expression neighborhood of a molecule
        :rtype: np.array
        """

        gene_index_dico = {}
        for gene_id in range(len(self.selected_genes)):
            gene_index_dico[self.selected_genes[gene_id]] = gene_id  #todo gene_index_dico to add in self
        ## this part should be factorize with create_directed_nn_graph
        try:
            df_spots_label = df_spots_label.reset_index()
        except Exception as e:
            print(e)

        if "z" in df_spots_label.columns:
            list_coordo_order_no_scaling = np.array([df_spots_label.z, df_spots_label.y, df_spots_label.x]).T
            list_coordo_order = list_coordo_order_no_scaling * np.array(
                [self.dict_scale['z'], self.dict_scale['y'], self.dict_scale["x"]])
        else:
            list_coordo_order_no_scaling = np.array([df_spots_label.y, df_spots_label.x]).T
            list_coordo_order = list_coordo_order_no_scaling * np.array([self.dict_scale['y'], self.dict_scale['x']])

        dico_list_features = {}
        assert self.gene_column in df_spots_label.columns
        for feature in df_spots_label.columns:
            dico_list_features[feature] = list(df_spots_label[feature])
        list_features = list(dico_list_features.keys())
        list_features_order = [(i, {feature: dico_list_features[feature][i] for feature in list_features}) for i in
                               range(len(df_spots_label))]
        array_gene_indexed = np.array([dico_list_features[self.gene_column][i] for i in range(len(df_spots_label))])

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(list_coordo_order)
        ad = nbrs.kneighbors_graph(list_coordo_order)  ## can be optimize here
        distance = nbrs.kneighbors_graph(list_coordo_order, mode='distance')
        ad[distance > radius] = 0
        ad.eliminate_zeros()

        rows, cols, BOL = sp.find(ad == 1)

        ## dratf optimisation
        unique_rows = np.unique(rows)
        if sampling:
            if len(unique_rows) > sampling_size:
                unique_rows = np.random.choice(unique_rows, sampling_size, replace=False)

        list_expression_vec = []
        for row in unique_rows:
            col_index = np.nonzero(ad[row])[1]
            if remove_self_node:
                col_index = col_index[col_index != row]
            vectors_gene = list(array_gene_indexed[col_index])
            vector_distance = np.array([distance[row, col] for col in col_index])
            expression_vector = np.zeros(len(self.selected_genes))
            for str_gene_index in range(len(vectors_gene)):
                str_gene = vectors_gene[str_gene_index]
                expression_vector[gene_index_dico[str_gene]] += (radius - 1 * vector_distance[str_gene_index]) / radius
            list_expression_vec.append(expression_vector)
        count_matrix = np.array(list_expression_vec)

        """edges = list(zip(rows.tolist(), cols.tolist()))

        G = nx.DiGraph()  # oriented graph
        G.add_nodes_from(list_features_order)
        weighted_edges = [(e[0], e[1], distance[e[0], e[1]]) for e in edges]
        G.add_weighted_edges_from(weighted_edges)

        list_expression_vec = []
        for node in list(G.nodes()):
            successors = set(list(G.successors(node)))
            if remove_self_node:
                successors.remove(node)
            vectors_gene = list(array_gene_indexed[list(successors)])
            vector_distance = np.array([G[node][suc]['weight'] for suc in G.successors(node)])
            # vector_distance
            expression_vector = np.zeros(len(self.selected_genes))
            for str_gene_index in range(len(vectors_gene)):
                str_gene = vectors_gene[str_gene_index]
                expression_vector[gene_index_dico[str_gene]] += (radius  - 1 *  vector_distance[str_gene_index]) / radius
            list_expression_vec.append(expression_vector)
        count_matrix = np.array(list_expression_vec)"""
        return count_matrix

    def get_dict_proba_edge_in_situ(self,
                                    count_matrix,
                                    distance="pearson",
                                    ):
        """
        compute the co-expression correlation matrix from the count_matrix

        :param count_matrix: cell by gene matrix
        :type count_matrix: np.array
        :param distance:  choose in ["pearson", "spearman"] default is pearson
        :type distance: str
        :return: a dictionary of dictionary corelation between genes dict[gene_source][gene_target] = correlation
        :rtype: dict
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
                dico_proba_edge[self.selected_genes[gene_source]][self.selected_genes[gene_target]] = corr
                dico_proba_edge[self.selected_genes[gene_target]][self.selected_genes[gene_source]] = corr
        return dico_proba_edge

    def compute_edge_weight(
            self,
            config=None,
            images_subset=None,
            n_neighbors=40,
            radius=None,  # in micormeter
            distance="pearson",
            sampling=True,
            sampling_size=10000,
            remove_self_node=False):

        #print("adapt to when I prune")

        """
        compute the gene co-expression correlation at the dataset scale

        :param config: dictionary of parameters to overwrite the default parameters, default is None
        :type config: dict
        :param images_subset: default None, if not None, only compute the correlation on the images in the list
        :type images_subset: list
        :param n_neighbors:  default 40 ,number of neighbors to consider in the knn graph
        :type n_neighbors: int
        :param radius: radius of the knn graph in micrometer default None, if None, radius = mean_cell_diameter/2
        :type radius: float
        :param distance: choose in ["pearson", "spearman"] default is pearson
        :type distance: str
        :param sampling: default False, if True, sample the dataset to compute the correlation
        :type sampling: bool
        :param sampling_size: if sampling is True : number of proximity weighted expression vector to sample
        :return:
           - dico_proba_edge - a dictionary of dictionary correlation between genes. dict[gene_source][gene_target] = correlation
           - count_matrix - the count matrix used to compute the correlation
        :rtype:  dict, np.array
        """

        if config is not None:
            #print("config dict overwritting default parameters")
            images_subset = config.get("images_subset", images_subset)
            n_neighbors = config.get("n_neighbors", n_neighbors)
            radius = config.get("radius", radius)
            distance = config.get("distance", distance)
            sampling = config.get("sampling", sampling)
            sampling_size = config.get("sampling_size", sampling_size)

        if radius is None:
            radius = self.mean_cell_diameter / 2

        dico_proba_edge = {}

        list_of_count_matrix = []
        assert self.__len__() > 0, "no images in the dataset"
        ## check that all the images in images_subset are in the dataset
        if images_subset is not None:
            for image_name in images_subset:
                assert image_name in list(self.path_image_dict.keys()), f"{image_name} not in the dataset"
            list_img = images_subset
        else:
            list_img = list(self.path_image_dict.keys())

        for image_name in tqdm(list_img):
            df_spots_label = pd.read_csv(self.path_image_dict[image_name])
            #print(df_spots_label)

            print("image name : ", image_name)
            count_matrix = self.count_matrix_in_situ_from_knn(df_spots_label=df_spots_label,  # df_spots_label,
                                                              n_neighbors=n_neighbors,
                                                              radius=radius,
                                                              remove_self_node=remove_self_node,
                                                              sampling=True,
                                                              sampling_size=int(sampling_size / len(list_img) + 1))
            list_of_count_matrix.append(count_matrix)
            count_matrix = np.concatenate(list_of_count_matrix, axis=0)
        if sampling:
            if len(count_matrix) > sampling_size:
                #print("count_matrix.shape", count_matrix.shape)
                #print(f"sampling {sampling} vectors")
                count_matrix = count_matrix[np.random.choice(count_matrix.shape[0], sampling_size, replace=False), :]
                #print("count_matrix.shape", count_matrix.shape)

        dict_co_expression = self.get_dict_proba_edge_in_situ(count_matrix=count_matrix,
                                                              distance=distance,
                                                              )
        self.dict_co_expression = dict_co_expression
        return dico_proba_edge, count_matrix

    ### fct adding co-expression matrix
