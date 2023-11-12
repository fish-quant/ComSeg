#!/bin/bash

#SBATCH -N 1

#nombre threads max sur GPU 48

#SBATCH -n 1



#SBATCH --mem 100000    # Memory per node in MB (0 allocates all the memory)

#SBATCH --ntasks=1              # Number of processes to run (default is 1)
#SBATCH --cpus-per-task=6      # CPU cores per process (default 1)


#SBATCH -p cbio-cpu


cd /cluster/CBIO/data1/data3/tdefard/simulation/mycode

python comseg_main_ipt.py \
--path_anndata /cluster/CBIO/data1/data3/tdefard/simulation/anndata/20220321_lung_merge_27samples_raw_selected_with_subtype.h5ad \
--path_cyto /cluster/CBIO/data1/data3/tdefard/T7/sp_data/In_situ_Sequencing_16/ \
--path_save /cluster/CBIO/data1/data3/tdefard/T7/sp_data/In_situ_Sequencing_16/comsegres/max10_15j/ \
--selected_gene_list iss_topo138_pcw13 \
--do_preprocessing 0 \
--real_data 1 \
--normalize_seq_centroid 1 \
--path_input_real_data /cluster/CBIO/data1/data3/tdefard/T7/sp_data/In_situ_Sequencing_16/input_comseg/dico_dico_commu_pw13_10_10__no_filt.npy \
--path_dico_coord_map /cluster/CBIO/data1/data3/tdefard/T7/sp_data/In_situ_Sequencing_16/dico_centroid/mask_tile_10_10/ \
--min_index_image $1 \
--max_index_image $2 \
--n_neighbors 10 \
--add_prior_knowledge_nucleus 0 \
--mean_cell_diameter 20 \
--max_cell_diameter 200 \
