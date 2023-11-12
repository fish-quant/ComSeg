#!/bin/bash

#SBATCH -N 1

#nombre threads max sur GPU 48

#SBATCH -n 1


#SBATCH -J  main_non_conv


#SBATCH --output="main_non_conv.out"
#SBATCH --mem 65000    # Memory per node in MB (0 allocates all the memory)

#SBATCH --ntasks=1              # Number of processes to run (default is 1)
#SBATCH --cpus-per-task=12      # CPU cores per process (default 1)


#SBATCH -p cbio-cpu

echo rrrrrrr
cd /cluster/CBIO/data1/data3/tdefard/simulation/ComSeg_pkg/src/

python main_non_conv.py \
--mean_cell_diameter $1 \
--max_cell_diameter $2 \
--path_dataset_folder /cluster/CBIO/data1/data3/tdefard/T7/simulation/exp_same_cyto/same_param1_4_0/dataframe_cluster_filter_max3 \
--path_to_mask_prior /cluster/CBIO/data1/data3/tdefard/T7/simulation/exp_same_cyto/same_param1_4_0/remove20/nuclei/ \
--path_dict_cell_centroid /cluster/CBIO/data1/data3/tdefard/T7/simulation/exp_same_cyto/same_param1_4_0/remove20/dico_centroid/  \
--path_simulation_gr /cluster/CBIO/data1/data3/tdefard/T7/simulation/exp_same_cyto/same_param1_4_0/dico_simulation_nsforest0_thalassa/ \
