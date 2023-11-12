#!/bin/bash

#SBATCH -N 1

#nombre threads max sur GPU 48

#SBATCH -n 1


#SBATCH -J  main_non_conv


#SBATCH --output="main_non_conv.out"
#SBATCH --mem 45000    # Memory per node in MB (0 allocates all the memory)

#SBATCH --ntasks=1              # Number of processes to run (default is 1)
#SBATCH --cpus-per-task=4      # CPU cores per process (default 1)


#SBATCH -p cbio-cpu

echo rrrrrrr
cd /cluster/CBIO/data1/data3/tdefard/simulation/ComSeg_pkg/src/

python main_cube_remove.py \
--mean_cell_diameter $1 \
--max_cell_diameter $2 \
--path_dataset_folder /cluster/CBIO/data1/data3/tdefard/T7/regular_grid/simu1912/cube2D_step100/dataframe_folder/ns0_talassa_max3/ \
--path_to_mask_prior /cluster/CBIO/data1/data3/tdefard/T7/regular_grid/simu1912/cube2D_step100/nuclei/ \
--path_dict_cell_centroid /cluster/CBIO/data1/data3/tdefard/T7/regular_grid/simu1912/cube2D_step100/dico_centroid/  \
--path_simulation_gr /cluster/CBIO/data1/data3/tdefard/T7/regular_grid/simu1912/cube2D_step100/dico_simulation_nsforest0_thalassa/ \
--path_centroid /cluster/CBIO/data1/data3/tdefard/simulation/mycode/centroid/ \
