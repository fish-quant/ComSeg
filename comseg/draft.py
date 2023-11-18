



import seaborn as sns
from matplotlib import pyplot as plt
corr_matrix = []

for gene0 in  dataset.dict_co_expression:
    list_corr_gene0 = []
    for gene1 in dataset.dict_co_expression:
        list_corr_gene0.append(dataset.dict_co_expression[gene0][gene1])
    corr_matrix.append(list_corr_gene0)


import random




list_gene = ['g', 'y', 'p', 'c', 'b', 'dg', 'db']


corr_matrix = [[1, 0.4, random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1,random.randrange(2000) * 0.0001 -0.1,random.randrange(2000) * 0.0001 -0.1,0.1, ],
                [0.4, 1, random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1,random.randrange(2000) * 0.0001 -0.1,random.randrange(2000) * 0.0001 -0.1, 0.1],
                   [random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1, 1, 0.6, random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1],
                     [random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1, 0.6, 1, random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1],
                    [random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1, 1, 0.8, random.randrange(2000) * 0.0001 -0.1],
               [random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1, 0.8, 1, random.randrange(2000) * 0.0001 -0.1],
                [0.1, 0.1, random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1, random.randrange(2000) * 0.0001 -0.1, 1]
               ]

#plotting the heatmap for correlation
ax = sns.heatmap(corr_matrix, xticklabels = list_gene, yticklabels = list_gene,)

plt.savefig("/home/tom/Bureau/logiciel/inkskape.svg", format="svg", dpi=600)
plt.savefig("/home/tom/Bureau/logiciel/corr_matrix.png", format="png", dpi=600)
plt.show()

plt.show()


image_format = 'svg' # e.g .png, .svg, etc.
file_name = Path(folder_save) / 'sup_2case_acc.svg'
fig.savefig(file_name, format=image_format, dpi=600)





import numpy as np
import pandas as pd


dico_commu  = np.load("/media/tom/T7/lustra_all/2023-09-06_LUSTRA-14rounds/image_per_pos/comseg_imput/14juin_dico_dico_commu_mask_removal_23j_6round.npy",
                        allow_pickle=True).item()



df = dico_commu['stich0']['df_spots_label']
df.to_csv("/media/tom/T7/lustra_all/2023-09-06_LUSTRA-14rounds/image_per_pos/dataframe_ComSeg/stich0.csv")


"/media/tom/T7/lustra_all/2023-09-06_LUSTRA-14rounds/image_per_pos/dataframe_ComSeg"



import pandas as pd
import numpy as np


dico_commu = np.load("/media/tom/T7/sp_data/In_situ_Sequencing_16/input_comseg/dico_dico_commu_pw13_10_10__no_filt.npy" , allow_pickle=True).item()


for img in dico_commu:
    df = dico_commu[img]['df_spots_label']
    df.to_csv("/media/tom/T7/sp_data/In_situ_Sequencing_16/dataframe_ComSeg/" + img + ".csv")




import pandas as pd
import numpy as np
from pathlib import Path

path_dico_centroid = "/media/tom/T7/regular_grid/simu1912/cube2D_step100/remove20/dico_centroid"


for path_dict in Path( path_dico_centroid).glob('*.npy'):
    dico_centroid = np.load(path_dict, allow_pickle=True).item()
    new_dico_centroid = {}

    for nuc in dico_centroid:
        new_dico_centroid[nuc] = np.mean(dico_centroid[nuc], axis = 0)

    np.save(path_dict, new_dico_centroid)


## load model

from src.comseg.dictionary import ComSegDict
import sys
from pathlib import Path
import numpy as np
model = ComSegDict()
from src import comseg
sys.modules['comseg'] = comseg
path_parameter = '/media/tom/T7/simulation/exp_same_cyto/same_param1_4_0/remove20/result/Comsegdict.10_d29_h11_min31_s25_r2899pickle.txt'
model.load(path_parameter)
import pickle

path_folder_graph =  "/media/tom/T7/simulation/exp_same_cyto/same_param1_4_0/remove20/result//10_d29_h11_min31_s25_r2899/"
Path(path_folder_graph).mkdir(parents = True, exist_ok=True)
for k in model:
    print(k)
    file_name = path_folder_graph + k
    with open(file_name, "wb") as file_:

        pickle.dump(model[k].G, file_, -1)




for k in model:
    key_cell = 'in_nucleus'
    dico_spots_pos = {}  ## gene : [coord]
    G = model[k].G
    unique_comm = np.unique([y[key_cell] for x, y in G.nodes(data=True) if key_cell in y])[:-1]
    print(len(unique_comm))
    nn