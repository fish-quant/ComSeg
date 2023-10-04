
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


dico_cell_color = {}
for i in range(1, 1000):
        dico_cell_color[i] = "#" + "%06x" % random.randint(0, 0xFFFFFF)


def plot_result(nuclei,
                G,
                key_node = 'cell_index_pred',
                title = None,
                dico_cell_color = None,
                figsize=(15, 15),
                spots_size = 1,
                plot_outlier = True):


    mip_nuclei = np.amax(nuclei, 0)
    dico_spots_pos = {}  ## gene : [coord]
    unique_comm = np.unique([y[key_node] for x, y in G.nodes(data=True) if key_node in y])[:]
    for label in unique_comm:
        dico_spots_pos[label] = []
    for label in unique_comm:
        dico_spots_pos[label] = [[y["y"], y["x"]] for x, y in G.nodes(data=True) if
                                key_node in y and y[key_node] == label and y["gene"]
                                != "centroid"]

    if dico_cell_color is None:
        dico_cell_color = {}
        for label in unique_comm:
            dico_cell_color[label] = "#" + "%06x" % random.randint(0, 0xFFFFFF)

    fig, ax = plt.subplots(figsize=figsize)


    for cell in dico_spots_pos.keys():

        if not plot_outlier:
            if cell == 0:
                continue
        #print(cell, len(dico_spots_pos[cell]))
        if len(dico_spots_pos[cell]) == 0:
            continue
        ax.scatter(np.array(dico_spots_pos[cell])[:, 1], np.array(dico_spots_pos[cell])[:, 0],
                   c=dico_cell_color[cell],
                   # cmap=list_cmap1,,  #  gene_color_dico[gene],   #'#%02X%02X%02X' % (r(),r(),r()),
                   s=spots_size)

    if title is None:
        title = key_node
    plt.title(title)
    ax.imshow(mip_nuclei > 0, cmap='gist_gray', alpha=0.8)
    ax.imshow(mip_nuclei, cmap='gist_gray', alpha=0.8)
    return fig, ax