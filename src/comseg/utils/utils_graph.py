


import numpy as np




def _gen_graph(graph,
               super_node_prior_key='in_nucleus',
               distance='distance',
               key_pred="leiden2",
               ):
    """

    Parameters
    ----------
    graph
    prior_key :  the key of node taken as started point in dikstra
    distance
    key_pred

    Returns
    -------

    """
    ## change the variable in_nucleus to prior in_nucleus_list  in_nucleus
    """Generate a new graph based on the partitions of a given graph"""

    ########### creation of partition


    partition = []
    list_nodes = np.array([index for index, data in graph.nodes(data=True)])
    array_super_node_prior = np.array([data[super_node_prior_key] for index, data in graph.nodes(data=True)])
    unique_super_node_prior = np.unique(array_super_node_prior)
    if 0 in unique_super_node_prior:
        assert unique_super_node_prior[0] == 0
        unique_super_node_prior = unique_super_node_prior[1:]
        partition += [{u} for u in list_nodes[array_super_node_prior == 0]]
    for super_node in unique_super_node_prior:
        partition += [set(list_nodes[array_super_node_prior == super_node])]



    ### generation of new grpah with merged partition
    ### get all centroid  adn add them to in nucleus node



    H = graph.__class__().to_undirected(reciprocal=False)
    node2com = {}
    com2node = {}
    for i, part in enumerate(partition):
        nodes = set()
        com2node[i] = []
        if "gene" in graph.nodes[list(part)[0]] and graph.nodes[list(part)[0]]['gene'] == "centroid" and len(part) > 1:
            key_pred_label = graph.nodes[list(part)[1]][key_pred]
            prior_keys_labeled = graph.nodes[list(part)[1]][super_node_prior_key]
        else:
            key_pred_label = graph.nodes[list(part)[0]][key_pred]
            prior_keys_labeled = graph.nodes[list(part)[0]][super_node_prior_key]
        for node in part:
            node2com[node] = i
            com2node[i].append(node)
            nodes.update(graph.nodes[node].get("nodes", {node}))
            assert key_pred_label == graph.nodes[node][key_pred] or graph.nodes[node]['gene'] == 'centroid'
            assert prior_keys_labeled == graph.nodes[node][super_node_prior_key]

        H.add_node(i, nodes=nodes,
                   key_pred=key_pred_label,
                   super_node_prior_key=prior_keys_labeled)

    for node1, node2, wt in graph.edges(data=True):
        if distance not in wt:
            if "gene" in graph.nodes[node1] or  "gene" in graph.nodes[node2]:
                if "gene" in graph.nodes[node1]:
                    assert graph.nodes[node1]['gene'] == 'centroid'
                if "gene" in graph.nodes[node2]:
                    assert graph.nodes[node2]['gene'] == 'centroid'
            continue
        wt = wt[distance]
        com1 = node2com[node1]
        com2 = node2com[node2]
        if com1 == com2:
            continue
        if com2 in H[com1]:
            assert com1 in H[com2]
            temp = H.get_edge_data(com1, com2)[distance]
            if temp > graph[node1][node2][distance]:  ## take smallest distance between supernode
                H.add_edge(com1, com2, **{distance: wt})
        else:
            H.add_edge(com1, com2, **{distance: wt})
    return H

