


#%%
"""

Function for detecting communities based on Louvain Community Detection
Algorithm and using prior information to guide the partitioning.

code inspired from https://networkx.org/documentation/stable/_modules/networkx/algorithms/community/louvain.html#louvain_communities

"""


from collections import defaultdict, deque
import networkx as nx
from networkx.algorithms.community import modularity
from networkx.utils import py_random_state
import random
import statistics
import numpy as np



__all__ = ["louvain_communities", "louvain_partitions"]




@py_random_state("seed")
def louvain_communities(
    G,
    weight="weight",
    resolution=1,
    threshold=0.0000001,
    seed=random.Random(),
    partition = None,
    prior_key="in_nucleus",
    confidence_level = 0.99
    ):


    d = louvain_partitions(
        G=G,
        weight = weight,
        resolution = resolution,
        threshold = threshold,
        seed = seed,
        partition = partition,
        prior_key = prior_key,
        confidence_level = confidence_level,
                           )
    q = deque(d, maxlen=1)
    return q.pop()



@py_random_state("seed")
def louvain_partitions(
    G,
    weight="weight",
    resolution=1,
    threshold=0.0000001,
    seed=random.Random(),
    partition =None,
    prior_key = "in_nucleus",
    confidence_level = 0.99
    ):


    is_directed = G.is_directed()
    if partition is None:
        partition = [{u} for u in G.nodes()]
    if G.is_multigraph():
        graph = _convert_multigraph(G, weight, is_directed)
    else:
        graph = G.__class__()
        graph.add_nodes_from([(u, {prior_key: v[prior_key]}) for u, v in G.nodes(data=True)])
        graph.add_weighted_edges_from(G.edges(data=weight, default=1))
        graph = _gen_graph(graph, partition = partition, prior_key = prior_key)



    mod = modularity(G, partition, resolution=resolution, weight=weight)
    m = graph.size(weight="weight")
    partition, inner_partition, improvement = _one_level(
        graph = graph,
        m=m,
        partition=partition,
        resolution = resolution,
        is_directed = is_directed,
        seed = seed,
        confidence_level = confidence_level,
        prior_key="prior_index", ## harcoded prior key name from _gen_graph
    )
    #print(improvement)
    #print(len(partition))
    improvement = True
    while improvement:
        new_mod = modularity(
            graph, inner_partition, resolution=resolution, weight="weight"
        )
        graph = _gen_graph(graph, partition = inner_partition)
        yield [s.copy() for s in partition], graph

        if new_mod - mod <= threshold:
            print(f'stop because of improvement of modularity {new_mod - mod}  below  the threshold  {threshold}')
            improvement = False
        else:
            print(f'improvement of modularity {new_mod - mod}')
            mod = new_mod
            partition, inner_partition, improvement = _one_level(
                graph=graph,
                m=m,
                partition=partition,
                resolution=resolution,
                is_directed=is_directed,
                seed=seed,
                confidence_level=confidence_level,
                prior_key="prior_index", ## harcoded prior key name from _gen_graph
            )
        #print(len(partition))


def compute_prior_factor(source, ind_node, confidence_level = 0.99):

    if confidence_level is None:
        return 1

    assert confidence_level > 0
    assert confidence_level < 1

    if source==0 or ind_node==0:
        return 1

    if source == ind_node:
        return 1/(1-confidence_level)
    else:
        return (1-confidence_level)


def compute_prior_factor_scaled(label_commu, label_node, max_weight2com, confidence_level = 0.99):

    if confidence_level is None:
        return 1

    assert confidence_level >= 0
    assert confidence_level <= 1

    if label_commu==0 or label_node==0:
        return 0

    if label_commu == label_node:
        return max_weight2com * confidence_level
    else:
        assert label_commu != label_node
        return -max_weight2com * confidence_level




def _one_level(graph,
               m,
               partition,
               resolution=1,
               is_directed=False,
               seed=random.Random(),
               prior_key = "prior_index",
               confidence_level = 0.99):

    """

    Calculate one level of the Louvain partitions tree
    Parameters
    ----------
    graph : NetworkX Graph/DiGraph
        The graph from which to detect communities
    m : number
        The size of the graph `graph`.
    partition : list of sets of nodes
        A valid partition of the graph `graph`
    resolution : positive number
        The resolution parameter for computing the modularity of a partition
    is_directed : bool
        True if `graph` is a directed graph.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    """

    node2com = {u: i for i, u in enumerate(graph.nodes())}
    inner_partition = [{u}  for u in graph.nodes()] #inner_partition index supernode whereas partition index
    if is_directed:
        in_degrees = dict(graph.in_degree(weight="weight"))
        out_degrees = dict(graph.out_degree(weight="weight"))
        Stot_in = [deg for deg in in_degrees.values()]
        Stot_out = [deg for deg in out_degrees.values()]
        # Calculate weights for both in and out neighbours
        nbrs = {}
        for u in graph:
            nbrs[u] = defaultdict(float)
            for _, n, wt in graph.out_edges(u, data="weight"):
                nbrs[u][n] += wt
            for n, _, wt in graph.in_edges(u, data="weight"):
                nbrs[u][n] += wt
    else:
        degrees = dict(graph.degree(weight="weight"))
        Stot = [deg for deg in degrees.values()]
        nbrs = {u: {v: data["weight"] for v, data in graph[u].items() if v != u} for u in graph}
        ## add dico {(super) node : prior label}
        prior_label_node = {u:  v[prior_key] for u, v in  graph.nodes(data=True)} # stay the same during the pass
        prior_label_node_commu = {u:  v[prior_key] for u, v in  graph.nodes(data=True)} # Is updated during iteration
    rand_nodes = list(graph.nodes)
    seed.shuffle(rand_nodes)
    nb_moves = 1
    improvement = False
    list_move = []
    while nb_moves > 0:
        nb_moves = 0
        for u in rand_nodes:

            best_mod = 0
            best_com = node2com[u]
            weights2com = _neighbor_weights(nbrs[u], node2com)
            if is_directed:
                in_degree = in_degrees[u]
                out_degree = out_degrees[u]
                Stot_in[best_com] -= in_degree
                Stot_out[best_com] -= out_degree
                remove_cost = (
                    -weights2com[best_com] / m
                    + resolution
                    * (out_degree * Stot_in[best_com] + in_degree * Stot_out[best_com])
                    / m**2
                )
            else:
                degree = degrees[u]
                Stot[best_com] -= degree ## we remove i from the community best comm so Stot[best_com] do not contain the edge of i
                remove_cost =(-weights2com[best_com] / m
                              ) + resolution * ( Stot[best_com] * degree ) / (2 * m**2)


            for nbr_com, wt in weights2com.items(): # iterarat on neiboring community (including the original best_comm
                max_weight2com = max(weights2com.values())
                # to choose the one that optimise the most the modularity

                gain = (
                    remove_cost
                    + (wt + compute_prior_factor_scaled(
                          label_commu = prior_label_node_commu[nbr_com],
                          label_node = prior_label_node[u],
                           max_weight2com = max_weight2com,
                          #  max_weight2com = wt,
                        confidence_level = confidence_level)
                       ) / m
                    - resolution * (Stot[nbr_com] * degree) / (2 * m**2)
                )
                if gain > best_mod:
                    best_mod = gain
                    best_com = nbr_com
                    ## update weight to community

            if is_directed:
                Stot_in[best_com] += in_degree
                Stot_out[best_com] += out_degree
            else:
                Stot[best_com] += degree
            if best_com != node2com[u]: ## where a (super) node mode to a community to an other
                com = graph.nodes[u].get("nodes", {u})
                partition[node2com[u]].difference_update(com) # remove node u from its partition
                inner_partition[node2com[u]].remove(u) # remove node u from its partition
                partition[best_com].update(com) # add node u to its partion
                inner_partition[best_com].add(u) # add node u to its partion
                improvement = True
                nb_moves += 1

                #### update node prior label for best comm
                #todo optise this part factorize
                for commu in [best_com, node2com[u]]:
                    prior_freq_list = []
                    for node in inner_partition[commu]:
                        if 'nodes' in graph.nodes[node]:
                            prior_freq_list += [graph.nodes[node]['prior_index']] * len(graph.nodes[node]['nodes'])
                        else:
                            prior_freq_list += [graph.nodes[node]['prior_index']]

                    if len(set(prior_freq_list) - set([0])) == 0:
                        in_nucleus = 0
                    else:
                        prior_freq_array = np.array(prior_freq_list)  # remove zero, prior stay in the community even if there is only one node label
                        in_nucleus = statistics.mode(prior_freq_array[np.nonzero(prior_freq_array)[0]])
                    prior_label_node_commu[commu] = in_nucleus
                ####
                node2com[u] = best_com
        list_move.append(nb_moves)
    partition = list(filter(len, partition))
    inner_partition = list(filter(len, inner_partition))
    return partition, inner_partition, improvement


def _neighbor_weights(nbrs, node2com):
    """Calculate weights between nodes and its neighbor communities.

    Parameters
    ----------
    nbrs : dictionary
           Dictionary with nodes' neighbours as keys and their edge weight as value.
    node2com : dictionary
           Dictionary with all graph's nodes as keys and their community index as value.

    """
    weights = defaultdict(float)
    for nbr, wt in nbrs.items():
        weights[node2com[nbr]] += wt
    return weights


def _gen_graph(graph,
               partition,
               prior_key = 'in_nucleus'):
    ## change the variable in_nucleus to prior in_nucleus_list  in_nucleus
    """Generate a new graph based on the partitions of a given graph"""
    H = graph.__class__()
    node2com = {}
    for i, part in enumerate(partition):
        nodes = set()
        prior_list = []
        for node in part:
            node2com[node] = i
            nodes.update(graph.nodes[node].get("nodes", {node}))
            if "prior_list" in graph.nodes[node]:
                prior_list += graph.nodes[node]["prior_list"] # in nucleus list should be call prior_list
            else:
                prior_list.append(graph.nodes[node][prior_key])
        if len(set(prior_list) - set([0])) == 0:
            prior_index = 0
        else:
            prior_index_array = np.array(prior_list) # remove zero, prior stay in the community even if there is only one node label
            prior_index = statistics.mode(prior_index_array[np.nonzero(prior_index_array)[0]])

        H.add_node(i, nodes=nodes,
                   prior_list = prior_list,
                   prior_index = prior_index)

    for node1, node2, wt in graph.edges(data=True):
        wt = wt["weight"]
        com1 = node2com[node1]
        com2 = node2com[node2]
        temp = H.get_edge_data(com1, com2, {"weight": 0})["weight"]
        H.add_edge(com1, com2, **{"weight": wt + temp})
    return H


def _convert_multigraph(G, weight, is_directed):
    """Convert a Multigraph to normal Graph"""
    if is_directed:
        H = nx.DiGraph()
    else:
        H = nx.Graph()
    H.add_nodes_from(G)
    for u, v, wt in G.edges(data=weight, default=1):
        if H.has_edge(u, v):
            H[u][v]["weight"] += wt
        else:
            H.add_edge(u, v, weight=wt)
    return H




