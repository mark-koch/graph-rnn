import data
import extension_data

import torch
import random
import numpy as np
import networkx as nx

from tqdm import tqdm


N_ITERS = 9000

def bfs(g):

    a = nx.to_numpy_matrix(g)

    reordering = np.random.permutation(g.number_of_nodes())
    permuted_graph = nx.from_numpy_matrix(a[reordering][:, reordering])

    comps = [list(comp) for comp in nx.connected_components(permuted_graph)]

    traversal = []

    for comp in comps:
        successor_listing = [node[1] for node in nx.bfs_successors(permuted_graph, source=comp[0])]

        traversal.append(comp[0])

        for successor_list in successor_listing:
            for successor in successor_list:
                traversal.append(successor)

    return permuted_graph, traversal


def calculate_m_for_matrix(adj_matrix):
    """ Returns position of the nonzero value in adj_matrix which is furthest from the principle diagonal """

    return max(abs(x-y) for x, y in np.nonzero(adj_matrix))


def bfs_permute_directed(g):
    """Randomly permutes given DIRECTED graph, performs BFS, and returns
        the adjacency matrix ordered by BFS traversal.

    :param g NX Graph:          Graph for BFS traversal

    :return np matrix:          Permuted adjacency matrix
    """

    a = nx.to_numpy_matrix(g)
    n = g.number_of_nodes()

    reordering = np.random.permutation(g.number_of_nodes())
    permuted_graph = nx.from_numpy_matrix(a[reordering][:, reordering])

    a_reord = a[reordering][:, reordering]

    visited_nodes = set()
    unvisited_nodes = set(permuted_graph.nodes)

    traversal = []

    while len(visited_nodes) < n:
        src = random.choice(tuple(unvisited_nodes))
        successor_listing = [node[1] for node in nx.bfs_successors(permuted_graph, source=src)]

        traversal.append(src)
        visited_nodes.add(src)
        unvisited_nodes.remove(src)

        for successor_list in successor_listing:
            for successor in successor_list:
                if successor not in visited_nodes:
                    traversal.append(successor)
                    visited_nodes.add(successor)
                    unvisited_nodes.remove(successor)

    return a_reord[traversal][:, traversal]


def calculate_m_value_undirected(dset_name, dataset):
    gs = dataset.graphs
    max_M = -1

    for _ in tqdm(range(N_ITERS)):
        g = gs[np.random.randint(len(gs))]
        permuted_g, ordering = bfs(g)
        permuted_adj_mat = torch.tensor(np.tril(nx.adjacency_matrix(permuted_g, ordering).toarray()))
        max_M = max(max_M, calculate_m_for_matrix(permuted_adj_mat))
    print(f"M for {dset_name} :: {max_M}")


def calculate_m_value_directed(dset_name, dataset):
    gs = dataset.graphs
    max_M = -1
    for _ in tqdm(range(N_ITERS)):
        permuted_adj_mat = torch.tensor(bfs_permute_directed(gs[np.random.randint(len(gs))]))
        max_M = max(max_M, calculate_m_for_matrix(permuted_adj_mat))
    print(f"M for {dset_name} :: {max_M}")


if __name__ == "__main__":

    undirected_dset_names = ["grid-small",
                             "community-small",
                             "grid",
                             "community",
                             "ba",
                             "protein"]

    #for dset_name in undirected_dset_names:
    #    print(f"Processing: {dset_name}")
    #    dset = data.GraphDataSet(dset_name)
    #    calculate_m_value_undirected(dset_name, dset)


    dset = extension_data.DirectedGraphDataSet('ego-directed-multiclass')
    calculate_m_value_directed('ego-directed-multiclass', dset)
