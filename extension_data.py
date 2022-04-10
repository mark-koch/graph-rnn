import numpy as np
import networkx as nx
import torch
import random


def topological_sort(g):
    """Returns the adjacency matrix of G, reordered according to a random topological sort."""

    a = nx.to_numpy_matrix(g)
    n = g.number_of_nodes()

    reordering = np.random.permutation(g.number_of_nodes())
    permuted_matrix = a[reordering][:, reordering]
    permuted_graph = nx.from_numpy_matrix(permuted_matrix, create_using=nx.DiGraph)

    topsort = list(nx.lexicographical_topological_sort(permuted_graph))

    return permuted_matrix[topsort][:, topsort]


def bfs_permute(g):
    """Randomly permutes given DIRECTED graph, performs BFS, and returns
        the adjacency matrix reordered by a randomized BFS traversal.
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

class DirectedGraphDataSet(torch.utils.data.Dataset):
    """Dataset to handle directed DAGs and ego-networks in various representations"""


    def __init__(self, dataset, m=None, training=True, train_split=0.8, bfs=True):

        self.dataset_type = dataset
        self.max_node_count = -1
        self.m = m

        np.random.seed(42)

        if dataset == 'dag-multiclass':
            self.graphs =  self.load_DAG_dataset()
        elif 'ego' in dataset:
            # 'ego-directed-multiclass'
            # 'ego-directed-topsort'
            self.graphs = self.load_citeseer_ego_dags()
        else:
            raise Exception(f"No data-loader for dataset `{dataset}`")

        # Shuffle for random train/test slit
        np.random.shuffle(self.graphs)

        train_size = int(len(self.graphs) * train_split)
        self.start_idx = 0 if training else train_size
        self.length = train_size if training else len(self.graphs) - train_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if self.dataset_type == 'ego-directed-topsort':
            g = self.graphs[self.start_idx + idx]
            n = g.number_of_nodes()

            permuted_matrix = topological_sort(g).T[1:]
            # Cannot use M-trick, return padded matrix

            return {'x': np.pad(permuted_matrix, [(0, self.max_node_count - n), (0, self.max_node_count - n)]), 'len': n-1}

        if 'multiclass' in self.dataset_type:
            # Need to convert the NX DAGs into upper right triangular matrices, using BFS

            g = self.graphs[self.start_idx + idx]
            permuted_matrix = bfs_permute(g)

            augmented_matrix = permuted_matrix + 2 * permuted_matrix.T
            augmented_matrix = augmented_matrix.astype(int)
            # 0: No edge
            # 1: edge ->
            # 2: edge <-
            # 3: edge <>

            nclasses = 4
            tnsr = np.zeros((augmented_matrix.shape[0], augmented_matrix.shape[1], nclasses))

            for i in range(augmented_matrix.shape[0]):
                for j in range(i):
                    tnsr[i, j, augmented_matrix[i,j]] = 1

            scratch = []

            # Use only M-width data off the diagonal
            for i in range(1, tnsr.shape[0]):
                # Data that actually can have 1s:
                critical_strip = tnsr[i, max(i-self.m, 0):i]
                m_dash = critical_strip.shape[0]
                scratch.append(np.pad(critical_strip, [(self.m - m_dash, 0), (0, 0)])[::-1])

            result = np.array(scratch)

            return {'x': np.pad(result, [(0, self.max_node_count - result.shape[0]), (0,0), (0,0)]), 'len': result.shape[0]}


    def load_DAG_dataset(self, graph_count=3000, min_nodes=4, max_nodes=4):
        """Generate random `graph_count` weakly-connected DAGs

        :param graph_count Int:     Number of graphs to produce
        :param min_nodes Int:       Minimum number of nodes in any graph
        :param max_nodes Int:       Maximum number of nodes in any graph

        :return List:               List of random graphs generated using given params
        """

        retval = []

        while len(retval) < graph_count:
            n = np.random.choice((range(min_nodes, max_nodes+1)))
            A = np.zeros((n,n))

            for i in range(1,n):
                for j in range(i):
                    A[i][j] = random.choice([0,1])

            G = nx.to_networkx_graph(A, create_using=nx.DiGraph)
            if nx.is_weakly_connected(G):
                self.max_node_count = max(self.max_node_count, G.number_of_nodes())
                retval.append(G)

        return retval

    def load_citeseer_ego_dags(self, min_nodes=7, max_nodes=30):
        """Loads citeseer network from disk, and generates 3-hop
         weakly-connected ego-DAGs centered at random nodes"""

        edges = []
        fpath = "dataset/EGO/citeseer.cites"
        for line in open(fpath):
            edges.append(line.strip("\n").split("\t"))

        full_g = nx.DiGraph()
        full_g.add_edges_from(edges)

        g = full_g.subgraph(max(nx.weakly_connected_components(full_g), key=len))
        g = nx.convert_node_labels_to_integers(g)

        # Self-loops essentially meaningless
        g.remove_edges_from(nx.selfloop_edges(g))

        dags = []
        for node in list(g.nodes):
            ego_g = nx.ego_graph(g, node, radius=3, undirected=True)
            if nx.is_directed_acyclic_graph(ego_g):
                n_nodes = ego_g.number_of_nodes()
                if (n_nodes >= min_nodes and n_nodes <= max_nodes):
                    dags.append(ego_g)
                    self.max_node_count = max(self.max_node_count, n_nodes)
        np.random.shuffle(dags)
        return dags[:200]
