import numpy as np
import networkx as nx
import torch
import random
import pickle as pkl


def generate_adjacency_vector_sequence(g, node_sequence):
    """
    :param g NX Graph:          Graph for which we want to generate sequence
    :param node_sequence List:  A reordering of g's node labels (g.Nodes())

    :return:                    Upper right triangluar adjacency matrix, 0-padded
    """

    return np.tril(nx.adjacency_matrix(g, node_sequence).toarray())



def bfs(g):
    """
    :param g NX Graph:          Graph for BFS traversal

    :return :                   A randomly permuted equivalent of g, and a BFS-traversal thereof
    """

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


class GraphDataSet(torch.utils.data.Dataset):

    def __init__(self, dataset, m=None, bfs=True, training=True, train_split=0.8):
        """
        Arguments:
            dataset: String describing which dataset to load
            m: Precalculated M-value (Ref. paper)
            bfs: Set to False to disable BFS
            training: Loads the training split if set to True. Otherwise loads
                the test split.
            train_split: Percentage of data to use for training
        """

        self.max_node_count = -1
        self.training = training
        self.bfs = bfs
        self.m = m

        np.random.seed(42)

        if dataset == 'grid':
            self.graphs = self.load_grid_dataset()
        elif dataset == 'grid-small':
            self.graphs = self.load_grid_dataset(min_side=2, max_side=6)
        elif dataset == 'ba':
            self.graphs = self.load_BA_dataset()
        elif dataset == 'community':
            self.graphs = self.load_community_dataset()
        elif dataset == 'community-small':
            self.graphs = self.load_community_dataset(min_nodes=12, max_nodes=20)
        elif dataset == 'protein':
            self.graphs = self.load_protein_dataset()
        elif dataset == 'ego':
            self.graphs = self.load_ego_dataset()
        elif dataset == 'ego-small':
            self.graphs = self.load_ego_dataset(graph_count=200, min_nodes=4, max_nodes=18, radius=3)
        elif dataset == 'DAG':
            self.graphs = self.load_DAG_dataset()
        else:
            raise Exception(f"No data-loader for dataset `{dataset}`")

        for g in self.graphs:
            g.remove_edges_from(list(nx.selfloop_edges(g)))

        # Shuffle for random train/test slit
        np.random.shuffle(self.graphs)

        train_size = int(len(self.graphs) * train_split)
        self.start_idx = 0 if training else train_size
        self.length = train_size if training else len(self.graphs) - train_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        n.b. Random BFS traversal happens at this stage

        :return :   {'x': <M-length sequence vectors paddded to fit largest graph>,
                     'len': <Number of sequnce vectors actually containing data>}
        """

        g = self.graphs[self.start_idx + idx]

        if self.bfs:
            permuted_g, bfs_seq = bfs(g)
            adjacency_vector_seq = generate_adjacency_vector_sequence(permuted_g, bfs_seq)
        else:
            g = nx.convert_node_labels_to_integers(g)
            adjacency_vector_seq = generate_adjacency_vector_sequence(g, np.random.permutation(g.nodes))

        scratch = []

        for i in range(1, adjacency_vector_seq.shape[0]):
            # Data that actually can have 1s:
            critical_strip = adjacency_vector_seq[i, max(i-self.m, 0):i]
            m_dash = len(critical_strip)
            scratch.append(np.pad(critical_strip, (self.m - m_dash, 0))[::-1])

        result = np.array(scratch)
        return {'x': np.pad(result, [(0, self.max_node_count - result.shape[0]), (0,0)]), 'len': result.shape[0]}


    def load_grid_dataset(self, min_side=10, max_side=20):
        """Generate all 2D grid shaped graphs with given grid shape constraints.

        :param min_side Int:    Minimum # of vertices on grid edge
        :param max_side Int:    Maximum # of vertices on grid edge

        :return List:           List of all NX grid graphs matching given params
        """

        retval = []

        for w in range(min_side, max_side + 1):
            for h in range(min_side, max_side + 1):
                retval.append(nx.grid_graph([w, h]))

        self.max_node_count = max(self.max_node_count, max_side * max_side)

        return retval

    def load_BA_dataset(self, graph_count=500, min_nodes=100, max_nodes=200, new_edges=4):
        """Generate `graph_count` random graphs using the Barabasi-Albert model.

        :param graph_count Int:     Number of graphs to produce
        :param min_nodes Int:       Minimum number of nodes in any graph
        :param max_nodes Int:       Maximum number of nodes in any graph
        :param new_edges Int:       Number of edges to add for each new node in the BA model.

        :return List:               List of random graphs generated using the BA-model and given params
        """

        retval = []

        for _ in range(graph_count):
            node_count = np.random.randint(min_nodes, max_nodes+1)
            retval.append(nx.barabasi_albert_graph(node_count, new_edges))
            self.max_node_count = max(self.max_node_count, node_count)

        return retval
    
    def community_dataset(self, c_sizes, p_inter=0.05, p_intra=0.3):
        """Helper function to a generate random graph using the Erdős-Rényi model.

        :param c_sizes numpy_ndarray:     1-D array of number of nodes in each community in a graph
        :param p_inter Int:               Number of intercommunity edges between communities in a graph
       
        :return graph:                    Random graph generated using the Erdős-Rényi model and given params
        """    
    
        g = [nx.gnp_random_graph(c_sizes[i], p=p_intra, directed=False) for i in range(len(c_sizes))]

        G = nx.disjoint_union_all(g)

        g1 = list(g[0].nodes())
        g2 = list(g[1].nodes())

        # Adding one inter-community edge by default
        # This ensures that we have a connected graph
        n1 = random.choice(g1)
        n2 = random.choice(g2) + len(g1)
        G.add_edge(n1,n2)

        V = sum(c_sizes)
        for i in range(int(p_inter*V)):

            n1 = random.choice(g1)
            n2 = random.choice(g2) + len(g1)
            G.add_edge(n1,n2)


        return G

    def load_community_dataset(self, graph_count=500, min_nodes=60, max_nodes=160, num_communities=2, p_inter=0.05, p_intra=0.3):
        """Generate `graph_count` random graphs using the Erdős-Rényi model.

        :param graph_count Int:     Number of graphs to produce
        :param min_nodes Int:       Minimum number of nodes in any graph
        :param max_nodes Int:       Maximum number of nodes in any graph
        :param p_inter Int:         Number of intercommunity edges in any graph

        :return List:               List of random graphs generated using the Erdős-Rényi model and given params
        """

        retval = []

        for _ in range(graph_count):
            c_sizes = np.random.choice(list(range(int(min_nodes/2),int(max_nodes/2)+1)), num_communities) 
            retval.append(self.community_dataset(c_sizes, p_inter, p_intra))
            self.max_node_count = max(self.max_node_count, sum(c_sizes))

        return retval
    
    def load_protein_dataset(self, graph_count=918, min_nodes=100, max_nodes=500): 
        """Load `graph_count` protein graphs

        :param graph_count Int:     Number of graphs to produce
        :param min_nodes Int:       Minimum number of nodes in any graph
        :param max_nodes Int:       Maximum number of nodes in any graph

        :return List:               List of random graphs loaded using given params
        """
        
        G = nx.Graph()

        path = 'dataset/PROTEIN/'

        adj_mat = np.loadtxt(path+'DD_A.txt',delimiter = ',').astype(int)
        node_label = np.loadtxt(path+'DD_node_labels.txt',delimiter=',').astype(int)
        graph_indicator = np.loadtxt(path+'DD_graph_indicator.txt',delimiter=',').astype(int)
        graph_labels = np.loadtxt(path+'DD_graph_labels.txt',delimiter=',').astype(int)

        edge_tuple = list(map(tuple, adj_mat))
        G.add_edges_from(edge_tuple)

        for i in range(len(node_label)):
            G.add_node(i+1, label = node_label[i])

        G.remove_nodes_from(list(nx.isolates(G)))

        num_graphs = max(graph_indicator)
        list_nodes = np.arange(len(graph_indicator))+1

        retval = []

        for i in range(num_graphs):

            node = list_nodes[graph_indicator == i+1]
            g = G.subgraph(node)
            g.graph['label'] = graph_labels[i]

            if (g.number_of_nodes()>=min_nodes and g.number_of_nodes()<=max_nodes):
                retval.append(g)
                self.max_node_count = max(self.max_node_count, g.number_of_nodes())
            if len(retval) > graph_count:
                break

        return retval

    
    def ego_dataset(self):
        """Helper function to load citeseer graph
        :return List:               loaded citeseer graph
        """

        graph = pkl.load(open("dataset/EGO/ind.citeseer.graph",'rb'), encoding='latin1')
        G = nx.from_dict_of_lists(graph)
        
        return G

    def load_ego_dataset(self, graph_count = 757, min_nodes = 50, max_nodes = 399, radius = 3):
        """Generate `graph_count` ego graphs

        :param graph_count Int:     Number of graphs to produce
        :param min_nodes Int:       Minimum number of nodes in any graph
        :param max_nodes Int:       Maximum number of nodes in any graph

        :return List:               List of random graphs generated using given params
        """
        graph = self.ego_dataset() 
        graph_sub = [graph.subgraph(g) for g in nx.connected_components(graph)]
        graph = max(graph_sub, key=len)
        graph = nx.convert_node_labels_to_integers(graph)
        retval = []

        for i in range(graph.number_of_nodes()):
            ego = nx.ego_graph(graph, i, radius=radius)
            if ego.number_of_nodes() >= min_nodes and (ego.number_of_nodes() <= max_nodes):
                retval.append(ego)
                self.max_node_count = max(self.max_node_count, ego.number_of_nodes())

        random.shuffle(retval)

        return retval[:graph_count]
