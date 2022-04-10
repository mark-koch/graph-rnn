import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def average_degree(graph):
    """Return average degree of a networkx graph."""
    degrees = list(graph.degree)
    total_degree = 0
    for pair in degrees:
        total_degree += pair[1]
    return float(total_degree) / len(degrees)


def _avg_networkx_metric_dict(graph, fn):
    """Helper function to compute the average value from a dict
     returned by a networkx graph metric function."""
    vals = list(fn(graph).values())
    return np.mean(np.array(vals))


def average_degree_centrality(graph):
    """Return average degree centrality of a networkx graph."""
    return _avg_networkx_metric_dict(graph, nx.degree_centrality)


def average_betweenness_centrality(graph):
    """Return average betweenness centrality of a networkx graph."""
    return _avg_networkx_metric_dict(graph, nx.betweenness_centrality)


def average_closeness_centrality(graph):
    """Return average closeness centrality of a networkx graph."""
    return _avg_networkx_metric_dict(graph, nx.closeness_centrality)


def average_eigenvector_centrality(graph):
    """Return average eigenvector centrality of a networkx graph."""
    return _avg_networkx_metric_dict(graph, nx.eigenvector_centrality)


def get_histogram_of_clustering_coeffs(graph):
    """Get the clustering coefficients of the nodes in a graph as a list, converted to a histogram."""
    clustering_coeffs = nx.clustering(graph)
    list_coeffs = list(clustering_coeffs.values())
    # conversion to histogram is replicated from Stanford code so we can ensure reproducibility.
    hist, _ = np.histogram(list_coeffs, bins=100, range=(0.0, 1.0), density=False)
    return hist

def test():
    graph = nx.grid_graph([10,5])
    print(get_histogram_of_clustering_coeffs(graph))
    print(average_degree(graph))
    print(average_degree_centrality(graph))
    print(average_betweenness_centrality(graph))
    # plt.figure()
    # nx.draw(graph)
    # plt.show()


# test()