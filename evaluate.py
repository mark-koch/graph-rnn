"""Evaluate performance of the graph generation model through visualization and metrics."""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch

import mmd
import data
import extension_data
from data import GraphDataSet
import generate
import model
import graph_metrics
import mmd_stanford_impl
import orbit_stats




def generated_graph_to_networkx(list_adj_vecs, directed=False):
    """
    Convert output of graph generation from model to networkx graph object.

    :param list_adj_vecs: list of torch tensors, each of which is adjacency vector for a node
    :return: networkx graph object
    """
    adj_matrix = np.array(list_adj_vecs)
    return nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph if directed else None)


def draw_generated_graph(list_adj_vecs, file_name="graph", directed=False):
    """
    Draw output of graph generation from model.

    :param list_adj_vecs: list of torch tensors, each of which is adjacency vector for a node
    :param file_name: the file name of the outputted graph drawing
    """
    graph = generated_graph_to_networkx(list_adj_vecs, directed)
    plt.figure()
    pos = nx.spring_layout(graph, k=1 / (np.sqrt(graph.number_of_nodes())), iterations=1000)
    nx.draw(graph, pos=pos)
    plt.savefig(file_name)


def compare_graphs_mmd(graph1, graph2, mmd_func):
    """
    Return MMD (maximum mean discrepancy) score between two graphs.

    :param graph1: networkx graph object of first graph
    :param graph2: networkx graph object of second graph
    :param mmd_func: MMD function to use
    :return: the MMD score between two graphs
    """
    adj_mat1 = nx.to_numpy_array(graph1)
    adj_mat2 = nx.to_numpy_array(graph2)
    return mmd_func(adj_mat1, adj_mat2)


def _diff_func(graph1, graph2, graph_metric_fn):
    """Applies a function to get a metric for each graph and returns the absolute
    value of the distance between them."""
    return abs(graph_metric_fn(graph1) - graph_metric_fn(graph2))


def compare_graphs_avg_degree(graph1, graph2):
    """
    Return difference between average degree of two networkx graphs.

    :param graph1: first networkx graph
    :param graph2: second networkx graph
    :return: absolute value of difference in average degree between two graphs
    """
    return _diff_func(graph1, graph2, graph_metrics.average_degree)


def compare_graphs_avg_clustering_coeff(graph1, graph2):
    """Return difference between avgerage clustering coefficients """
    avg1 = np.mean(np.array(list(nx.clustering(graph1).values())))
    avg2 = np.mean(np.array(list(nx.clustering(graph2).values())))
    return abs(avg1 - avg2)


def compare_graphs_avg_orbit_stats(graph1, graph2):
    """Return difference between avgerage clustering coefficients """
    avg1 = np.mean(np.array(get_orbit_stats([graph1])))
    avg2 = np.mean(np.array(get_orbit_stats([graph2])))
    return abs(avg1 - avg2)


def compare_graphs_avg_degree_centrality(graph1, graph2):
    """Return difference between average degree centrality of two networkx graphs."""
    return _diff_func(graph1, graph2, graph_metrics.average_degree_centrality)


def compare_graphs_avg_betweenness_centrality(graph1, graph2):
    """Return difference between average betweenness centrality of two networkx graphs."""
    return _diff_func(graph1, graph2, graph_metrics.average_betweenness_centrality)


def compare_graphs_avg_closeness_centrality(graph1, graph2):
    """Return difference between average closeness centrality of two networkx graphs."""
    return _diff_func(graph1, graph2, graph_metrics.average_closeness_centrality)


def compare_graphs_avg_eigenvector_centrality(graph1, graph2):
    """Return difference between average eigenvector centrality of two networkx graphs."""
    return _diff_func(graph1, graph2, graph_metrics.average_eigenvector_centrality)


def compare_graphs_transitivity(graph1, graph2):
    """Return difference between transitivity (triadic closure) of two networkx graphs."""
    return _diff_func(graph1, graph2, nx.transitivity)


def compare_graphs_density(graph1, graph2):
    """Return difference between average density of two networkx graphs."""
    return _diff_func(graph1, graph2, nx.density)


def _generate_graph_attribute_list(graph_list, fn):
    """Return a list of the result of applying the given function to each graph in the given list of graphs.

    :param graph_list: a list of networkx graphs
    :param fn: the function to apply to each list
    :return: a list of objects resulting from applying the function to a networkx graph
    """
    graph_attribute_list = []
    for graph in graph_list:
        graph_attribute_list.append(fn(graph))
    return graph_attribute_list


def _mmd_comparison_helper_func(graph_list1, graph_list2, mmd_func, metric_func):
    """Generates list of graph attributes using given metric function and runs MMD on the resulting list of lists."""
    list1 = _generate_graph_attribute_list(graph_list1, metric_func)
    list2 = _generate_graph_attribute_list(graph_list2, metric_func)
    return mmd_func(list1, list2)


def compare_graphs_mmd_degree(graph_list1, graph_list2, mmd_func):
    """Return MMD between distributions of degrees between two lists of networkx graphs."""
    return _mmd_comparison_helper_func(graph_list1, graph_list2, mmd_func, nx.degree_histogram)


def compare_graphs_mmd_clustering_coeff(graph_list1, graph_list2, mmd_func):
    """Return MMD between distributions of clustering coefficients between two lists of networkx graphs."""
    return _mmd_comparison_helper_func(graph_list1, graph_list2, mmd_func,
                                       graph_metrics.get_histogram_of_clustering_coeffs)


def get_orbit_stats(graph_list):
    """Given a list of graphs (networkx graph objects), return the orbit statistics, a list of lists.
    Based on Stanford code."""
    total_counts = []
    for graph in graph_list:
        orbit_counts = orbit_stats.orca(graph)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / graph.number_of_nodes()
        total_counts.append(orbit_counts_graph)
    return total_counts


def compare_graphs_mmd_orbit_stats(graph_list1, graph_list2, mmd_func):
    """Given two graph lists and an MMD function, return the MMD between the orbit statistics of the two graph lists."""
    total_counts1 = np.array(get_orbit_stats(graph_list1))
    total_counts2 = np.array(get_orbit_stats(graph_list2))
    return mmd_func(total_counts1, total_counts2)


def _generate_graph_attribute_list_dict(graph_list, fn):
    """Return a list of the result of applying the given function to each graph in the given list of graphs.
    Used for networkx graph metrics that are outputted as a dict.

    :param graph_list: a list of networkx graphs
    :param fn: the function to apply to each list
    :return: a list of objects resulting from applying the function to a networkx graph
    """
    graph_attribute_list = []
    for graph in graph_list:
        vals = np.array(list(fn(graph).values()))
        graph_attribute_list.append(vals)
    return graph_attribute_list


def _mmd_helper_func_dict(graph_list1, graph_list2, mmd_func, metric_func):
    """Generates list of graph attributes using given metric function (for graph metrics outputting a dict)
    and runs MMD on the resulting list of lists."""
    list1 = np.array(_generate_graph_attribute_list_dict(graph_list1, metric_func), dtype=object)
    list2 = np.array(_generate_graph_attribute_list_dict(graph_list2, metric_func), dtype=object)
    return mmd_func(list1, list2)


def compare_graphs_mmd_degree_centrality(graph_list1, graph_list2, mmd_func):
    """Return MMD between distributions of degree centrality between two lists of networkx graphs."""
    return _mmd_helper_func_dict(graph_list1, graph_list2, mmd_func, nx.degree_centrality)


def compare_graphs_mmd_betweenness_centrality(graph_list1, graph_list2, mmd_func):
    """Return MMD between distributions of betweenness centrality between two lists of networkx graphs."""
    return _mmd_helper_func_dict(graph_list1, graph_list2, mmd_func, nx.betweenness_centrality)


def compare_graphs_mmd_closeness_centrality(graph_list1, graph_list2, mmd_func):
    """Return MMD between distributions of closeness centrality between two lists of networkx graphs."""
    return _mmd_helper_func_dict(graph_list1, graph_list2, mmd_func, nx.closeness_centrality)


def generate_graph(models, num_nodes):
    """
    Use the trained model to generate a graph with a specified number of nodes.

    :param models: pytorch model objects to use
    :param num_nodes: number of nodes in the graph to be outputted
    :return: networkx object of the generated graph
    """
    node_model, edge_model, input_size, edge_gen_function, mode = models
    while True:
        adj_matrix = generate.generate(num_nodes, node_model, edge_model, input_size, edge_gen_function, mode)
        g = nx.to_networkx_graph(adj_matrix, create_using=nx.DiGraph if mode != "undirected" else None)
        if g.number_of_nodes() > 0:
            return g


def generate_new_graphs(test_graphs, models, graph_generator_mode, dataset_name, log_interval=20):
    """
    Return list of generated networkx graphs, each with same number of nodes as the corresponding test graph.

    :param test_graphs: a list of networkx graphs comprising test dataset
    :param models: pytorch model objects to use
    :param graph_generator_mode: the name of the generator method to evaluate
    :param dataset_name: the name of the dataset used to train the graph generation model
    :param log_interval: the number of graphs generated after which to print each progress indicator
    :return: a list of networkx graphs generated from trained model
    """
    print("Generating", len(test_graphs), "graphs...")

    generated_graphs = []

    dataset = data.GraphDataSet(dataset_name, training=True)
    np.random.seed(123)
    train_graph_dataset = dataset.graphs[dataset.start_idx: dataset.start_idx + dataset.length]

    for i, test_graph in enumerate(test_graphs):
        num_nodes = np.shape(test_graph)[0]

        if graph_generator_mode == 'GraphRNN':
            new_graph = generate_graph(models, num_nodes)

        elif graph_generator_mode == "BA":
            np.random.shuffle(train_graph_dataset)
            idx = (np.abs(np.array([g.number_of_nodes() for g in train_graph_dataset]) - num_nodes)).argmin()
            g = train_graph_dataset[idx]
            m = int((num_nodes - np.sqrt(max(0, num_nodes**2-4*g.number_of_edges())))/2)
            if m < 1:
                m = 1
            new_graph = nx.barabasi_albert_graph(num_nodes, m)

        elif graph_generator_mode == "Gnp":
            np.random.shuffle(train_graph_dataset)
            idx = (np.abs(np.array([g.number_of_nodes() for g in train_graph_dataset]) - num_nodes)).argmin()
            g = train_graph_dataset[idx]
            p = float(g.number_of_edges()) / ((g.number_of_nodes() - 1) * g.number_of_nodes() / 2)
            new_graph = nx.fast_gnp_random_graph(num_nodes, p)

        generated_graphs.append(new_graph)

        if (i - 1) % log_interval == 0:
            print("generated", i, "graphs")
    print("Done generating graphs.")
    return generated_graphs


def compute_average_metric_score(test_graphs, generated_graphs, metric_comparison_fn):
    """
    Compute average metric score between every graph in original testing dataset its corresponding graph generated by model.

    :param test_graphs: a list of NX graphs from the test dataset
    :param generated_graphs: a list of networkx graphs generated from trained model
    :param metric_comparison_fn: function that takes in two networkx graphs and outputs the comparison metric score
    :return: the average metric score between all the graphs
    """
    total_metric_score = 0
    count = 0

    for i, test_graph in enumerate(test_graphs):
        generated_graph = generated_graphs[i]
        metric_val = metric_comparison_fn(test_graph, generated_graph)

        total_metric_score += metric_val
        count += 1

    avg_metric_score = total_metric_score / count
    return avg_metric_score


def compute_average_metric_score_MMD(test_graphs, generated_graphs, metric_comparison_fn):
    """
    Compute average metric score across an entire distribution of graphs.
    The purpose of this function is to compute the MMD score between the distributions of a certain graph
    metric across two entire lists of graphs at once.

    :param test_graphs: a list of networkx graphs from the test dataset
    :param generated_graphs: a list of networkx graphs generated from trained model
    :param metric_comparison_fn: function that takes in two lists of networkx graphs and outputs the comparison metric score
    :return: the  metric score between all the graphs
    """
    return metric_comparison_fn(test_graphs, generated_graphs)


def run_all_metrics(metric_info, dataset_name, model_path, generator_name, f, small_dataset=False):
    """
    Run all evaluation metrics on the test dataset and prints their results.

    :param metric_info: list of tuples containing names of the metric and corresponding metric comparison functions
    :param dataset_name: string with name of the dataset from data.GraphDataSet to be used
    :param model_path: string with file path of the model
    :param generator_name: the name of the generator method to evaluate
    :param small_dataset: if True, do not use the full dataset for running all the metrics.
                        Use only a small number of data points for testing.
    """
    dataset = data.GraphDataSet(dataset_name, training=False)
    # dataset = data.GraphDataSet(dataset_name, training=True)
    if small_dataset:
        test_graph_dataset = dataset.graphs[dataset.start_idx: dataset.start_idx + 10]
    else:
        test_graph_dataset = dataset.graphs[dataset.start_idx: dataset.start_idx + dataset.length]
    if generator_name == "GraphRNN":
        models = generate.load_model_from_config(model_path)
    else:
        models = None
    generated_graphs = generate_new_graphs(test_graph_dataset, models, generator_name, dataset_name)
    for i in metric_info:
        name, fn = i
        if "MMD" not in name:
            # the non-MMD metrics compare each pair of graphs one at a time and return the average score
            val = compute_average_metric_score(test_graph_dataset, generated_graphs, fn)
        else:
            # the MMD metrics compare the entire datasets at once
            val = compute_average_metric_score_MMD(test_graph_dataset, generated_graphs, fn)
        f.write(name + ": " + str(val) + "\n")


def evaluate_all_models(model_info, metric_info, generator_name, f, small_dataset=False):
    """
    Run all evaluation metrics on all the trained models.

    :param model_info: list of tuples containing names of the dataset, dataset name from data.py, and file path of trained model
    :param metric_info: list of tuples containing names of the metric and corresponding metric comparison functions
    :param generator_name: the name of the generator method to evaluate
    :param small_dataset: if True, do not use the full dataset for running all the metrics.
                            Use only a small number of data points for testing.
    """
    for i in model_info:
        nn_model_type, dataset_name, model_file = i
        if generator_name == "GraphRNN":
            f.write(nn_model_type + " ")
            print(nn_model_type, dataset_name)
        f.write(dataset_name + " Dataset Results:" + "\n")
        run_all_metrics(metric_info, dataset_name, model_file, generator_name, f, small_dataset)
        f.write("===============" + "\n")


def run_all_generators(generator_list, model_info, metric_info, f, small_dataset=False):
    """
    Run the evaluation suite for all the graph generators.

    :param generator_list: A list of all generation methods used to generate the graphs
    :param model_info: list of tuples containing names of the dataset, dataset name from data.py, and file path of trained model
    :param metric_info: list of tuples containing names of the metric and corresponding metric comparison functions
    :param small_dataset: if True, do not use the full dataset for running all the metrics.
                            Use only a small number of data points for testing.
    """
    for generator in generator_list:
        print("Using generator " + generator)
        f.write("Using generator " + generator + "\n")
        evaluate_all_models(model_info, metric_info, generator, f, small_dataset)
        f.write("===============================================\n")


if __name__ == '__main__':
    generator_list_vals = ["GraphRNN", "BA", "Gnp"]

    # put info for all metrics to be run into metric_info
    mmd_stanford_fn_no_hist = lambda x, y: mmd_stanford_impl.compute_mmd(x, y, kernel=mmd_stanford_impl.gaussian_emd,
                                                                         is_hist=False)
    mmd_stanford_fn_is_hist = lambda x, y: mmd_stanford_impl.compute_mmd(x, y, kernel=mmd_stanford_impl.gaussian_emd,
                                                                         is_hist=True)
    mmd_stanford_fn_is_hist_clustering_settings = lambda x, y: mmd_stanford_impl.compute_mmd(x, y,
                                                                                             kernel=mmd_stanford_impl.gaussian_emd,
                                                                                             is_hist=True,
                                                                                             sigma=1.0 / 10,
                                                                                             distance_scaling=100)
    mmd_stanford_fn_orbit_settings = lambda x, y: mmd_stanford_impl.compute_mmd(x, y, kernel=mmd_stanford_impl.gaussian,
                                                                                is_hist=False, sigma=30.0)

    metric_info_vals = [
        ("MMD of degree distribution",
            lambda x, y: compare_graphs_mmd_degree(x, y, mmd_stanford_fn_is_hist)),
        ("MMD of clustering coefficient distribution",
            lambda x, y: compare_graphs_mmd_clustering_coeff(x, y, mmd_stanford_fn_is_hist_clustering_settings)),
        ("MMD of orbit stats distribution",
            lambda x, y: compare_graphs_mmd_orbit_stats(x, y, mmd_stanford_fn_orbit_settings)),

        ("MMD of degree centrality",
            lambda x, y: compare_graphs_mmd_degree_centrality(x, y, mmd_stanford_fn_no_hist)),
        ("MMD of betweenness centrality",
            lambda x, y: compare_graphs_mmd_betweenness_centrality(x, y, mmd_stanford_fn_no_hist)),
        ("MMD of closeness centrality",
            lambda x, y: compare_graphs_mmd_closeness_centrality(x, y, mmd_stanford_fn_no_hist)),

        ("Avg of degree distribution",
         lambda x, y: compare_graphs_avg_degree(x, y)),
        ("Avg of clustering coefficient distribution",
         lambda x, y: compare_graphs_avg_clustering_coeff(x, y)),
        ("Avg of orbit stats distribution",
         lambda x, y: compare_graphs_avg_orbit_stats(x, y)),

        ("Average Degree Difference", compare_graphs_avg_degree),
        ("Average Degree Centrality Difference", compare_graphs_avg_degree_centrality),
        ("Average Betweenness Centrality Difference", compare_graphs_avg_betweenness_centrality),
        ("Average Closeness Centrality Difference", compare_graphs_avg_closeness_centrality),
        ("Density Difference", compare_graphs_density),
        ("Transitivity (Triadic Closure) Difference", compare_graphs_transitivity)
    ]

    # put info for all models to be run into model_info_vals
    model_info_vals = [
                       ("RNN", "ego-directed-multiclass",
                        "checkpoint-10000.pth"),
                       ]

    f = open("eval_results.txt", "w")

    run_all_generators(generator_list_vals, model_info_vals, metric_info_vals, f, small_dataset=False)

    f.write("============== FINISHED RUN ====================\n\n")
    f.close()
    print("Done.")
