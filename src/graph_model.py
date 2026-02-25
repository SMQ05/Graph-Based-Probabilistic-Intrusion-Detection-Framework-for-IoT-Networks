"""
Graph Model Module
==================
Models IoT networks as weighted directed graphs.

Mathematical Foundation:
    G = (V, E, W) where V is a set of nodes (IoT devices), E is a set of
    directed edges (communication links), and W: E -> R+ assigns positive
    weights representing traffic intensity.

    Adjacency matrix A:  A[i][j] = w_{ij} if (i, j) in E, else 0
    Degree matrix D:     D[i][i] = sum_j A[i][j]  (out-degree for directed)
    Graph Laplacian:     L = D - A
    Normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}
"""

from typing import Optional

import networkx as nx
import numpy as np


def generate_iot_network(
    num_nodes: int = 100,
    density: float = 0.05,
    weight_mean: float = 5.0,
    weight_std: float = 1.5,
    seed: Optional[int] = None,
) -> nx.DiGraph:
    """Generate a synthetic IoT network as a weighted directed graph.

    Parameters
    ----------
    num_nodes : int
        Number of IoT devices (nodes) in the network.
    density : float
        Edge density in [0, 1]. Fraction of possible edges that exist.
    weight_mean : float
        Mean of the Gaussian distribution for edge weights (traffic intensity).
    weight_std : float
        Standard deviation for edge weights.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    nx.DiGraph
        A weighted directed graph representing the IoT network.
    """
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    max_edges = num_nodes * (num_nodes - 1)
    num_edges = int(density * max_edges)

    edges_added = 0
    while edges_added < num_edges:
        u = rng.integers(0, num_nodes)
        v = rng.integers(0, num_nodes)
        if u != v and not G.has_edge(u, v):
            weight = max(0.1, rng.normal(weight_mean, weight_std))
            G.add_edge(u, v, weight=weight)
            edges_added += 1

    return G


def get_adjacency_matrix(G: nx.DiGraph) -> np.ndarray:
    """Return the weighted adjacency matrix A of the graph.

    A[i][j] = w_{ij} if edge (i,j) exists, else 0.
    """
    return nx.to_numpy_array(G, weight="weight")


def get_degree_matrix(A: np.ndarray) -> np.ndarray:
    """Return the out-degree matrix D = diag(A * 1).

    D[i][i] = sum_j A[i][j].
    """
    return np.diag(A.sum(axis=1))


def get_laplacian(A: np.ndarray) -> np.ndarray:
    """Return the graph Laplacian L = D - A."""
    D = get_degree_matrix(A)
    return D - A


def get_normalized_laplacian(A: np.ndarray) -> np.ndarray:
    """Return the symmetric normalized Laplacian L_norm = D^{-1/2} L D^{-1/2}.

    For nodes with zero degree, the corresponding diagonal entry is set to 0.
    """
    D = get_degree_matrix(A)
    L = D - A

    d_inv_sqrt = np.zeros_like(np.diag(D))
    nonzero = np.diag(D) > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(np.diag(D)[nonzero])
    D_inv_sqrt = np.diag(d_inv_sqrt)

    return D_inv_sqrt @ L @ D_inv_sqrt


def get_markov_transition_matrix(A: np.ndarray) -> np.ndarray:
    """Return the row-stochastic Markov transition matrix P.

    P[i][j] = A[i][j] / D[i][i], representing the probability of
    transitioning from node i to node j.

    Rows with zero out-degree are set to uniform 1/n.
    """
    row_sums = A.sum(axis=1, keepdims=True)
    n = A.shape[0]
    P = np.where(row_sums > 0, A / row_sums, 1.0 / n)
    return P
