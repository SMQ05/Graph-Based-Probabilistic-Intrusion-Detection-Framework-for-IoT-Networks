"""
Graph Metrics Module
====================
Computes structural and spectral features of the IoT network graph.

Extracted metrics include degree distribution, centrality measures,
clustering coefficients, spectral properties of the Laplacian, PageRank,
network density, and connected component counts.
"""

from typing import Any, Dict, List

import networkx as nx
import numpy as np

from .graph_model import get_adjacency_matrix, get_laplacian


def degree_distribution(G: nx.DiGraph) -> Dict[str, List[int]]:
    """Return in-degree and out-degree sequences.

    Returns
    -------
    dict
        Keys ``'in_degree'`` and ``'out_degree'``, each mapping to a list of
        degree values ordered by node index.
    """
    nodes = sorted(G.nodes())
    return {
        "in_degree": [G.in_degree(n, weight="weight") for n in nodes],
        "out_degree": [G.out_degree(n, weight="weight") for n in nodes],
    }


def betweenness_centrality(G: nx.DiGraph) -> Dict[int, float]:
    """Return betweenness centrality for each node."""
    return nx.betweenness_centrality(G, weight="weight")


def clustering_coefficient(G: nx.DiGraph) -> Dict[int, float]:
    """Return local clustering coefficient for each node.

    Uses the undirected projection for clustering computation.
    """
    return nx.clustering(G.to_undirected())


def laplacian_eigenvalues(G: nx.DiGraph) -> np.ndarray:
    """Return sorted eigenvalues of the graph Laplacian L = D - A.

    Eigenvalues are returned in ascending order.
    """
    A = get_adjacency_matrix(G)
    L = get_laplacian(A)
    eigenvalues = np.sort(np.real(np.linalg.eigvals(L)))
    return eigenvalues


def spectral_gap(G: nx.DiGraph) -> float:
    """Return the spectral gap lambda_2 - lambda_1.

    The spectral gap measures algebraic connectivity; a smaller spectral gap
    under attack indicates reduced network robustness.
    """
    eigenvalues = laplacian_eigenvalues(G)
    if len(eigenvalues) < 2:
        return 0.0
    return float(eigenvalues[1] - eigenvalues[0])


def pagerank(G: nx.DiGraph, alpha: float = 0.85) -> Dict[int, float]:
    """Return PageRank scores for each node.

    Parameters
    ----------
    alpha : float
        Damping factor in [0, 1].
    """
    return nx.pagerank(G, alpha=alpha, weight="weight")


def network_density(G: nx.DiGraph) -> float:
    """Return the density of the directed graph."""
    return nx.density(G)


def connected_components_count(G: nx.DiGraph) -> int:
    """Return the number of weakly connected components."""
    return nx.number_weakly_connected_components(G)


def compute_all_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    """Compute and return all graph metrics as a structured dictionary.

    Returns
    -------
    dict
        Contains keys: ``degree_distribution``, ``betweenness_centrality``,
        ``clustering_coefficient``, ``laplacian_eigenvalues``,
        ``spectral_gap``, ``pagerank``, ``density``,
        ``connected_components``.
    """
    return {
        "degree_distribution": degree_distribution(G),
        "betweenness_centrality": betweenness_centrality(G),
        "clustering_coefficient": clustering_coefficient(G),
        "laplacian_eigenvalues": laplacian_eigenvalues(G),
        "spectral_gap": spectral_gap(G),
        "pagerank": pagerank(G),
        "density": network_density(G),
        "connected_components": connected_components_count(G),
    }
