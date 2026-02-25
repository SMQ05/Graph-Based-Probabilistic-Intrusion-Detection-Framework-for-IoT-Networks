"""
Attack Simulation Module
=========================
Simulates various cyber-attack scenarios on IoT network graphs.

Supported attack types:
    - **DDoS**: Inflates edge weights on a targeted node to simulate
      volumetric flooding.
    - **Node compromise**: Adds high-weight edges to a compromised node,
      increasing its centrality.
    - **Edge rewiring**: Randomly removes existing edges and adds new ones,
      destabilising network structure.
"""

from typing import Optional

import networkx as nx
import numpy as np


def ddos_attack(
    G: nx.DiGraph,
    target_node: Optional[int] = None,
    inflation_factor: float = 10.0,
    seed: Optional[int] = None,
) -> nx.DiGraph:
    """Simulate a DDoS attack by inflating edge weights toward a target node.

    Parameters
    ----------
    G : nx.DiGraph
        The original network graph.
    target_node : int, optional
        Node to target. If ``None``, the node with the highest in-degree
        is selected.
    inflation_factor : float
        Multiplicative factor applied to incoming edge weights.
    seed : int, optional
        Random seed (unused here but kept for API consistency).

    Returns
    -------
    nx.DiGraph
        A copy of the graph with inflated edge weights on the target.
    """
    G_attacked = G.copy()

    if target_node is None:
        target_node = max(G.nodes(), key=lambda n: G.in_degree(n, weight="weight"))

    for u, v, data in G_attacked.edges(data=True):
        if v == target_node:
            data["weight"] *= inflation_factor

    return G_attacked


def node_compromise(
    G: nx.DiGraph,
    compromised_node: Optional[int] = None,
    num_new_edges: int = 15,
    edge_weight: float = 20.0,
    seed: Optional[int] = None,
) -> nx.DiGraph:
    """Simulate a node compromise by adding high-weight edges.

    The compromised node gains additional outgoing edges to random nodes,
    increasing its out-degree centrality.

    Parameters
    ----------
    G : nx.DiGraph
        The original network graph.
    compromised_node : int, optional
        Node to compromise. If ``None``, a random node is chosen.
    num_new_edges : int
        Number of new outgoing edges to add.
    edge_weight : float
        Weight assigned to newly added edges.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    nx.DiGraph
        A copy of the graph with the compromised node.
    """
    rng = np.random.default_rng(seed)
    G_attacked = G.copy()
    nodes = list(G.nodes())

    if compromised_node is None:
        compromised_node = rng.choice(nodes)

    targets = [n for n in nodes if n != compromised_node and not G_attacked.has_edge(compromised_node, n)]
    num_to_add = min(num_new_edges, len(targets))
    chosen = rng.choice(targets, size=num_to_add, replace=False)

    for t in chosen:
        G_attacked.add_edge(compromised_node, int(t), weight=edge_weight)

    return G_attacked


def edge_rewiring(
    G: nx.DiGraph,
    rewire_fraction: float = 0.1,
    seed: Optional[int] = None,
) -> nx.DiGraph:
    """Simulate network destabilisation by randomly rewiring edges.

    A fraction of edges are removed and replaced by new random edges.

    Parameters
    ----------
    G : nx.DiGraph
        The original network graph.
    rewire_fraction : float
        Fraction of edges to rewire, in [0, 1].
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    nx.DiGraph
        A copy of the graph with rewired edges.
    """
    rng = np.random.default_rng(seed)
    G_attacked = G.copy()
    edges = list(G_attacked.edges())
    num_rewire = max(1, int(len(edges) * rewire_fraction))
    nodes = list(G.nodes())

    indices = rng.choice(len(edges), size=min(num_rewire, len(edges)), replace=False)
    for idx in indices:
        u, v = edges[idx]
        if G_attacked.has_edge(u, v):
            G_attacked.remove_edge(u, v)

        # Add a new random edge
        for _ in range(100):
            nu = rng.choice(nodes)
            nv = rng.choice(nodes)
            if nu != nv and not G_attacked.has_edge(nu, nv):
                weight = max(0.1, rng.normal(5.0, 2.0))
                G_attacked.add_edge(int(nu), int(nv), weight=weight)
                break

    return G_attacked
