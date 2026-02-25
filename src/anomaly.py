"""
Probabilistic Anomaly Detection Module
=======================================
Implements statistical and probabilistic methods to detect anomalies in IoT
network graphs by comparing baseline (normal) and attacked states.

Methods:
    - **Z-score detection**: Flags features deviating beyond a threshold from
      the baseline mean.
    - **KL divergence**: Measures distributional shift between normal and
      attacked feature distributions.
    - **Bayesian anomaly scoring**: Computes posterior probability of the
      network being in an anomalous state given observed metrics.
    - **Markov transition comparison**: Quantifies change in transition
      probabilities via Frobenius norm.

Statistical hypothesis framework:
    H0: The network is in its normal operating state.
    H1: The network has been compromised (anomalous state).
"""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from scipy.stats import entropy

from .graph_model import get_adjacency_matrix, get_markov_transition_matrix


def _node_feature_vectors(G: nx.DiGraph) -> np.ndarray:
    """Build a feature vector for each node: [in_degree, out_degree, pagerank].

    Returns
    -------
    np.ndarray
        Shape ``(n, 3)`` feature matrix.
    """
    pr = nx.pagerank(G, weight="weight")
    nodes = sorted(G.nodes())
    features = np.array([
        [
            G.in_degree(n, weight="weight"),
            G.out_degree(n, weight="weight"),
            pr[n],
        ]
        for n in nodes
    ])
    return features


# ── Z-score anomaly detection ──────────────────────────────────────────────

def zscore_anomaly_nodes(
    G_normal: nx.DiGraph,
    G_attacked: nx.DiGraph,
    threshold: float = 2.0,
) -> Dict[str, object]:
    """Detect anomalous nodes via Z-score on feature vectors.

    For each node, a Z-score is computed per feature relative to the baseline
    distribution.  Nodes whose maximum absolute Z-score exceeds *threshold*
    are flagged.

    Parameters
    ----------
    G_normal : nx.DiGraph
        Baseline (normal) network.
    G_attacked : nx.DiGraph
        Network after attack injection.
    threshold : float
        Z-score threshold for flagging a node as anomalous.

    Returns
    -------
    dict
        ``'z_scores'``: array of per-node max |Z|,
        ``'anomalous_nodes'``: list of flagged node indices,
        ``'threshold'``: the threshold used.
    """
    feat_normal = _node_feature_vectors(G_normal)
    feat_attacked = _node_feature_vectors(G_attacked)

    mu = feat_normal.mean(axis=0)
    sigma = feat_normal.std(axis=0)
    sigma[sigma == 0] = 1e-10  # avoid division by zero

    z = np.abs((feat_attacked - mu) / sigma)
    max_z = z.max(axis=1)

    anomalous = np.where(max_z > threshold)[0].tolist()
    return {
        "z_scores": max_z,
        "anomalous_nodes": anomalous,
        "threshold": threshold,
    }


# ── KL divergence ──────────────────────────────────────────────────────────

def kl_divergence(
    G_normal: nx.DiGraph,
    G_attacked: nx.DiGraph,
    bins: int = 30,
) -> Dict[str, float]:
    """Compute KL divergence between degree distributions.

    KL(P || Q) = sum_x P(x) log(P(x) / Q(x))

    where P is the baseline distribution and Q is the attacked distribution.
    Applies Laplace smoothing to avoid log(0).

    Parameters
    ----------
    G_normal, G_attacked : nx.DiGraph
        Normal and attacked graphs.
    bins : int
        Number of histogram bins.

    Returns
    -------
    dict
        ``'kl_in_degree'``, ``'kl_out_degree'``: KL divergence values.
    """
    results: Dict[str, float] = {}
    for direction, degree_fn in [("in_degree", "in_degree"), ("out_degree", "out_degree")]:
        normal_deg = np.array([getattr(G_normal, degree_fn)(n, weight="weight") for n in sorted(G_normal.nodes())])
        attacked_deg = np.array([getattr(G_attacked, degree_fn)(n, weight="weight") for n in sorted(G_attacked.nodes())])

        all_vals = np.concatenate([normal_deg, attacked_deg])
        bin_edges = np.linspace(all_vals.min(), all_vals.max() + 1e-10, bins + 1)

        p, _ = np.histogram(normal_deg, bins=bin_edges, density=True)
        q, _ = np.histogram(attacked_deg, bins=bin_edges, density=True)

        # Laplace smoothing
        p = p + 1e-10
        q = q + 1e-10
        p = p / p.sum()
        q = q / q.sum()

        results[f"kl_{direction}"] = float(entropy(p, q))
    return results


# ── Bayesian anomaly scoring ──────────────────────────────────────────────

def bayesian_anomaly_score(
    G_normal: nx.DiGraph,
    G_attacked: nx.DiGraph,
    prior_attack: float = 0.1,
) -> Dict[str, float]:
    """Compute Bayesian posterior probability of an attack.

    Uses the spectral gap shift as the observed evidence.

    P(H1 | x) = P(x | H1) * P(H1) / P(x)

    The likelihood is modelled as the normalised absolute change in the
    spectral gap.  A larger shift increases the posterior probability.

    Parameters
    ----------
    G_normal, G_attacked : nx.DiGraph
        Normal and attacked graphs.
    prior_attack : float
        Prior probability of attack P(H1).

    Returns
    -------
    dict
        ``'posterior_attack'``: P(H1 | evidence),
        ``'spectral_gap_normal'``, ``'spectral_gap_attacked'``: gap values.
    """
    from .metrics import spectral_gap as sg_fn

    sg_normal = sg_fn(G_normal)
    sg_attacked = sg_fn(G_attacked)

    # Likelihood proportional to relative change
    delta = abs(sg_attacked - sg_normal) / (abs(sg_normal) + 1e-10)
    likelihood_attack = min(1.0, delta)
    likelihood_normal = max(0.0, 1.0 - delta)

    prior_normal = 1.0 - prior_attack
    evidence = likelihood_attack * prior_attack + likelihood_normal * prior_normal

    if evidence == 0:
        posterior = prior_attack
    else:
        posterior = (likelihood_attack * prior_attack) / evidence

    return {
        "posterior_attack": float(posterior),
        "spectral_gap_normal": float(sg_normal),
        "spectral_gap_attacked": float(sg_attacked),
    }


# ── Markov transition comparison ──────────────────────────────────────────

def markov_transition_distance(
    G_normal: nx.DiGraph,
    G_attacked: nx.DiGraph,
) -> Dict[str, float]:
    """Quantify shift in Markov transition probabilities.

    Computes the Frobenius norm ||P_normal - P_attacked||_F as a measure of
    how much transition behaviour has changed.

    Returns
    -------
    dict
        ``'frobenius_norm'``: scalar distance between transition matrices.
    """
    A_normal = get_adjacency_matrix(G_normal)
    A_attacked = get_adjacency_matrix(G_attacked)

    P_normal = get_markov_transition_matrix(A_normal)
    P_attacked = get_markov_transition_matrix(A_attacked)

    diff = np.linalg.norm(P_normal - P_attacked, ord="fro")
    return {"frobenius_norm": float(diff)}


# ── Global anomaly summary ────────────────────────────────────────────────

def global_anomaly_report(
    G_normal: nx.DiGraph,
    G_attacked: nx.DiGraph,
    zscore_threshold: float = 2.0,
    prior_attack: float = 0.1,
) -> Dict[str, object]:
    """Produce a comprehensive anomaly report comparing two graph states.

    Returns
    -------
    dict
        Combined results from Z-score, KL divergence, Bayesian scoring,
        and Markov transition comparison.
    """
    z = zscore_anomaly_nodes(G_normal, G_attacked, threshold=zscore_threshold)
    kl = kl_divergence(G_normal, G_attacked)
    bayes = bayesian_anomaly_score(G_normal, G_attacked, prior_attack=prior_attack)
    markov = markov_transition_distance(G_normal, G_attacked)

    return {
        "zscore": z,
        "kl_divergence": kl,
        "bayesian": bayes,
        "markov": markov,
    }
