"""
Visualization Module
====================
Generates publication-quality plots comparing baseline and attacked network
states, including topology views, degree distributions, spectral properties,
anomaly scores, and adjacency heatmaps.

All plots use matplotlib exclusively.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .graph_model import get_adjacency_matrix
from .metrics import (
    degree_distribution,
    laplacian_eigenvalues,
    spectral_gap,
)
from .anomaly import zscore_anomaly_nodes


def _ensure_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_network_topology(
    G_normal: nx.DiGraph,
    G_attacked: nx.DiGraph,
    output_dir: str = "data",
    filename: str = "topology_comparison.png",
) -> None:
    """Plot normal vs attacked network topology side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    pos = nx.spring_layout(G_normal, seed=42)

    nx.draw_networkx(
        G_normal, pos=pos, ax=axes[0],
        node_size=30, with_labels=False, edge_color="gray",
        node_color="steelblue", alpha=0.8, width=0.3,
    )
    axes[0].set_title("Normal Network", fontsize=14)

    nx.draw_networkx(
        G_attacked, pos=pos, ax=axes[1],
        node_size=30, with_labels=False, edge_color="gray",
        node_color="crimson", alpha=0.8, width=0.3,
    )
    axes[1].set_title("Attacked Network", fontsize=14)

    plt.tight_layout()
    plt.savefig(str(_ensure_dir(output_dir) / filename), dpi=150)
    plt.close(fig)


def plot_degree_distribution(
    G_normal: nx.DiGraph,
    G_attacked: nx.DiGraph,
    output_dir: str = "data",
    filename: str = "degree_distribution.png",
) -> None:
    """Plot in-degree and out-degree distribution comparison."""
    dd_normal = degree_distribution(G_normal)
    dd_attacked = degree_distribution(G_attacked)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(dd_normal["in_degree"], bins=25, alpha=0.6, label="Normal", color="steelblue")
    axes[0].hist(dd_attacked["in_degree"], bins=25, alpha=0.6, label="Attacked", color="crimson")
    axes[0].set_title("In-Degree Distribution")
    axes[0].set_xlabel("Weighted In-Degree")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].hist(dd_normal["out_degree"], bins=25, alpha=0.6, label="Normal", color="steelblue")
    axes[1].hist(dd_attacked["out_degree"], bins=25, alpha=0.6, label="Attacked", color="crimson")
    axes[1].set_title("Out-Degree Distribution")
    axes[1].set_xlabel("Weighted Out-Degree")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(str(_ensure_dir(output_dir) / filename), dpi=150)
    plt.close(fig)


def plot_eigenvalue_spectrum(
    G_normal: nx.DiGraph,
    G_attacked: nx.DiGraph,
    output_dir: str = "data",
    filename: str = "eigenvalue_spectrum.png",
) -> None:
    """Plot eigenvalue spectrum of the Laplacian for both graph states."""
    eig_normal = laplacian_eigenvalues(G_normal)
    eig_attacked = laplacian_eigenvalues(G_attacked)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eig_normal, "o-", markersize=2, label="Normal", color="steelblue", alpha=0.7)
    ax.plot(eig_attacked, "s-", markersize=2, label="Attacked", color="crimson", alpha=0.7)
    ax.set_title("Laplacian Eigenvalue Spectrum")
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    ax.legend()

    plt.tight_layout()
    plt.savefig(str(_ensure_dir(output_dir) / filename), dpi=150)
    plt.close(fig)


def plot_spectral_gap_comparison(
    G_normal: nx.DiGraph,
    G_attacked: nx.DiGraph,
    output_dir: str = "data",
    filename: str = "spectral_gap_comparison.png",
) -> None:
    """Bar chart comparing spectral gap before and after attack."""
    sg_normal = spectral_gap(G_normal)
    sg_attacked = spectral_gap(G_attacked)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Normal", "Attacked"], [sg_normal, sg_attacked],
        color=["steelblue", "crimson"],
    )
    ax.set_ylabel("Spectral Gap (λ₂ − λ₁)")
    ax.set_title("Spectral Gap Comparison")
    for bar, val in zip(bars, [sg_normal, sg_attacked]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(str(_ensure_dir(output_dir) / filename), dpi=150)
    plt.close(fig)


def plot_anomaly_scores(
    G_normal: nx.DiGraph,
    G_attacked: nx.DiGraph,
    threshold: float = 2.0,
    output_dir: str = "data",
    filename: str = "anomaly_scores.png",
) -> None:
    """Histogram of per-node anomaly Z-scores."""
    result = zscore_anomaly_nodes(G_normal, G_attacked, threshold=threshold)
    z_scores = result["z_scores"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(z_scores, bins=30, color="darkorange", edgecolor="black", alpha=0.7)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {threshold}")
    ax.set_title("Node-Level Anomaly Z-Scores")
    ax.set_xlabel("Max |Z-Score|")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()
    plt.savefig(str(_ensure_dir(output_dir) / filename), dpi=150)
    plt.close(fig)


def plot_adjacency_heatmap(
    G_normal: nx.DiGraph,
    G_attacked: nx.DiGraph,
    output_dir: str = "data",
    filename: str = "adjacency_heatmap.png",
    max_nodes: int = 80,
) -> None:
    """Heatmap of adjacency matrices before and after attack.

    For readability, only the first ``max_nodes`` nodes are shown.
    """
    A_normal = get_adjacency_matrix(G_normal)[:max_nodes, :max_nodes]
    A_attacked = get_adjacency_matrix(G_attacked)[:max_nodes, :max_nodes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(A_normal, cmap="Blues", aspect="auto")
    axes[0].set_title("Normal Adjacency Matrix")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(A_attacked, cmap="Reds", aspect="auto")
    axes[1].set_title("Attacked Adjacency Matrix")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(str(_ensure_dir(output_dir) / filename), dpi=150)
    plt.close(fig)


def generate_all_plots(
    G_normal: nx.DiGraph,
    G_attacked: nx.DiGraph,
    output_dir: str = "data",
) -> None:
    """Generate all visualization plots and save to *output_dir*."""
    plot_network_topology(G_normal, G_attacked, output_dir)
    plot_degree_distribution(G_normal, G_attacked, output_dir)
    plot_eigenvalue_spectrum(G_normal, G_attacked, output_dir)
    plot_spectral_gap_comparison(G_normal, G_attacked, output_dir)
    plot_anomaly_scores(G_normal, G_attacked, output_dir=output_dir)
    plot_adjacency_heatmap(G_normal, G_attacked, output_dir)
