"""
Main Entry Point
=================
Orchestrates the full Graph-Based Probabilistic Intrusion Detection pipeline:

1. Generate a synthetic IoT network (baseline).
2. Inject attacks (DDoS, node compromise, edge rewiring).
3. Extract graph metrics for both states.
4. Run probabilistic anomaly detection.
5. Generate comparison visualisations.
6. Print a summary report to the console.
"""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from .graph_model import generate_iot_network
from .metrics import compute_all_metrics
from .attacks import ddos_attack, node_compromise, edge_rewiring
from .anomaly import global_anomaly_report
from .visualization import generate_all_plots


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj: object) -> object:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


def run_pipeline(
    num_nodes: int = 100,
    density: float = 0.05,
    seed: int = 42,
    output_dir: str = "data",
    attack_type: str = "ddos",
) -> Dict:
    """Execute the full IDS pipeline.

    Parameters
    ----------
    num_nodes : int
        Number of IoT devices.
    density : float
        Edge density of the network.
    seed : int
        Random seed.
    output_dir : str
        Directory for output artefacts.
    attack_type : str
        One of ``'ddos'``, ``'compromise'``, ``'rewiring'``, or ``'all'``.

    Returns
    -------
    dict
        Summary of metrics and anomaly scores.
    """
    print(f"[1/5] Generating IoT network ({num_nodes} nodes, density={density}) ...")
    G_normal = generate_iot_network(num_nodes=num_nodes, density=density, seed=seed)

    # ── Attack injection ─────────────────────────────────────────────────
    print(f"[2/5] Injecting attack: {attack_type} ...")
    if attack_type == "ddos":
        G_attacked = ddos_attack(G_normal, seed=seed)
    elif attack_type == "compromise":
        G_attacked = node_compromise(G_normal, seed=seed)
    elif attack_type == "rewiring":
        G_attacked = edge_rewiring(G_normal, seed=seed)
    elif attack_type == "all":
        G_attacked = ddos_attack(G_normal, seed=seed)
        G_attacked = node_compromise(G_attacked, seed=seed)
        G_attacked = edge_rewiring(G_attacked, seed=seed)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    # ── Metrics ──────────────────────────────────────────────────────────
    print("[3/5] Computing graph metrics ...")
    metrics_normal = compute_all_metrics(G_normal)
    metrics_attacked = compute_all_metrics(G_attacked)

    # ── Anomaly detection ────────────────────────────────────────────────
    print("[4/5] Running anomaly detection ...")
    anomaly_report = global_anomaly_report(G_normal, G_attacked)

    # ── Visualisation ────────────────────────────────────────────────────
    print("[5/5] Generating visualisations ...")
    generate_all_plots(G_normal, G_attacked, output_dir=output_dir)

    # ── Summary ──────────────────────────────────────────────────────────
    summary = {
        "network": {
            "nodes": num_nodes,
            "density": density,
            "attack_type": attack_type,
        },
        "metrics_normal": {
            "spectral_gap": metrics_normal["spectral_gap"],
            "density": metrics_normal["density"],
            "connected_components": metrics_normal["connected_components"],
        },
        "metrics_attacked": {
            "spectral_gap": metrics_attacked["spectral_gap"],
            "density": metrics_attacked["density"],
            "connected_components": metrics_attacked["connected_components"],
        },
        "anomaly_report": anomaly_report,
    }

    # Save JSON report
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = out_path / "anomaly_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)
    print(f"\nReport saved to {report_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("  ANOMALY DETECTION SUMMARY")
    print("=" * 60)
    z = anomaly_report["zscore"]
    print(f"  Anomalous nodes (Z > {z['threshold']}): {len(z['anomalous_nodes'])}")
    kl = anomaly_report["kl_divergence"]
    print(f"  KL divergence (in-degree):  {kl['kl_in_degree']:.6f}")
    print(f"  KL divergence (out-degree): {kl['kl_out_degree']:.6f}")
    b = anomaly_report["bayesian"]
    print(f"  Bayesian P(attack | data):  {b['posterior_attack']:.6f}")
    m = anomaly_report["markov"]
    print(f"  Markov transition Frobenius norm: {m['frobenius_norm']:.6f}")
    print("=" * 60)

    return summary


def main() -> None:
    """CLI entry point."""
    attack_type = sys.argv[1] if len(sys.argv) > 1 else "ddos"
    run_pipeline(attack_type=attack_type)


if __name__ == "__main__":
    main()
