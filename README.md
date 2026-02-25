# Graph-Based Probabilistic Intrusion Detection Framework for IoT Networks

A research-grade framework for modelling IoT networks as weighted directed graphs and detecting cyber-attacks through spectral analysis and probabilistic anomaly detection.

## Research Motivation

The proliferation of Internet of Things (IoT) devices has created large-scale, heterogeneous networks that are increasingly targeted by sophisticated cyber-attacks. Traditional intrusion detection systems (IDS) often rely on signature matching or shallow heuristics, which fail to capture the structural properties of networked systems. This project addresses that gap by:

- **Modelling IoT networks as weighted directed graphs** G = (V, E, W), where nodes represent devices and weighted edges encode traffic intensity.
- **Extracting spectral and structural features** — including the graph Laplacian eigenvalue spectrum, spectral gap, betweenness centrality, and PageRank — to characterise normal network behaviour.
- **Simulating attack scenarios** (DDoS flooding, node compromise, edge rewiring) and measuring their impact on graph-theoretic properties.
- **Applying probabilistic anomaly detection** using Z-score hypothesis testing, KL divergence, Bayesian posterior scoring, and Markov transition matrix comparison.

## Mathematical Background

| Concept | Definition |
|---|---|
| Adjacency matrix **A** | A[i][j] = w\_ij if edge (i, j) ∈ E, else 0 |
| Degree matrix **D** | D[i][i] = Σ\_j A[i][j] |
| Graph Laplacian **L** | L = D − A |
| Normalised Laplacian | L\_norm = D^{−½} L D^{−½} |
| Spectral gap | λ₂ − λ₁ of L; measures algebraic connectivity |
| Markov transition matrix **P** | P[i][j] = A[i][j] / D[i][i] |
| KL divergence | KL(P ‖ Q) = Σ\_x P(x) log(P(x) / Q(x)) |

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── graph_model.py      # IoT network graph generation & matrix algebra
│   ├── metrics.py           # Structural and spectral feature extraction
│   ├── attacks.py           # Attack simulation (DDoS, compromise, rewiring)
│   ├── anomaly.py           # Probabilistic anomaly detection
│   ├── visualization.py     # Matplotlib visualization
│   └── main.py              # Pipeline orchestration & CLI entry point
├── data/                    # Output artifacts (plots, reports)
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/SMQ05/Graph-Based-Probabilistic-Intrusion-Detection-Framework-for-IoT-Networks.git
cd Graph-Based-Probabilistic-Intrusion-Detection-Framework-for-IoT-Networks

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the full pipeline from the repository root:

```bash
# Default: DDoS attack simulation
python -m src.main

# Node compromise attack
python -m src.main compromise

# Edge rewiring attack
python -m src.main rewiring

# All attacks combined
python -m src.main all
```

Output plots and a JSON anomaly report are saved to the `data/` directory.

### Programmatic Usage

```python
from src.main import run_pipeline

summary = run_pipeline(
    num_nodes=150,
    density=0.04,
    seed=42,
    attack_type="ddos",
    output_dir="data",
)

print(summary["anomaly_report"]["bayesian"]["posterior_attack"])
```

## Anomaly Detection Methods

| Method | Level | Description |
|---|---|---|
| Z-score | Node | Flags nodes whose feature vector deviates > threshold σ from baseline |
| KL divergence | Global | Measures distributional shift of degree distributions |
| Bayesian scoring | Global | Posterior P(attack \| evidence) from spectral gap shift |
| Markov transition | Global | Frobenius norm ‖P\_normal − P\_attacked‖\_F |

## Requirements

- Python ≥ 3.9
- NetworkX ≥ 3.1
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- Matplotlib ≥ 3.7

## License

This project is provided for academic and research purposes.
