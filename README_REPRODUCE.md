# Reproduction Guide & Execution Protocols

This repository provides full reproducibility for all results and IEEE publication figures reported in the study:

> **Paper Title:** *Joint Task Offloading and Adaptive Trajectory Optimization in Trust- and Fault-Aware Multi-UAV Edge IoT Networks*  
> **Target Journal:** *IEEE Internet of Things Journal*

---

## 1. System Requirements & Environment Setup

- **Operating System:** macOS (Apple Silicon ARM64) or Linux (x86_64 / ARM64).
- **Python Version:** Python 3.10 to 3.14 supported.
- **Dependencies:** Pure NumPy, PyTorch (>=2.0), Matplotlib (>=3.5), PyYAML. No external proprietary packages or internet connection required during execution.

To set up the environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch numpy matplotlib pyyaml
```

---

## 2. Reproduction Commands

### Stage 1: Smoke Verification Run (< 15 seconds)
Verifies that all 5 algorithms (`PROPOSED`, `MAPPO`, `IPPO`, `GREEDY`, `RANDOM`), action masking, centralized critic CTDE, and evaluation metric trackers execute with zero numerical errors:
```bash
export MPLCONFIGDIR=/tmp/matplotlib_cache
./venv/bin/python run_experiments.py --mode smoke
```

### Stage 2: Full Production Pipeline
Executes the comprehensive multi-part experimental matrix (Experiments B0, A, B1, C, D1, D2) and renders publication Figures `fig01` to `fig18`:
```bash
export MPLCONFIGDIR=/tmp/matplotlib_cache
./venv/bin/python run_experiments.py --mode full --total-steps 30000
```

### Geometric Cluster Screening Only (Experiment B0)
Screens cluster counts $C \in [2 \dots 12]$ using Lloyd K-means++ with 20 restarts, computing inertia $J_{\text{clu}}(C)$, silhouette $S(C)$, and checking $r_{\max}(C) \le R_{\text{plan}}$:
```bash
./venv/bin/python run_experiments.py --mode geometry
```

### Re-rendering Figures from Saved Logs
Generates all IEEE Transactions vector PDF and 300-DPI PNG figures from existing JSON/CSV results:
```bash
./venv/bin/python run_experiments.py --mode plot --results-dir results/new_experiment --figures-dir outputs/paper_figures
```

### Unit Test Verification
Executes the comprehensive 31-test suite validating spatial clustering, non-oracle workloads, physical concurrency, action masking, and supervisor acceptance gates:
```bash
./venv/bin/python -m unittest discover tests
```

---

## 3. Directory Layout

```
TF_MAPPO_Multi-UAV/
├── config/
│   └── reference.yaml             # Single source of truth configuration
├── configs/
│   └── event_task_profiles.yaml   # Non-oracle discrete event triggers
├── src/
│   ├── env.py                     # MultiUAVMECEnv core simulation
│   ├── mappo_agent.py             # Proposed Action-Masked CTDE MAPPO
│   ├── actor.py                   # Hybrid Continuous-Discrete Actor
│   ├── computation.py             # Centralized CTDE Critic & CPU Queues
│   ├── clustering.py              # Pure-NumPy Spatial K-Means++ & C_geo Selector
│   ├── action_mask.py             # Feasibility Masking (Coverage, Tau, Phi, Radio)
│   ├── channel.py                 # 3D Urban LoS/NLoS Channel Model
│   ├── metrics.py                 # Latency, Energy & Problem P1 Tracker
│   └── baselines/                 # Baseline Agents (IPPO, Greedy, Random)
├── experiments/
│   └── run_clustering_selection.py# Standalone Experiment B0 runner
├── outputs/
│   ├── paper_figures/             # Fig 01 to Fig 18 (PDF + 300-DPI PNG)
│   └── clustering/                # Cluster selection metrics & visual maps
├── results/
│   └── new_experiment/            # Evaluation JSONs, CSV trajectories & checkpoints
└── run_experiments.py             # Unified Master CLI Runner
```

