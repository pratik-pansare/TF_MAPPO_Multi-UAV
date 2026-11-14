# Experimental Matrix & Protocol Specification

This document details the multi-part experimental design executed to validate the cooperative multi-UAV edge computing system.

> **Manuscript Title:** *Joint Task Offloading and Adaptive Trajectory Optimization in Trust- and Fault-Aware Multi-UAV Edge IoT Networks*  
> **Evaluation Protocol:** Multi-part decoupled evaluation separating geometric organization from validation-best operational outcomes.

---

## 1. Experiment Overview

The evaluation suite addresses three fundamental research questions:
1. **Algorithm Superiority & Ablation (Experiment A):** Under identical cluster count $C=4$, does the proposed action-masked CTDE MAPPO outperform policy baselines and heuristic policies?
2. **Cluster Count Selection (Experiments B0 & B1):** How does the spatial cluster count $C$ affect navigation and cooperative MEC performance? Which $C$ is geometrically compact ($C_{\text{geo}}$), and which $C$ minimizes the empirical Problem P1 multi-objective cost ($C_{\text{val}}$)?
3. **Model $\times$ Cluster Interaction (Experiment C):** Does the proposed method consistently outperform baselines across multiple cluster count configurations?

---

## 2. Summary of Experimental Phases

```
+-----------------------------------------------------------------------------------+
| Experiment B0: Pre-Deployment Geometric Screening (Lloyd K-means++, 20 restarts)   |
| Candidates: C in [2 .. 12] on N=80 static IoT nodes                               |
| Objective: J_clu(C), Silhouette S(C), Max Radius r_max(C) <= R_plan = 280.52 m    |
| Outcome: C_geo = 12 (Feasible r_max = 154.2 m, Max Silhouette = 0.427)            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Experiment A: Fair Baseline Comparison at Fixed C=4 (M=4, N=80, K_comm=3)         |
| Algorithms: PROPOSED, MAPPO (No Mask), IPPO, GREEDY (EDF), RANDOM                 |
| Metrics: J_P1, Completion Ratio (%), Latency (s), Energy (kJ), Trajectory (km)    |
| Table: table_algorithm_comparison_C4.csv                                          |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Experiment B1: Proposed Model Cluster-Count Sweep (M=4, N=80)                      |
| Candidates: C in {2, 4, 6, 8, 12, C_geo}                                          |
| Selection Rule: min validation J_P1 subject to hard feasibility invariants         |
| Outcome: Selection of C_val, exported to cluster_selection_decision.json         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Experiment C: Algorithm x Cluster Count Interaction                               |
| Factorial Grid: {PROPOSED, MAPPO, GREEDY} x {C=4, C_geo, C_val}                   |
| Table: table_model_by_C.csv                                                       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Experiments D1 & D2: Fleet & Radio Sensitivity Studies                             |
| D1 (Fleet Scalability): M in [4, 5, 6, 8] with fixed C=4                          |
| D2 (Transceiver Ceiling): K_comm in [1, 2, 3, 4, 5] with fixed C=4, M=4           |
+-----------------------------------------------------------------------------------+
```

---

## 3. Evaluation Metrics and Problem P1 Objective

The primary scalar evaluation criterion is the Problem P1 multi-objective cost:

$$J_{P1} = \omega_E \frac{E_{\text{fleet}}}{E_{\text{ref}}} + \omega_T \frac{T_{\text{avg}}}{T_{\text{ref}}} + \omega_D (1 - \eta^{\text{task}})$$

Where:
- $\omega_E = 0.4$, $\omega_T = 0.4$, $\omega_D = 0.2$ (pre-registered weights).
- $E_{\text{ref}} = 600.0\text{ kJ}$ (reference fleet energy budget over 300 slots).
- $T_{\text{ref}} = 2.0\text{ s}$ (reference single-slot execution latency).
- $\eta^{\text{task}} = \frac{N_{\text{ddl}}}{N_{\text{gen}}}$ (fraction of generated tasks completed on or before deadline).

All comparisons evaluate policies on identical held-out scenarios generated from real non-oracle discrete-event environmental schedules.

---

## 4. Standardized Training Protocol & Convergence Guarantee

To maintain a defensible, rigorous, and computationally reproducible experimental protocol across all comparative baselines:

1. **Default Training Budget**: Exactly **150,000 joint environment training steps** per learning model per seed. All compared learning algorithms (`PROPOSED`, `MAPPO`, `IPPO`, and each cluster-count configuration) share this identical budget.
2. **Rollout Length**: Maintained strictly at **500 steps** per rollout batch.
3. **Independent Validation & Periodic Checkpoints**: Conducted every **25,000 steps** on a disjoint validation seed (zero data leakage).
4. **Checkpoint Selection**: The best checkpoint (`checkpoint_best.pt`) is selected strictly using validation Problem P1 cost subject to physical feasibility constraints, never held-out test performance.
5. **Multi-Stage Execution**:
   - **Stage 1 (Smoke Verification)**: 1,000 steps testing all 5 algorithms and geometric screening.
   - **Stage 2 (Pilot Verification)**: 30,000 steps testing core comparative models before full runs (`--mode pilot`).
   - **Stage 3 (Full Experiments)**: 150,000 steps per model across the full matrix (`--mode full`).
6. **Seed Strategy & Statistical Significance**:
   - **Preliminary Screening**: 3 independent seeds (`[42, 43, 44]`).
   - **Final Reported Comparisons**: 5 independent seeds (`[42, 43, 44, 45, 46]`) reporting mean $\pm$ 95% confidence intervals ($1.96 \cdot \frac{\sigma}{\sqrt{N_{\text{seeds}}}}$).
7. **Convergence & Stability Audit**: The final validation window (last 3 intervals: 100k, 125k, 150k steps) is inspected. If the relative spread exceeds 10%, a uniform extension to **250,000 steps** across the entire comparison group is recommended to preserve experimental parity.
8. **Checkpoint Reuse**: Completed checkpoints are automatically detected and reused across pipeline stages (e.g., `PROPOSED` at $C=4$ is trained once and reused in Experiments A, B1, C, and sensitivity baselines).

