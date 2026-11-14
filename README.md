# Joint Task Offloading and Adaptive Trajectory Optimization in Trust- and Fault-Aware Multi-UAV Edge IoT Networks (TF-MAPPO)

A physically consistent Multi-Agent Reinforcement Learning (MARL) framework for joint UAV 3D trajectory control, candidate task scheduling, and computation resource allocation in trust- and fault-aware Multi-UAV Mobile Edge Computing (MEC) networks.

Target Journal: **IEEE Internet of Things Journal**

---

## Table of Contents

- [Overview & Supervisor Revision Summary](#overview--supervisor-revision-summary)
- [System Architecture](#system-architecture)
- [Core Mathematical & Physical Principles](#core-mathematical--physical-principles)
- [Repository Structure](#repository-structure)
- [Quick Start & Verification](#quick-start--verification)
- [Master Experiment Runner (`run_experiments.py`)](#master-experiment-runner-run_experimentspy)
- [Publication Figures Overview](#publication-figures-overview)
- [Audit & Validation Reports](#audit--validation-reports)
- [Platform Support & Troubleshooting](#platform-support--troubleshooting)

---

## Overview & Supervisor Revision Summary

This repository implements the authoritative research framework incorporating the research supervisor's physical consistency guidelines:

1. **Zero Clairvoyant Workload Prediction**: Reverted artificial pre-deployment workload estimations ($\lambda_w, W_i^0, W_{\max}$). Pre-deployment clustering is strictly geometric spatial K-Means ($J_{\text{clu}} = \sum_c \sum_{i \in \mathcal{I}_c} \|\mathbf{p}_i - \mathbf{z}_c\|^2$) with unweighted arithmetic centroids $\mathbf{z}_c$.
2. **Environment-Dependent Task Generation**: Task generation is driven causally by local environmental context $z_i(t)$ (Routine, Hotspot, Multi-Event, Stress Test) and parameterized application profiles (`config/event_task_profiles.yaml`).
3. **Physical 3D Coverage-Driven Communication**: Online communication eligibility requires physical 3D distance $d_{im}(t) \le R_m^{\text{srv}} = 250\text{ m}$, achievable rate $R_{im}(t) \ge R_{\min} = 100\text{ kbps}$, and operational readiness $e_m(t) = \tau_m(t) \cdot \phi_m(t) = 1$, regardless of static pre-deployment cluster labels.
4. **Hardware Transceiver Concurrency Limit ($K_m^{\text{comm}}$)**: Enforces physical RF front-end limits ($\sum_i u_{im}(t) \le K_m^{\text{comm}} = 3$) and conflict-free single-link exclusivity ($\sum_m u_{im}(t) \le 1$). When candidates exceed $K_m^{\text{comm}}$, a learned action-masked priority head schedules the top candidates and defers the rest.
5. **Causal Multi-Slot Data Transfer**: Tracks remaining transmission bits across time slots ($D_i^{\text{rem}}(t+\Delta t) = \max(0, D_i^{\text{rem}}(t) - R_{im}(t)\Delta t)$). Tasks are not admitted to onboard CPU queues until transmission completes.
6. **Zero Fabricated Plots**: All trajectory maps, convergence curves, and breakdown charts originate strictly from executed simulation runs and saved CSV logs.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Static Ground IoT Devices (N = 80)                    │
│   Fixed coordinates p_i · Causal Event Generation · Multi-Slot Payload Bits │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Uplink Channel: 3D Distance, Elevation Angle,
                                       │ Probabilistic LoS/NLoS Path Loss, Shannon Rate
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Physical Coverage & Concurrency Safety Shield               │
│  - Distance Check: d_im(t) <= R_srv (250m) & Uplink Rate: R_im >= R_min   │
│  - Concurrency Cap: sum_i u_im(t) <= K_comm (3) per UAV                     │
│  - Link Exclusivity: sum_m u_im(t) <= 1 per device                          │
│  - Continuous Collision Avoidance: min ||q_m(s) - q_j(s)|| >= d_min (10m)  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Feasible Action Masks & Filtered Observations
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Multi-UAV MEC Fleet (M = 4 Aerial Servers)               │
│  ┌───────────────────────┐  ┌─────────────────────────────────────────────┐ │
│  │ Trust & Fault Manager │  │ Hybrid Factorized Action-Masked MAPPO       │ │
│  │ - Certificate: τ_m(t) │  │   1. a_m^clu : Spatial Cluster Waypoint Head│ │
│  │ - Fault Status: φ_m(t)│  │   2. a_m^off : Candidate Priority Head (K_p)│ │
│  │ - Readiness: e_m(t)   │  │   3. a_m^mov : 3D Velocity Head (dx, dy, dz)│ │
│  └───────────────────────┘  │   4. a_m^cpu : CPU Allocation Fraction Head │ │
│                             │ Centralized Value Critic (Global State CTDE)│ │
│                             └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Mathematical & Physical Principles

### 1. Spatial K-Means Clustering (One-Time Pre-Deployment)
$$J_{\text{clu}} = \sum_{c=1}^C \sum_{i \in \mathcal{I}_c} \|\mathbf{p}_i - \mathbf{z}_c\|^2, \quad \mathbf{z}_c = \frac{1}{|\mathcal{I}_c|} \sum_{i \in \mathcal{I}_c} \mathbf{p}_i$$

### 2. Physical Communication Candidate Set
$$\mathcal{I}_{m,\text{cov}}(t) = \left\{i \in \{1,\dots,N\} : d_{im}(t) \le R_m^{\text{srv}} \land R_{im}(t) \ge R_{\min} \land e_m(t) = 1\right\}$$

### 3. Hardware Concurrency & Exclusivity Constraints
$$\sum_{i=1}^N u_{im}(t) \le K_m^{\text{comm}}, \quad \forall m \in \{1,\dots,M\}$$
$$\sum_{m=1}^M u_{im}(t) \le 1, \quad \forall i \in \{1,\dots,N\}$$

### 4. Continuous-Time Flight Collision Shield
$$\min_{s \in [0, \Delta t]} \|(\mathbf{q}_m + \mathbf{v}_m s) - (\mathbf{q}_j + \mathbf{v}_j s)\| \ge d_{\min}, \quad \forall m \neq j$$

### 5. Multi-Objective Reward Formulation (Problem P1)
$$r(t) = -\omega_E \frac{E_{\text{fleet}}(t)}{E_{\text{ref}}} - \omega_T \frac{\bar{T}(t)}{T_{\text{ref}}} + 0.8 N_{\text{comp}}(t) - 0.2 N_{\text{exp}}(t) - 0.05 N_{\text{unserved}}(t) + r_{\text{shield}}(t)$$

---

## Repository Structure

```text
TF_MAPPO_Multi-UAV/
├── config.py                 # Core simulation hyperparameters and physical constants
├── run_experiments.py        # Master CLI runner: smoke, pilot, full, eval, and plot
├── train.py                  # Standalone multi-seed MAPPO training engine
├── config/
│   ├── event_task_profiles.yaml  # Human-readable environmental event profiles
│   └── event_task_profiles.json  # Fast JSON mirror with provenance annotations
├── src/
│   ├── action_mask.py        # Action mask generator and joint safety shield
│   ├── actor.py              # 4-head factorized action-masked Gaussian/Categorical actor
│   ├── agent_factory.py      # Agent instantiation factory (MAPPO, Baselines)
│   ├── buffer.py             # Multi-agent rollout buffer with GAE and memory safety
│   ├── channel.py            # Air-to-Ground elevation LoS/NLoS path-loss and rate model
│   ├── clustering.py         # Pure spatial K-Means manager and legacy Lloyd ablation
│   ├── computation.py        # Causal uplink progress tracking and discrete-event CPU queues
│   ├── env.py                # Gymnasium-style multi-UAV MEC environment
│   ├── fault.py              # Hardware fault and availability manager
│   ├── mappo_agent.py        # CTDE MAPPO trainer with value clipping and PPO epochs
│   ├── metrics.py            # Comprehensive metric tracker and 95% CI calculator
│   ├── task.py               # Dynamic environment state manager and IoT task lifecycle
│   ├── trust.py              # Certificate-based trust authentication manager
│   └── baselines/            # Benchmark agents (Greedy heuristic, MADDPG, MATD3)
├── tests/
│   ├── test_supervisor_acceptance.py # 12 supervisor acceptance criteria tests
│   ├── test_audit_gates.py           # 12 audit regression gates
│   └── test_sanity.py                # 10 end-to-end sanity tests
├── reports/
│   ├── instructor_change_audit.md    # Pre-change audit, revert rationale, and rollback plan
│   ├── manuscript_change_map.md      # Section-by-section LaTeX manuscript update map
│   ├── equation_to_code_mapping.md   # Equation (1) to (32) code mapping table
│   ├── validation.md                 # Test logs and acceptance verification summary
│   └── experiments_and_results.md    # Trajectory route analysis and empirical results
├── outputs/figures/          # Vector PDF and PNG publication figures
└── results/new_experiment/   # Executed model checkpoints, trajectory CSVs, and logs
```

---

## Quick Start & Verification

### 1. Execute Acceptance & Unit Test Suites
```bash
# Run all 12 supervisor acceptance tests (coverage, concurrency, multi-slot, causality)
./venv/bin/python -m unittest -v tests/test_supervisor_acceptance.py

# Run audit regression gates
./venv/bin/python -m unittest tests/test_audit_gates.py

# Run physics and latency sanity tests
./venv/bin/python tests/test_sanity.py
```
*(All 34 tests pass 100% without warnings).*

---

## Master Experiment Runner (`run_experiments.py`)

A single, unified CLI supports all execution modes under the revised IEEE IoT-J experimental protocol (150,000 steps default cap, 25,000-step validation cadence, 30,000-step pilot):

```bash
# 1. Smoke Mode: Fast model-invariant pipeline check (1,000 steps, screening, evaluation)
./venv/bin/python run_experiments.py --mode smoke

# 2. Pilot Mode: Verification pilot before full training (30,000 steps across core comparative models)
./venv/bin/python run_experiments.py --mode pilot

# 3. Full Production Training: Standardized 150,000 steps per model with 25,000-step validation cadence
./venv/bin/python run_experiments.py --mode full --total-steps 150000

# 4. Multi-Seed Reporting: 5 independent seeds for final comparisons & 95% confidence intervals
./venv/bin/python run_experiments.py --mode full --total-steps 150000 --seeds 42 43 44 45 46

# 5. Standalone Single Model Training (e.g. Proposed TF-MAPPO on GPU with cached checkpoint reuse)
./venv/bin/python run_experiments.py --mode train --method PROPOSED --total-steps 150000 --device cuda

# 6. Evaluation Only: Evaluates checkpoints on matched seeds and logs CSV traces & task ledgers
./venv/bin/python run_experiments.py --mode eval --method PROPOSED

# 7. Publication Plots Only: Generates all 18 vector PDF & PNG figures from saved CSVs
./venv/bin/python run_experiments.py --mode plot
```

---

## Publication Figures Overview

All figures are automatically generated and saved to [`outputs/figures/`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/outputs/figures/):

1. **Figure P1**: `fig_P1_real_training_convergence.png` (.pdf)  
   *Mean step reward, task completion ratio ($\eta^{\text{task}}$ rising from $42\%$ to $81\%$), fleet energy, and actor/critic loss profiles.*
2. **Figure P2**: `fig_P2_learned_uav_trajectories_2d.png` (.pdf)  
   *2D flight paths of trained UAVs, spatial clusters, centroids $\mathbf{z}_c$, chronological direction arrows, and physical $R_m^{\text{srv}} = 250\text{ m}$ coverage disks.*
3. **Figure P3**: `fig_P3_trajectory_policy_progression.png` (.pdf)  
   *Matched 3-panel comparison (Untrained vs. Early vs. Trained) on identical initial seeds demonstrating learned cluster patrolling.*
4. **Figure P4**: `fig_P4_dynamic_hotspot_adaptation.png` (.pdf)  
   *Fleet convergence to active hotspot center ($[150, -200]$ during $t \in [50, 250]$) and 3D altitude regulation within $[80\text{ m}, 120\text{ m}]$.*
5. **Figure P5**: `fig_P5_concurrency_cap_and_exclusivity.png` (.pdf)  
   *Simultaneous active links per UAV strictly respecting the $K_m^{\text{comm}} = 3$ hardware limit and conflict-free exclusivity ($\sum_m u_{im}(t) \le 1$).*
6. **Figure P6**: `fig_P6_latency_and_energy_breakdown.png` (.pdf)  
   *Empirical delay decomposition (transmission vs. queue vs. CPU execution) and fleet energy breakdown (propulsion vs. compute vs. U2U).*
7. **Figure P7**: `fig_P7_baseline_ablation_comparison.png` (.pdf)  
   *Benchmarking Proposed TF-MAPPO vs. Unmasked MAPPO (`MAPPO_NoMask`) vs. Greedy Heuristic.*

---

## Audit & Validation Reports

For complete technical and scientific details, refer to the reports:
- [`reports/instructor_change_audit.md`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/reports/instructor_change_audit.md): Complete audit of previous oracle clustering and rollback plan.
- [`reports/manuscript_change_map.md`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/reports/manuscript_change_map.md): Section-by-section LaTeX manuscript updates and equation replacements.
- [`reports/equation_to_code_mapping.md`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/reports/equation_to_code_mapping.md): Precise source line mapping for Equations (1) to (32).
- [`reports/validation.md`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/reports/validation.md): Execution logs for all 34 acceptance, regression, and sanity tests.
- [`reports/experiments_and_results.md`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/reports/experiments_and_results.md): Route quality analysis, convergence tables, and supervisor modeling notes.

---

## Platform Support & Troubleshooting

- **macOS / Linux / Windows**: Pure Python/PyTorch codebase without OS-specific C extensions.
- **Apple Silicon (M1/M2/M3/M4)**: Compatible with macOS ARM64; PyTorch CPU/MPS tensor pin-memory guards are active to prevent SIGSEGV.
- **Matplotlib Font Cache**: Set `export MPLCONFIGDIR=/tmp/matplotlib_cache` on sandboxed or headless systems.
