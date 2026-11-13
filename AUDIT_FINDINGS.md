# AUDIT FINDINGS: Comprehensive System, Provenance & Codebase Audit
**Repository:** `pratik-pansare/TF_MAPPO_Multi-UAV`  
**Branch:** `feature-ieee-causal-engine` (derived from commit `07f11cfd1ba50b22b3417bef48d9c2f915ed37f7`)  
**Audit Timestamp:** November 13, 2026  
**Auditor:** Research Software & MARL Reproducibility Engineer (DeepMind / IEEE IoT-J Standards)

---

## 1. Executive Summary & Provenance Baseline

This audit establishes a rigorous, uncompromised baseline for the multi-UAV mobile edge computing (MEC) repository. All legacy experimental data in `outputs/` and `results/` has been preserved in a timestamped, read-only archive at `legacy_results/20261113_pre_causal_archive/`. Every input and output file in the repository has been indexed and hashed via SHA-256 in [`provenance/phase0_inventory.json`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/provenance/phase0_inventory.json) (223 files indexed).

### 1.1 Hardware and Software Environment
- **Host Architecture:** Apple Silicon Mac (Apple M1, 8 cores: 4 performance + 4 efficiency)
- **OS Platform:** macOS Darwin 24.3.0 (arm64)
- **Python Runtime:** Python 3.14.0 (`./venv/bin/python`)
- **Core Scientific Stack:**
  - PyTorch: `2.13.0` (CPU / MPS enabled)
  - NumPy: `2.5.2`
  - Matplotlib: `3.11.1`
- **Configuration & CLI Parameters (`config.py`):**
  - Ground devices: $N = 80$, deployed within radius $R_{\text{area}} = 500\text{ m}$ (or square $[-500, 500]\times [-500, 500]\text{ m}$)
  - UAV MEC Fleet: $M = 4$, altitude range $[80, 120]\text{ m}$ (nominal $100\text{ m}$)
  - Slot Duration: $\Delta t = 2.0\text{ s}$, episode length $T = 300\text{ slots}$ (horizon $600\text{ s}$)
  - Radio Concurrency Cap: $K_m^{\text{comm}} = 3$ concurrent uplinks per UAV
  - Radio Bandwidth: $B_m^{\text{total}} = 10\text{ MHz}$, IoT Tx Power: $20\text{ dBm}$
  - PPO Hyperparameters: Rollout 2048, Epochs 5, Mini-batches 4, Clip $\epsilon = 0.15$, LR $3\times 10^{-4}$

---

## 2. Phase 0 Audit Items & Specific Flaw Identifications

### Item 1: Provenance Divergence (`eval_summary_PROPOSED_C4.json` vs. `cluster_selection_decision.json`)
- **Documented Divergence:**
  - `cluster_selection_decision.json` (Entry "4"): Completion = **70.78%**, Energy = **520.54 kJ**, Tasks Generated = 616, Tasks Completed = 436, $J_{P1} = 0.692$.
  - `eval_summary_PROPOSED_C4.json`: Completion = **79.52%**, Energy = **333.91 kJ**, Tasks Generated = 2466, Tasks Completed = 1961, $J_{P1} = 0.555$.
- **Exact Root Cause:**
  1. `cluster_selection_decision.json` was generated during Step 3 of `run_stage2_full_pipeline` when the model had only been trained for **30,000 steps** (100 episodes).
  2. Subsequently, `experiments/run_extended_convergence.py` trained `PROPOSED_C4` for **900,000 steps** (3,000 episodes) and evaluated it, overwriting `eval_summary_PROPOSED_C4.json`.
  3. `cluster_selection_decision.json` was never re-evaluated or updated with the 900,000-step model, leaving conflicting records on disk.
  4. Crucially, the total number of tasks generated in 300 slots varied wildly from 616 to 2466 because the environment lacked a fixed, policy-independent exogenous task arrival trace.

### Item 2: Unverifiable Task Delays (Missing Raw Event Records)
- Recomputing trajectory flight distance and battery drain from `trajectory_PROPOSED_C4.csv` confirmed that flight metrics match `eval_summary_PROPOSED_C4.json` to machine precision ($|\Delta E| = 1.16\times 10^{-10}\text{ J}$, $|\Delta d| = 7.28\times 10^{-12}\text{ m}$).
- However, **no raw per-task event records** (with exact timestamps for arrival, upload start/end, queue entry, execution start/end, deadline expiry) were exported to disk.
- Consequently, individual completed-task delays and sub-component decompositions ($T_i^{\text{wait}}, T_i^{\text{up}}, T_i^{\text{que}}, T_i^{\text{exe}}$) in legacy JSON files are **unverifiable from disk artifacts alone**.
- **Action:** Phase 1 implements a comprehensive per-task lifecycle event ledger exported to `task_ledger_<TAG>.csv` for every run.

### Item 3: Fabrication of Inference Times in Fig 18
- **Evidence:** In `run_experiments.py` line 1180:
  ```python
  inf_times = [0.65 + 0.02 * c for c in cs_rt]
  ```
- **Severity:** High. The figure drew a synthetic linear interpolation ($0.65 + 0.02 \times C$) rather than measuring real neural network forward-pass elapsed time.
- **Action:** Synthetic formulas are flagged, rejected, and banned. Fig 18 will either be generated from real `time.perf_counter()` measurements across 100 repeats with device synchronization and explicit hardware disclosure, or omitted.

### Item 4: Unequal Training Budgets Presented as Matched Baselines
- **Evidence from `training_progress.csv` in `results/new_experiment/logs/`:**
  - `PROPOSED_C4_M4_seed42`: **900,000 steps** (1,800 updates)
  - `PROPOSED_C2_M4_seed42`: **300,000 steps** (600 updates)
  - `MAPPO_C4_M4_seed42`: **150,000 steps** (300 updates)
  - `IPPO_C4_M4_seed42`: **150,000 steps** (300 updates)
  - `PROPOSED_C6`, `PROPOSED_C8`, `PROPOSED_C12`: **30,000 steps** (60 updates)
  - `MAPPO_C2`, `MAPPO_C12`: **30,000 steps** (60 updates)
  - `PROPOSED_C4_M5`, `PROPOSED_C4_M6`, `PROPOSED_C4_M8`: **30,000 steps** (60 updates)
- **Severity:** High. Presenting convergence curves and model-by-C interactions when C4 had 30x the training budget of C6/C8/C12 violates scientific baseline parity.
- **Action:** Equal-budget protocols must be enforced for all comparative plots, and any unequal pilot sweeps must be explicitly labeled as exploratory.

---

## 3. Prior Architectural & Physical Flaws (To Be Corrected in Phase 1-3)

1. **Non-Causal Task Generation:**
   - Previous `env.step()` generated tasks conditionally on whether `device.current_task is None`. If a device was waiting or uploading, new task arrivals were artificially blocked, distorting Poisson/event arrival physics.
2. **Missing Fly-Then-Hover Structure:**
   - Previous UAV physics allowed instantaneous displacement without computing the physical flight transit time $t_{\text{fly}} = \|\Delta q\| / V_{\text{cr}}$.
   - Uploads were allowed to proceed over the entire slot $\Delta t$, ignoring that a flying UAV cannot maintain stable hovering links during transit.
3. **Queue Entry Boundary Violation:**
   - Tasks completing upload in slot $t$ were immediately allowed to enter the CPU queue and execute in the same slot $t$, violating physical slot boundaries.
4. **Scalar CPU Allocator:**
   - Onboard CPU allocation was modeled as a single scalar fraction $f_m \in [0, 1]$ per UAV rather than allocating individual CPU resources $f_{j,m}$ to discrete resident jobs.
5. **P1 Objective Misalignment:**
   - Legacy code computed $J_{P1}$ using the average completed-task latency rather than $T_{\text{svc}}$ over ALL generated tasks (including deadline miss penalties $T_{\text{miss}}$ for uncompleted tasks).
6. **Strict Cluster Exclusivity:**
   - Default `strict_exclusivity=True` barred multiple UAVs from selecting the same cluster region during emergency or hotspot events.

---

## 4. Archival and Provenance Verification Table

| Archive Path | Description | Access Status |
| :--- | :--- | :--- |
| `legacy_results/20261113_pre_causal_archive/outputs/` | Legacy figures, clustering metrics, preliminary PNG/PDFs | **Read-Only** (`chmod a-w`) |
| `legacy_results/20261113_pre_causal_archive/results/` | Legacy checkpoints, JSON eval summaries, flight trajectories | **Read-Only** (`chmod a-w`) |
| `provenance/phase0_inventory.json` | Complete SHA-256 manifest of 223 indexed repository files | Active Baseline |

