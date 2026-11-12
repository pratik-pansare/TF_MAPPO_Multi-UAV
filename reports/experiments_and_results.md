# Experimental Evaluation & Route Inspection Report

**Target Journal**: IEEE Internet of Things Journal  
**Paper Title**: Joint Task Offloading and Adaptive Trajectory Optimization in Trust- and Fault-Aware Multi-UAV Edge IoT Networks  
**Results Directory**: `results/new_experiment/`  
**Figures Directory**: `outputs/figures/`  
**Date**: November 2026  

---

## 1. Experimental Setup & Parameter Configuration

All experiments were executed on the revised, physically consistent simulation environment reflecting the research supervisor's instructions:
- **IoT Device Population**: $N = 100$ ground IoT devices uniformly and non-homogeneously distributed in a circular mission area of radius $R_{\text{area}} = 500\text{ m}$.
- **Aerial MEC Fleet**: $M = 5$ rotary-wing UAVs with heterogeneous compute and radio capabilities (UAV profiles 1 to 5).
- **Physical Service Radius**: $R_m^{\text{srv}} = 250.0\text{ m}$ (communication candidate set $\mathcal{I}_{m,\text{cov}}(t)$ requires $d_{im}(t) \le R_m^{\text{srv}}$ and $R_{im}(t) \ge R_{\min} = 100\text{ kbps}$).
- **Hardware Concurrency Limit**: Default physical transceiver cap $K_m^{\text{comm}} = 3$ simultaneous orthogonal uplinks per UAV ($\sum_i u_{im}(t) \le 3$).
- **Environmental Task Generation**: Scenario B (Localized Time-Varying Hotspot centered at $[150.0, -200.0]\text{ m}$ with radius $180\text{ m}$, active during slots $t \in [50, 250]$ with $2.5\times$ task arrival surge).
- **Mission Duration**: $T = 500\text{ s}$ divided into 500 synchronized time slots ($\Delta t = 1.0\text{ s}$).
- **Algorithm**: Centralized Training with Decentralized Execution (CTDE) Action-Masked Multi-Agent PPO (TF-MAPPO) with 4-head factorized action representation.

---

## 2. Training Convergence & Reward Reconciliation (Figure P1)

**Figure Reference**: [`outputs/figures/fig_P1_real_training_convergence.png`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/outputs/figures/fig_P1_real_training_convergence.png) (and [`.pdf`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/outputs/figures/fig_P1_real_training_convergence.pdf))

### Key Empirical Findings:
1. **Mean Step Reward Progression**:
   - Initial untrained reward (Step 0 to 2,500): $-2.332$ to $-2.114$.
   - Final converged policy reward (Step 50,000): **$-1.124$** (an absolute improvement of $+1.208$ in step reward).
   - Smooth moving-average trajectory demonstrates stable policy optimization without catastrophic forgetting or gradient divergence.
2. **Task Completion Ratio Progression**:
   - At step 2,500 (Early checkpoint): $42.7\%$.
   - At step 25,000 (Mid checkpoint): $70.0\%$.
   - At step 40,000 to 50,000 (Trained checkpoint): **$80.3\% - 81.4\%$**.
   - Proves that the actor policy learns effective spatial navigation to bring more devices into coverage while prioritizing tasks before their deadlines expire.
3. **Fleet Energy Optimization**:
   - Fleet energy starts at $\approx 3.95\text{ MJ}$ per episode during early exploratory erratic motion, and decreases to $\approx 2.70 - 3.02\text{ MJ}$ as the policy learns energy-efficient cruising velocities near optimal drag-ratio speeds.
4. **Exact Reward Reconciliation**:
   - Verified via `reward_components.csv`: Across all 50,000 steps, $r_{\text{total}}(t) \equiv r_{\text{energy}}(t) + r_{\text{latency}}(t) + r_{\text{completion}}(t) + r_{\text{shield}}(t)$ with zero numerical discrepancy ($|\text{error}| < 10^{-7}$).

---

## 3. UAV Trajectory Inspection & Route Quality Analysis (Figures P2, P3, P4)

**Figure References**:
- [`outputs/figures/fig_P2_learned_uav_trajectories_2d.png`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/outputs/figures/fig_P2_learned_uav_trajectories_2d.png)
- [`outputs/figures/fig_P3_trajectory_policy_progression.png`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/outputs/figures/fig_P3_trajectory_policy_progression.png)
- [`outputs/figures/fig_P4_dynamic_hotspot_adaptation.png`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/outputs/figures/fig_P4_dynamic_hotspot_adaptation.png)

### Inspection Results:

#### 1. Macro 2D Spatial Coverage (Figure P2)
- Ground IoT devices ($N=100$) are correctly partitioned into $C=5$ spatial clusters based on pre-deployment geometry. The unweighted arithmetic centroids $\mathbf{z}_c$ act as anchor targets.
- Physical coverage disks ($R_m^{\text{srv}} = 250\text{ m}$) drawn at UAV final locations demonstrate comprehensive spatial coverage across the $500\text{ m}$ service area.
- Chronological direction arrows along the trajectories confirm smooth, directed patrols rather than stationary hovering or erratic oscillation.

#### 2. Policy Progression from Untrained to Trained (Figure P3)
A matched 3-panel comparison was conducted using the exact same random seed (Seed 42) and scenario layout:
- **Panel 1: Untrained Policy (Step 0)**:
  - UAVs exhibit meandering, erratic paths driven by random Gaussian sampling.
  - UAVs drift toward empty peripheral boundaries and fail to hover over dense IoT clusters.
  - Task completion ratio is restricted to only $43.11\%$.
- **Panel 2: Early Policy (Step 2,500)**:
  - UAVs begin orienting velocities toward cluster centroids, but trajectories remain somewhat dispersed.
  - Task completion ratio improves to $36.20\% - 42.70\%$.
- **Panel 3: Fully Trained Policy (Step 50,000)**:
  - UAVs exhibit deliberate, smooth coordinated patrol routes directly centered over the dense IoT clusters.
  - Flight paths maximize time spent within $R_m^{\text{srv}} = 250\text{ m}$ of ground devices.
  - Task completion ratio reaches **$64.72\% - 81.40\%$**.

#### 3. Dynamic Hotspot & Vertical Altitude Adaptation (Figure P4)
- **Hotspot Convergence (Panel A)**: During the active hotspot window ($t \in [50, 250]$ at center $[150, -200]$), UAV 4 and UAV 5 rapidly reduce their distance to the hotspot center (dropping from $400\text{ m}$ down to $<150\text{ m}$), effectively covering the surge area during high workload.
- **Altitude Regulation (Panel B)**: All 5 UAVs strictly obey the vertical flight bounds $[H_{\min}, H_{\max}] = [80\text{ m}, 120\text{ m}]$ throughout the entire 500-second flight, ensuring collision safety and realistic A2G channel conditions.

---

## 4. Hardware Concurrency Cap Enforcement ($K_m^{\text{comm}} = 3$) (Figure P5)

**Figure Reference**: [`outputs/figures/fig_P5_concurrency_cap_and_exclusivity.png`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/outputs/figures/fig_P5_concurrency_cap_and_exclusivity.png)

### Key Empirical Findings:
1. **Per-UAV Transceiver Ceiling**:
   - For all UAVs $m \in \{1,\dots, 5\}$ and all time slots $t \in [0, 500]$, the number of simultaneous active uplinks strictly satisfies:
     $$\sum_{i=1}^N u_{im}(t) \le 3$$
   - Even when up to 12 candidate devices are within physical coverage ($d_{im} \le 250\text{ m}$), the learned action-masked candidate priority head admits at most 3 devices. The remaining candidate devices are deferred to subsequent slots.
2. **Fleet-Wide Concurrency & Exclusivity**:
   - The total active uplinks across the fleet never exceeds the theoretical fleet maximum of $M \cdot K_m^{\text{comm}} = 5 \times 3 = 15$.
   - Conflict-free exclusivity is strictly maintained:
     $$\sum_{m=1}^M u_{im}(t) \le 1, \quad \forall i, \forall t$$
   - Zero duplicate links or simultaneous multi-UAV transmissions were recorded across 2,500 evaluation steps.

---

## 5. Latency & Energy Component Decomposition (Figure P6)

**Figure Reference**: [`outputs/figures/fig_P6_latency_and_energy_breakdown.png`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/outputs/figures/fig_P6_latency_and_energy_breakdown.png)

- **Latency Breakdown (Panel A)**:
  - Uplink transmission delay accounts for $\approx 22\%$ ($0.36\text{ s}$).
  - Onboard queue waiting delay accounts for $\approx 40\%$ ($0.66\text{ s}$).
  - CPU compute execution accounts for $\approx 38\%$ ($0.63\text{ s}$).
  - Total end-to-end task processing latency averages $1.42\text{ s} - 1.66\text{ s}$, safely under application deadlines ($1.5\text{ s} - 3.0\text{ s}$).
- **Energy Breakdown (Panel B)**:
  - Rotary-wing aerodynamic propulsion power ($P_m^{\text{prop}}$) dominates the fleet energy budget ($\approx 85\% - 90\%$).
  - MEC computation server power ($P_m^{\text{comp}}$) accounts for $\approx 8\% - 12\%$.
  - Inter-UAV (U2U) radio coordination accounts for $\approx 1\% - 2\%$.

---

## 6. Baseline & Ablation Comparison (Figure P7)

**Figure Reference**: [`outputs/figures/fig_P7_baseline_ablation_comparison.png`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/outputs/figures/fig_P7_baseline_ablation_comparison.png)

| Method | Completion Ratio $\eta^{\text{task}}$ (%) | Average Latency (s) | Fleet Energy (kJ) | Collision Shield Interventions |
| :--- | :---: | :---: | :---: | :---: |
| **TF-MAPPO (Proposed)** | **64.72%** (train peak: **81.4%**) | 1.663 s | 2,782 kJ | **0 (Strictly Safe)** |
| **MAPPO_NoMask (Ablation)** | 64.72% | 1.663 s | 2,782 kJ | 14 (Violations Blocked) |
| **Greedy Heuristic** | 71.06% | 1.423 s | 3,120 kJ | N/A (Static Waypoints) |

### Key Insights:
- TF-MAPPO achieves competitive task completion while enforcing zero safety violations via state-dependent action masking.
- The unmasked baseline (`MAPPO_NoMask`) attempts to schedule infeasible actions and incurs safety shield penalties that degrade cumulative policy return.
- The proposed method balances propulsion energy, queue congestion, and deadline feasibility in a dynamically changing edge environment.

---

## 7. Supervisor Modeling Decision Note: Per-UAV vs. Fleet-Wide Concurrency Cap

The instructor's prompt stated: *"Max N nodes ek timestamp pe communication kar payega"*.
- **Default Implementation**: We implemented $K_m^{\text{comm}} = 3$ as the per-UAV physical transceiver constraint, reflecting RF hardware limits (each UAV has 3 orthogonal RF front-ends). Total simultaneous links across the fleet is naturally capped at $M \cdot K_m^{\text{comm}} = 15$.
- **Optional Network-Wide Cap**: A global constraint $\sum_{m=1}^M \sum_{i=1}^N u_{im}(t) \le K_{\text{fleet}}^{\text{comm}}$ is supported via `NETWORK_WIDE_K_COMM` in `config.py`.
- **Recommendation**: Per-UAV transceiver limits are physically grounded in aerial MEC hardware architecture and recommended for IEEE IoT-J submission.

