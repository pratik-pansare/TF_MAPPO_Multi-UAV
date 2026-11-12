# Method Definitions and Fair Comparison Invariants

This document establishes the formal definitions, architectural boundaries, and common invariants across all evaluated algorithms in the study:

> **Authoritative Topic:** *Joint Task Offloading and Adaptive Trajectory Optimization in Trust- and Fault-Aware Multi-UAV Edge IoT Networks*  
> **Target Journal:** *IEEE Internet of Things Journal*

---

## 1. Table G1: Formal Matrix of Evaluated Methods

| Method ID | Actor Architecture | Critic Architecture | Action Masking | Information Access | Execution Paradigm |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`PROPOSED`** | Hybrid Continuous–Discrete Actor with Dynamic Cluster Head ($C+1$) | Centralized Critic $V(s)$ over Global State | **Full Feasibility Masking** (Coverage, $\tau_m$, $\phi_m$, $K_m^{\text{comm}}$, battery) | Local Obs $o_m(t)$ at inference; Global State $s(t)$ during training | **CTDE** (Centralized Training, Decentralized Execution) |
| **`MAPPO`** | Hybrid Continuous–Discrete Actor (identical to Proposed) | Centralized Critic $V(s)$ over Global State | **None** ($\mathbf{M}_{m} = \mathbf{1}$) | Local Obs $o_m(t)$ at inference; Global State $s(t)$ during training | **CTDE** (Unmasked Ablation) |
| **`IPPO`** | Hybrid Continuous–Discrete Actor (identical to Proposed) | **Local Critic** $V(o_m)$ over Local Observation only | **Full Feasibility Masking** | Local Obs $o_m(t)$ during both training and execution | **DTDE** (Decentralized Training, Decentralized Execution) |
| **`GREEDY`** | Heuristic Policy: Nearest-eligible ground device with Earliest Deadline First (EDF) | None (Non-learning heuristic) | Implicit Feasibility Filter | Local positions and task descriptors | Decentralized Heuristic |
| **`RANDOM`** | Uniformly random sampling over feasible masked action space | None (Non-learning baseline) | **Full Feasibility Masking** | Action feasibility mask $\mathbf{M}_m(t)$ | Decentralized Random Baseline |

---

## 2. Common Scientific Invariants

To ensure strictly fair and unbiased experimental comparisons (zero deliberate crippling of baselines, zero oracle workload forecasting), all 5 algorithms share identical runtime components:

1. **Common Physical Environment (`MultiUAVMECEnv`):**
   - Ground IoT positions $p_i \in \mathbb{R}^2$ ($N=80$ devices within $R=500\text{ m}$).
   - 3D UAV kinematics, rotary-wing propulsion power consumption $P_m^{\text{prop}}(v_m)$.
   - Probabilistic LoS/NLoS urban channel propagation and Shannon transmission rate $R_{im}(t)$.
   - Discrete-event non-oracle application workload generation.
2. **Common Concurrency and Hardware Constraints:**
   - Simultaneous active radio link limit $K_m^{\text{comm}} = 3$ uplinks per UAV ($\sum_i u_{im}(t) \le K_m^{\text{comm}}$).
   - Link exclusivity constraint: $\sum_m u_{im}(t) \le 1$ per device.
   - Total UAV MEC radio bandwidth limit $B_m^{\text{total}} = 5\text{ MHz}$.
   - UAV onboard CPU queue persistent service across 2.0s slot boundaries.
3. **Common Safety & Physical Execution Layer:**
   - Out-of-bounds containment and collision avoidance safety shield.
   - Real-time battery drain accounting and operational availability $\phi_m(t)$.
   - Zero completed-task counting for expired deadlines.
4. **Matched Evaluation Scenarios & Seeds:**
   - Exact identical random seeds and environmental event schedules are evaluated across all methods.

---

## 3. Scientific Isolation and Ablation Value

- **Contribution of Action Masking (`PROPOSED` vs `MAPPO`):**  
  Both algorithms share the identical CTDE hybrid actor-critic architecture and hyperparameter configurations. The only difference is that `MAPPO` operates without action masking ($\mathbf{M}_m(t) = \mathbf{1}$). Comparing `PROPOSED` against `MAPPO` directly isolates the quantitative contribution of action masking in preventing infeasible offloading, eliminating illegal UAV cluster jumps, and reducing wasted radio/propulsion energy.
- **Contribution of CTDE Centralized Critic (`PROPOSED` vs `IPPO`):**  
  Both algorithms share the identical action-masked hybrid actor and feasibility filters. However, `IPPO` trains local value functions $V(o_m)$ rather than a centralized value function $V(s)$. This isolates the cooperative multi-agent coordination benefit of Centralized Training with Decentralized Execution.

