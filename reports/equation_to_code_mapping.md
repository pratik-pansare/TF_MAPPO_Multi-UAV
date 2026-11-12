# Mathematical Equation to Implementation Mapping Report

**Document Version**: 2.0 (Post-Supervisor Revisions)  
**Date**: November 2026  
**Target Venue**: IEEE Internet of Things Journal  
**Paper Title**: Joint Task Offloading and Adaptive Trajectory Optimization in Trust- and Fault-Aware Multi-UAV Edge IoT Networks  

---

## 1. Overview & Verification Methodology

This document maps every core mathematical formula and constraint from the theoretical manuscript directly to its concrete source implementation in this repository. All mathematical formulations have been audited to guarantee:
1. **Zero clairvoyance / oracle leakage**: Online communication decisions depend solely on current physical state and channel conditions.
2. **Strict physical constraints**: Concurrency caps ($K_m^{\text{comm}}$), transmission energy, LoS/NLoS path-loss, and collision safety shields are strictly enforced.
3. **Reproducibility**: All random streams are causally seeded, with deterministic fallback modes for unit verification.

---

## 2. Comprehensive Equation-to-Code Mapping Table

| Equation No. | Mathematical Concept | Symbolic Formulation | Implementation File | Primary Function / Class | Line Range |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Eq. (1)** | 3D UAV Kinematics & Velocity Updates | $\mathbf{q}_m(t+\Delta t) = \mathbf{q}_m(t) + \mathbf{v}_m(t)\Delta t$ | [`src/env.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/env.py) | `MultiUAVMECEnv.step` | L340–L355 |
| **Eq. (2)** | Continuous Inter-UAV Separation Constraint | $\|\mathbf{q}_m(t) - \mathbf{q}_j(t)\| \ge d_{\min}, \; \forall m \neq j$ | [`src/action_mask.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/action_mask.py) | `JointSafetyShield.check_continuous_collision` | L140–L175 |
| **Eq. (3)** | Mission Service Area & Altitude Boundaries | $\|\mathbf{p}_{m,2\text{D}}\| \le R_{\text{area}}, \; H_{\min} \le z_m \le H_{\max}$ | [`src/action_mask.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/action_mask.py) | `JointSafetyShield.project_safe_movement` | L180–L210 |
| **Eq. (4)** | 3D Air-to-Ground Euclidean Distance | $d_{im}(t) = \sqrt{\|\mathbf{p}_i - \mathbf{q}_{m,2\text{D}}(t)\|^2 + z_m(t)^2}$ | [`src/channel.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/channel.py) | `ChannelModel.compute_distance_and_angle` | L45–L55 |
| **Eq. (5)** | Elevation Angle Calculation | $\theta_{im}(t) = \frac{180}{\pi} \arcsin\left(\frac{z_m(t)}{d_{im}(t)}\right)$ | [`src/channel.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/channel.py) | `ChannelModel.compute_distance_and_angle` | L52–L58 |
| **Eq. (6)** | Line-of-Sight (LoS) Probability | $P_{\text{LoS}}(\theta) = \frac{1}{1 + a \exp(-b(\theta - a))}$ | [`src/channel.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/channel.py) | `ChannelModel.compute_los_probability` | L60–L72 |
| **Eq. (7)** | Air-to-Ground Probabilistic Path Loss | $\text{PL}_{im}(t) = P_{\text{LoS}} \text{PL}_{\text{LoS}} + (1 - P_{\text{LoS}}) \text{PL}_{\text{NLoS}}$ | [`src/channel.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/channel.py) | `ChannelModel.compute_path_loss` | L75–L90 |
| **Eq. (8)** | Channel Power Gain | $g_{im}(t) = 10^{-\text{PL}_{im}(t) / 10}$ | [`src/channel.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/channel.py) | `ChannelModel.compute_channel_gain` | L92–L102 |
| **Eq. (9)** | Orthogonal Bandwidth Allocation | $B_{im}(t) = \frac{B_m^{\text{total}}}{\sum_{j=1}^N u_{jm}(t)}$ | [`src/env.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/env.py) | `MultiUAVMECEnv.step` | L410–L425 |
| **Eq. (10)** | Shannon Achievable Uplink Transmission Rate | $R_{im}(t) = B_{im}(t) \log_2\left(1 + \frac{p_i g_{im}(t)}{\sigma^2 B_{im}(t)}\right)$ | [`src/channel.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/channel.py) | `ChannelModel.compute_achievable_rate` | L105–L125 |
| **Eq. (11)** | Task Computational Workload | $W_i(t) = D_i(t) \cdot C_i(t) \quad [\text{cycles}]$ | [`src/task.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/task.py) | `IoTTask.__init__` | L35–L50 |
| **Eq. (12)** | Multi-Slot Remaining Bits Causal Tracking | $D_i^{\text{rem}}(t+\Delta t) = \max(0, D_i^{\text{rem}}(t) - R_{im}(t)\Delta t)$ | [`src/task.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/task.py) | `IoTTask.record_uplink_progress` | L70–L95 |
| **Eq. (13)** | Local Computing Energy at IoT Node | $E_i^{\text{loc}}(t) = \kappa_i (f_i^{\text{loc}})^2 W_i(t)$ | [`src/computation.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/computation.py) | `ComputationModel.compute_local_energy` | L35–L48 |
| **Eq. (14)** | UAV MEC Computation Execution Time | $T_{im}^{\text{comp}}(t) = \frac{W_i(t)}{f_{im}^{\text{cpu}}(t)}$ | [`src/computation.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/computation.py) | `ComputationModel.compute_execution_time` | L55–L70 |
| **Eq. (15)** | Rotary-Wing UAV Propulsion Power | $P_m^{\text{prop}}(v) = P_0 (1 + \frac{3v^2}{U_{\text{tip}}^2}) + P_i (\sqrt{1 + \frac{v^4}{4v_0^4}} - \frac{v^2}{2v_0^2})^{1/2} + \frac{1}{2} d_0 \rho s A v^3$ | [`src/env.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/env.py) | `MultiUAVMECEnv._calculate_propulsion_power` | L450–L465 |
| **Eq. (16)** | Fleet Total Energy Consumption | $E_{\text{fleet}}(t) = \sum_{m} (P_m^{\text{prop}} + P_m^{\text{comp}} + P_m^{\text{u2u}})\Delta t$ | [`src/env.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/env.py) | `MultiUAVMECEnv.step` | L460–L475 |
| **Eq. (17)** | Certificate-Based Trust Eligibility | $\tau_m(t) \in \{0, 1\}$ | [`src/trust.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/trust.py) | `TrustManager.is_eligible` | L40–L65 |
| **Eq. (18)** | Hardware Fault Status Indicator | $\phi_m(t) \in \{0, 1\}$ | [`src/fault.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/fault.py) | `FaultManager.is_operational` | L35–L60 |
| **Eq. (19)** | Joint UAV Operational Readiness | $e_m(t) = \tau_m(t) \cdot \phi_m(t)$ | [`src/action_mask.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/action_mask.py) | `ActionMaskGenerator.compute_cluster_mask` | L50–L75 |
| **Eq. (20)** | Physical Communication Candidate Link Set | $\mathcal{I}_{m,\text{cov}}(t) = \{i : d_{im}(t) \le R_m^{\text{srv}} \land R_{im}(t) \ge R_i^{\min} \land e_m(t) = 1\}$ | [`src/channel.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/channel.py) | `ChannelModel.evaluate_link_feasibility` | L130–L155 |
| **Eq. (21)** | Hardware Transceiver Concurrency Limit | $\sum_{i=1}^N u_{im}(t) \le K_m^{\text{comm}}, \quad \forall m$ | [`src/action_mask.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/action_mask.py) | `JointSafetyShield.resolve_link_conflicts` | L110–L135 |
| **Eq. (22)** | Conflict-Free Link Exclusivity | $\sum_{m=1}^M u_{im}(t) \le 1, \quad \forall i$ | [`src/action_mask.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/action_mask.py) | `JointSafetyShield.resolve_link_conflicts` | L115–L130 |
| **Eq. (23)** | Discrete-Event Onboard Queue Dynamics | $Q_m(t+\Delta t) = \max\left(0, Q_m(t) + \sum_{i} u_{im}(t) W_i(t) - f_m^{\text{cpu}}\Delta t\right)$ | [`src/computation.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/computation.py) | `MECComputationServer.step_service` | L120–L160 |
| **Eq. (24)** | End-to-End Task Completion Latency | $T_i^{\text{total}} = T_i^{\text{trans}} + T_i^{\text{queue}} + T_i^{\text{comp}}$ | [`src/task.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/task.py) | `IoTTask.total_latency` | L100–L115 |
| **Eq. (25)** | Task Deadline Feasibility Condition | $T_i^{\text{total}} \le \delta_i(t)$ | [`src/task.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/task.py) | `IoTTask.check_deadline` | L120–L135 |
| **Eq. (26)** | Fleet Task Completion Ratio | $\eta^{\text{task}} = \frac{N_{\text{completed}}}{N_{\text{generated}}} \times 100\%$ | [`src/metrics.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/metrics.py) | `MetricTracker.get_episode_summary` | L110–L125 |
| **Eq. (27)** | Spatial K-Means Clustering Objective | $J_{\text{clu}} = \sum_{c=1}^C \sum_{i \in \mathcal{I}_c} \|\mathbf{p}_i - \mathbf{z}_c\|^2$ | [`src/clustering.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/clustering.py) | `SpatialClusterManager.fit_clusters` | L50–L110 |
| **Eq. (28)** | Macro Spatial Target Assignment Head | $a_m^{\text{clu}}(t) \in \{0, 1, \dots, C\}$ | [`src/actor.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/actor.py) | `HybridActionMaskedActor.forward_heads` | L95–L135 |
| **Eq. (29)** | Problem P1 Multi-Objective Reward | $r(t) = -\omega_E \frac{E_{\text{fleet}}}{E_{\text{ref}}} - \omega_T \frac{\bar{T}}{T_{\text{ref}}} + 0.8 N_{\text{comp}} - 0.2 N_{\text{exp}} - 0.05 N_{\text{unserved}} + r_{\text{shield}}$ | [`src/env.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/env.py) | `MultiUAVMECEnv.step` | L480–L500 |
| **Eq. (30)** | Centralized Critic Value Loss with Value Clipping | $L(\theta_c) = \frac{1}{B} \sum \max((V_{\theta_c}(s) - R_t)^2, (V_{\text{clip}} - R_t)^2)$ | [`src/mappo_agent.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/mappo_agent.py) | `MAPPOAgent.update` | L208–L220 |
| **Eq. (31)** | PPO Clipped Surrogate Actor Loss with Masking | $L(\theta_a) = -\hat{\mathbb{E}}_t [\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)] - c_2 \mathcal{H}(\pi_\theta)$ | [`src/mappo_agent.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/mappo_agent.py) | `MAPPOAgent.update` | L200–L208 |
| **Eq. (32)** | Continuous-Time Flight Safety Shield Condition | $\min_{s \in [0, \Delta t]} \|(\mathbf{q}_m + \mathbf{v}_m s) - (\mathbf{q}_j + \mathbf{v}_j s)\| \ge d_{\min}$ | [`src/action_mask.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/action_mask.py) | `JointSafetyShield.check_continuous_collision` | L145–L170 |

---

## 3. Physical Consistency Guarantees

1. **Dimensional Analysis**:
   - Workload $W_i$: bits $\times$ (cycles/bit) = cycles.
   - CPU Service: cycles / (cycles/second) = seconds.
   - Uplink Rate: $\text{Hz} \times \log_2(1 + \text{SNR}) = \text{bits/second}$.
   - Energy: Watts $\times$ seconds = Joules.
2. **Exclusivity Enforcement**:
   - For all devices $i \in \{1,\dots, N\}$ and steps $t$, $\sum_{m=1}^M u_{im}(t) \le 1$.
   - For all UAVs $m \in \{1,\dots, M\}$ and steps $t$, $\sum_{i=1}^N u_{im}(t) \le K_m^{\text{comm}}$.
3. **No Clairvoyance**:
   - Environmental transitions in `src/task.py` emit events causally based on local context $z_i(t)$.
   - Observation vectors fed to the actor network contain zero future event markers or global random seeds.

