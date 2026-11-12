# Manuscript Equation-to-Code Alignment Map

This document cross-references every primary mathematical formulation and constraint from the authoritative paper (*Joint Task Offloading and Adaptive Trajectory Optimization in Trust- and Fault-Aware Multi-UAV Edge IoT Networks*, *IEEE Internet of Things Journal*) with its exact implementation in the codebase.

---

## 1. System Model and Physical Formulation Mapping

| Paper Element | Mathematical Definition | Code Implementation File | Exact Function / Class |
| :--- | :--- | :--- | :--- |
| **Spatial Clustering** | $J_{\text{clu}} = \sum_{c=1}^C \sum_{i \in \mathcal{I}_c} \|p_i - z_c\|^2$ | [`src/clustering.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/clustering.py) | `SpatialKMeans.fit()`, `GeometricClusterSelector` |
| **Planning Radius** | $R_{\text{plan}} = \sqrt{d_{\text{feas}}^2 - H_{\text{ref}}^2}$ | [`src/clustering.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/clustering.py) | `compute_r_plan_numerical()` |
| **UAV Kinematics** | $p_m(t+1) = p_m(t) + v_m(t)\Delta_t + \frac{1}{2}a_m(t)\Delta_t^2$ | [`src/env.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/env.py) | `MultiUAVMECEnv._update_uav_positions()` |
| **Propulsion Power** | $P_m^{\text{prop}}(v_m) = P_0(1 + \frac{3v_m^2}{U_{\text{tip}}^2}) + P_i(\sqrt{1+\frac{v_m^4}{4v_0^4}} - \frac{v_m^2}{2v_0^2})^{1/2} + \frac{1}{2}d_0\rho s A v_m^3$ | [`src/computation.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/computation.py) | `UAVComputationModel.compute_propulsion_power()` |
| **Channel Gain & LoS** | $P_{\text{LoS}} = \frac{1}{1 + a \exp(-b(\theta - a))}$, $g_{im} = K_0^{-1} d_{im}^{-\alpha_{\text{env}}}$ | [`src/channel.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/channel.py) | `WirelessChannel.get_channel_gain()` |
| **Uplink Data Rate** | $R_{im}(t) = B_{im}(t) \log_2(1 + \frac{P_i^{\text{tx}} g_{im}(t)}{\sigma^2 + I_{\text{ext}}})$ | [`src/channel.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/channel.py) | `WirelessChannel.compute_rate()` |
| **Transceiver Cap** | $\sum_{i=1}^N u_{im}(t) \le K_m^{\text{comm}}$ | [`src/env.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/env.py) | `MultiUAVMECEnv.step()` & `_resolve_offloading_requests()` |
| **Link Exclusivity** | $\sum_{m=1}^M u_{im}(t) \le 1, \quad \forall i$ | [`src/env.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/env.py) | `MultiUAVMECEnv._resolve_offloading_requests()` |
| **Fault & Trust Availability** | $e_m(t) = \tau_m(t) \cdot \phi_m(t)$ | [`src/action_mask.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/action_mask.py) | `ActionMaskManager.compute_action_mask()` |
| **CPU Persistent Queue** | $Q_m(t+1) = \max(0, Q_m(t) + \sum_i u_{im}(t) D_i C_i - f_m^{\text{uav}} \Delta_t)$ | [`src/computation.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/computation.py) | `UAVComputationModel.process_cpu_queue()` |
| **Multi-Objective Cost** | $J_{P1} = \omega_E \frac{E_{\text{fleet}}}{E_{\text{ref}}} + \omega_T \frac{T_{\text{avg}}}{T_{\text{ref}}} + \omega_D (1 - \eta^{\text{task}})$ | [`run_experiments.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/run_experiments.py) | `compute_p1_cost()` |
| **Hybrid Action Masking** | $\mathcal{M}_m(t) = \mathcal{M}_m^{\text{clu}}(t) \times \mathcal{M}_m^{\text{cand}}(t)$ | [`src/actor.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/actor.py) | `HybridActionMaskedActor.get_action_and_log_prob()` |
| **CTDE Centralized Critic** | $\mathcal{L}_V(\phi) = \frac{1}{B}\sum [V_\phi(s) - \hat{R}]^2$ | [`src/computation.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/computation.py), [`src/mappo_agent.py`](file:///Users/pratik/Desktop/TF_MAPPO_Multi-UAV/src/mappo_agent.py) | `CentralizedCritic`, `MAPPOAgent.update()` |

---

## 2. Invariant Verification

All constraints are enforced both in training rollouts and during held-out evaluations through:
1. **Pre-Action Feasibility Masking:** Prevents actors from taking actions that violate coverage, link caps, or availability.
2. **Physical Environment Validation:** Rejects any residual out-of-range or duplicate link assignments, ensuring that safety invariants are maintained.

