# METRIC DICTIONARY
**Paper Title:** Joint Task Offloading and Adaptive Trajectory Optimization in Trust- and Fault-Aware Multi-UAV Edge IoT Networks  
**Target Journal:** IEEE Internet of Things Journal  

This dictionary defines every recorded variable, equation, unit, and logging frequency across all evaluation phases.

---

## 1. Task Lifecycle & Latency Metrics

| Metric | Symbol | Definition / Equation | Logging Unit | Frequency |
|---|---|---|---|---|
| Generated Tasks | $N_{\text{gen}}$ | Cumulative stochastic tasks generated: $\sum_t \sum_i a_i(t)$ | count | per-episode |
| Admitted Tasks | $N_{\text{adm}}$ | Tasks successfully offloaded to UAV queues | count | per-episode |
| Completed on Time | $N_{\text{ddl}}$ | Tasks completed with $t_{\text{finish}} - t_{\text{gen}} \le \delta_i$ | count | per-episode |
| Expired / Dropped | $N_{\text{exp}}$ | Tasks discarded due to deadline violation or buffer overflow | count | per-episode |
| Deadline Completion Ratio | $\eta^{\text{task}}$ | $N_{\text{ddl}} / N_{\text{gen}} \times 100\%$ | percentage (%) | per-episode |
| End-to-End Latency | $T_{\text{avg}}$ | Mean duration: $\frac{1}{N_{\text{ddl}}} \sum (t_{\text{finish}} - t_{\text{gen}})$ | seconds (s) | per-episode |
| Uplink Delay | $T_{\text{up}}$ | Upload duration: $D_i / R_{i,m}$ | seconds (s) | per-task |
| Queue Delay | $T_{\text{que}}$ | Time spent awaiting execution in UAV onboard queue | seconds (s) | per-task |
| CPU Execution Delay | $T_{\text{exe}}$ | Time required to process task cycles: $W_i / f_{i,m}$ | seconds (s) | per-task |

---

## 2. Energy & Trajectory Metrics

| Metric | Symbol | Definition / Equation | Logging Unit | Frequency |
|---|---|---|---|---|
| Total Fleet Energy | $E_{\text{fleet}}$ | $\sum_m \sum_t (P_m^{\text{fly}} \Delta t + E_{i,m}^{\text{comp}} + E_m^{\text{u2u}})$ | Joules (J) | per-episode |
| Propulsion Energy | $E_{\text{fly}}$ | Cumulative rotary-wing mechanical flight energy | Joules (J) | per-episode |
| Computation Energy | $E_{\text{comp}}$ | Cumulative onboard CPU dynamic switched energy | Joules (J) | per-episode |
| U2U Coordination Energy | $E_{\text{u2u}}$ | Cumulative inter-UAV status signaling energy | Joules (J) | per-episode |
| Flight Distance | $D_{\text{traj}}$ | Total 3D distance traveled by UAV fleet: $\sum_m \sum_t \|q_m(t+1) - q_m(t)\|$ | meters (m) | per-episode |
| Average Velocity | $\bar{v}$ | Mean speed: $D_{\text{traj}} / (M \cdot T_{\text{tot}})$ | m/s | per-episode |

---

## 3. Trust, Fault & Safety Metrics

| Metric | Symbol | Definition / Equation | Logging Unit | Frequency |
|---|---|---|---|---|
| Certificate Trust State | $\tau_m(t)$ | Binary indicator of valid cryptographic certificate | $\{0, 1\}$ | per-slot |
| Operational Availability | $\phi_m(t)$ | Binary indicator of unfailed hardware state | $\{0, 1\}$ | per-slot |
| Combined Eligibility | $e_m(t)$ | $\tau_m(t) \cdot \phi_m(t)$ | $\{0, 1\}$ | per-slot |
| Triggered Failures | $N_{\text{fail}}$ | Count of simulated UAV hardware/battery faults | count | per-episode |
| Recovered Failures | $N_{\text{rec}}$ | Count of restored UAV operational states | count | per-episode |
| Mean Recovery Time | $\bar{T}_{\text{rec}}$ | Average slots required to restore or replace failed UAV | slots (s) | per-episode |
| Shield Interventions | $N_{\text{shield}}$ | Count of continuous trajectory collision overrides | count | per-episode |
| Mask Pruning Fraction | $r_{\text{mask}}$ | Percentage of candidate action space pruned by mask | percentage (%) | per-step |

---

## 4. Multi-Objective Optimization & Reward Components

| Metric | Symbol | Definition / Equation | Logging Unit | Frequency |
|---|---|---|---|---|
| Weighted Objective (P1) | $J_{\text{P1}}$ | $\omega_E \frac{E_{\text{fleet}}}{E_{\text{ref}}} + \omega_T \frac{T_{\text{avg}}}{T_{\text{ref}}}$ | dimensionless | per-episode |
| Step Reward | $r(t)$ | $r_{\text{lat}} + r_{\text{eng}} + r_{\text{comp}} + r_{\text{pen}}$ | dimensionless | per-step |
| Latency Reward Term | $r_{\text{lat}}$ | $-\omega_T (T_{\text{avg}}(t) / T_{\text{ref}})$ | dimensionless | per-step |
| Energy Reward Term | $r_{\text{eng}}$ | $-\omega_E (E_{\text{step}}(t) / E_{\text{step\_ref}})$ | dimensionless | per-step |
| Completion Reward Term | $r_{\text{comp}}$ | $+ \alpha_{\text{comp}} \cdot N_{\text{completed\_step}}$ | dimensionless | per-step |
| Safety Shield Penalty | $r_{\text{pen}}$ | $- \beta_{\text{pen}} \cdot \mathbf{1}[\text{shield intervention}]$ | dimensionless | per-step |

