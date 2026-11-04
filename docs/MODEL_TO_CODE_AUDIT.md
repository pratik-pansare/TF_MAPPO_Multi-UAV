# MODEL TO CODE AUDIT
**Paper Title:** Joint Task Offloading and Adaptive Trajectory Optimization in Trust- and Fault-Aware Multi-UAV Edge IoT Networks  
**Target Journal:** IEEE Internet of Things Journal  
**Audit Date:** November 2026  
**Auditor:** Senior Research Engineering Agent (Antigravity AI)

---

## 1. Executive Summary

This document performs an exhaustive audit comparing the mathematical formulations, physical models, assumptions, and constraints from the manuscript against the repository implementation.

The manuscript presents a multi-UAV mobile edge computing (MEC) framework where:
1. Pre-deployment workload-aware clustering partitions $N$ ground IoT nodes into $C$ spatial clusters.
2. An aerial MEC fleet of $M$ rotary-wing UAVs provides dynamic computational offloading under certificate-based authentication trust $\tau_m(t)$ and operational availability $\phi_m(t)$.
3. Joint optimization of cluster association, task offloading, CPU allocation, and 3D trajectory is solved using Multi-Agent Proximal Policy Optimization (MAPPO) with state-dependent Action Masking.

Several critical mathematical and software inconsistencies were identified in the preliminary codebase and theoretical equations. If left uncorrected, these would cause numerical divergence, inverted physical properties, or invalid research claims. Every discrepancy and its verified resolution are cataloged below.

---

## 2. Inconsistency Catalog & Mathematical Resolutions

### Item 1: NLoS Path Loss Attenuation Sign Inversion
- **Manuscript Equation:**
  $$L_{i,m}(t) = L_{\text{FS}}(d_{i,m}(t)) + P_{i,m}^L(t) \eta_L + (1 - P_{i,m}^L(t)) \eta_N$$
- **Flaw in Prior Code (`config.py` & `src/channel.py`):**
  `NLOS_ATTENUATION_DB` was set to `-20.0 dB`. When $P_{i,m}^L = 0$ (full non-line-of-sight), the total path loss became:
  $$L_{\text{avg}} = L_{\text{FS}} - 20 \text{ dB}$$
  This gave NLoS signals 20 dB *less* attenuation than free-space propagation, making an obstructed link 100 times stronger than line-of-sight!
- **Coherent Resolution:**
  Excess path loss is an attenuation penalty (loss must increase). In standard 3GPP and urban UAV propagation models (e.g., Al-Hourani et al.), $\eta_L \approx 0.0\text{--}1.0 \text{ dB}$ and $\eta_N \approx 20.0 \text{ dB}$ (positive additive loss). In `config.py` and `src/channel.py`, set:
  $$\eta_L = 1.0 \text{ dB}, \quad \eta_N = 20.0 \text{ dB}$$
  Total path loss in dB is strictly greater than or equal to $L_{\text{FS}}$.

---

### Item 2: Weighted K-Means Assignment vs. Centroid Update Contradiction
- **Manuscript Objective & Equations:**
  $$J_{\text{clu}} = \sum_{c=1}^C \sum_{i \in \mathcal{I}_c} w_i \|p_i - z_c\|^2$$
  $$w_i = 1 + \lambda_w \frac{W_i^0}{W_{\max}}$$
  $$z_c = \frac{\sum_{i \in \mathcal{I}_c} w_i p_i}{\sum_{i \in \mathcal{I}_c} w_i}$$
- **Contradiction / Misconception:**
  Some literature informally claims weighted K-means "groups nodes with similar workloads" or that assignment should scale distances by $w_i$.
  Mathematically, because $w_i > 0$ is a static property of device $i$ that does not depend on the choice of cluster $c$, minimizing $J_{\text{clu}}$ with respect to the partition $\{\mathcal{I}_c\}$ for fixed $\{z_c\}$ yields:
  $$\arg\min_c w_i \|p_i - z_c\|^2 \equiv \arg\min_c \|p_i - z_c\|^2$$
  Therefore, device assignment is strictly to the **spatially nearest centroid**. The workload weight $w_i$ exclusively shifts the centroid $z_c$ toward high-workload IoT devices during the centroid update step. It does not by itself balance cluster workloads.
- **Coherent Resolution:**
  1. Devices are assigned to nearest centroid $z_c$ by Euclidean distance.
  2. Centroid updates are calculated via the weighted center of mass.
  3. Workload balancing is not falsely asserted; rather, cluster workload coefficient of variation (CV) and maximum cluster size are explicitly logged and reported.

---

### Item 3: Nominal Workload Dimensionality and BS Commissioning
- **Manuscript Issue:**
  Manuscript text previously conflated single-task workload $W_i(t) = D_i(t) C_i(t)$ (in cycles) with nominal pre-deployment demand $W_i^0$ (in cycles/second).
- **Coherent Resolution:**
  Nominal pre-deployment demand is formally defined as:
  $$W_i^0 = \lambda_i^0 \cdot \mathbb{E}[D_i \cdot C_i] \quad [\text{CPU cycles / second}]$$
  where $\lambda_i^0$ is the average task arrival rate (tasks/sec), $D_i$ is bits/task, and $C_i$ is cycles/bit.
  During pre-deployment commissioning, each IoT node transmits an authenticated 32-byte registration report to the Ground Controller/Base Station (BS):
  $$\text{Report}_i = \{\text{DeviceID}_i, (x_i, y_i, 0), \lambda_i^0, \bar{D}_i, \bar{C}_i, t_{\text{reg}}\}$$
  The BS computes $W_i^0$, derives $w_i$, runs the weighted Lloyd algorithm once offline, and disseminates cluster centroid coordinates $\{z_c\}$ and partition mappings $\{\mathcal{I}_c\}$ to the UAV fleet before takeoff. Continuous real-time BS connectivity is not required during online execution.

---

### Item 4: Task Completion Ratio Denominator
- **Manuscript & Code Bug:**
  In preliminary code, task completion ratio was defined as:
  $$\eta^{\text{task}} = \frac{N_{\text{completed}}}{N \cdot T}$$
  With Bernoulli task generation ($p_{\text{gen}} = 0.7$), at most $0.70 \cdot N \cdot T$ tasks ever arrive. Even a perfect system serving 100% of arrivals would achieve an apparent completion ratio of at most $70\%$, falsely penalizing the policy.
- **Coherent Resolution:**
  The true deadline completion ratio must strictly normalize over generated tasks:
  $$\eta^{\text{task}} = \frac{N_{\text{completed\_on\_time}}}{N_{\text{generated}}}$$
  $N_{\text{generated}}$, $N_{\text{admitted}}$, $N_{\text{completed\_on\_time}}$, $N_{\text{expired}}$, and $N_{\text{rejected}}$ are logged separately. Unfinished tasks remaining in queues at episode termination are counted as incomplete.

---

### Item 5: Continuous-Time Inter-UAV Collision vs. Discrete Endpoints
- **Manuscript Gap:**
  Prior formulation checked inter-UAV separation only at discrete endpoints:
  $$\|q_m(t+1) - q_n(t+1)\| \ge d_{\text{safe}}$$
  If two UAVs cross paths during the 1.0-second interval, they can collide at $t + 0.5$ s while having safe initial and final endpoints.
- **Coherent Resolution:**
  Assuming piecewise-linear constant-velocity motion during $[t, t + \Delta t]$ with initial positions $q_m, q_n$ and displacement vectors $d_m, d_n$, the separation vector as a function of normalized time $s \in [0, 1]$ is:
  $$\Delta p(s) = (q_m - q_n) + s (d_m - d_n) = \Delta p_0 + s \Delta v$$
  The minimum distance squared over $s \in [0, 1]$ is analytically given by:
  $$s^* = \text{clip}\left(-\frac{\Delta p_0 \cdot \Delta v}{\|\Delta v\|^2 + \epsilon}, 0.0, 1.0\right)$$
  $$d_{\min}^2 = \|\Delta p_0 + s^* \Delta v\|^2$$
  A joint deterministic safety shield evaluates $d_{\min} \ge d_{\text{safe}} + d_{\text{margin}}$. If violated, the shield scales back or halts displacement vectors prior to execution, and increments the shield intervention counter.

---

### Item 6: Action Space Factorization & Broken PPO Masking
- **Fatal Code Bug in `src/actor.py`:**
  Prior actor used continuous Tanh squashing over a single flat vector. For invalid actions, it subtracted $10^6$ from Gaussian mean:
  $$\mu_{\text{masked}} = \mu - 10^6 (1 - M)$$
  After sampling, it multiplied the output action by $M$. In `evaluate_actions()`, it applied $\text{atanh}(0) = 0$ against $\mathcal{N}(-10^6, \sigma^2)$, which computes:
  $$\log \mathcal{N}(0; -10^6, \sigma^2) \approx -\frac{10^{12}}{2\sigma^2} \to -\infty \text{ or NaN}$$
  This produced `NaN` gradients and destroyed PPO training.
- **Coherent Resolution:**
  Action spaces are cleanly factorized into true probabilistic heads:
  1. **Cluster Selection $a_m^{\text{clu}}(t) \in \{0, 1, \dots, C\}$:** Categorical distribution with masked logits. For ineligible UAVs ($\tau_m = 0$ or $\phi_m = 0$ or battery critical), logits for $c \ge 1$ are set to $-10^9$, leaving idle ($c=0$) with probability 1.0. Log probability is exact: $\log \pi(c_m) = \text{logits}[c_m] - \log \sum \exp(\text{logits})$.
  2. **Motion Vector $a_m^{\text{mov}}(t) \in [-1, 1]^3$:** Continuous Gaussian with Tanh squashing, with Jacobian correction strictly evaluated on valid proposals.
  3. **CPU Allocation $a_m^{\text{cpu}}(t) \in [0, 1]$:** Continuous fraction of $F_m^{\max}$.
  4. **Task Admission $a_m^{\text{off}}(t) \in [0, 1]$:** Admission fraction / Bernoulli offloading.
  Rollout buffer stores exact sampled raw actions and behavior log probabilities; PPO updates use identical masks, ensuring strictly finite probability ratios.

---

### Item 7: Rotary-Wing Hover Power at Zero Speed
- **Manuscript Formulation:**
  Propulsion power model (Zeng et al.):
  $$P_m^{\text{fly}}(v) = P_1 \left(1 + \frac{3 v^2}{v_{\text{tip}}^2}\right) + P_2 \sqrt{\sqrt{1 + \frac{v^4}{4 v_0^4}} - \frac{v^2}{2 v_0^2}} + \frac{1}{2} d_r \rho s_r A_r v^3$$
- **Verification:**
  At zero forward airspeed ($v = 0$ m/s), parasitic drag is 0, blade profile power is $P_1 = 79.86$ W, and induced power is $P_2 = 88.63$ W.
  $$P_{\text{hover}} = 79.86 + 88.63 = 168.49 \text{ W}$$
  The code accurately incorporates hover power during stationary hovering slots, preventing zero-energy teleportation or free hovering.

---

### Item 8: Queue Evolution and Per-Task Event Timing
- **Manuscript vs. Discrete Event Simulation:**
  Manuscript gives macroscopic backlog update:
  $$Q_m(t+1) = [Q_m(t) - F_m(t)\Delta t]^+ + \sum_i y_{i,m}(t) W_i(t)$$
  and defines $T_m^{\text{que}} = Q_m / F_m$.
- **Resolution:**
  In discrete-time simulation, each admitted task is tracked with causal event timestamps:
  1. $t_{\text{gen}}$: Generation timestamp.
  2. $t_{\text{up\_start}}, t_{\text{up\_end}}$: Uplink transmission interval over orthogonal bandwidth $B_{i,m} = B_m / K_m$.
  3. $t_{\text{que\_entry}} = t_{\text{up\_end}}$: Entry into UAV onboard queue.
  4. $t_{\text{serv\_start}}, t_{\text{serv\_end}}$: Processing by onboard CPU at allocated frequency $f_{i,m}$.
  5. Completion check: Task is successful if $t_{\text{serv\_end}} \le t_{\text{gen}} + \delta_i$.
  Macroscopic queue $Q_m(t)$ tracks remaining CPU cycles of all queued tasks, ensuring exact equivalence between cumulative cycles and individual task progress.

