# EXPERIMENT PROTOCOL
**Paper Title:** Joint Task Offloading and Adaptive Trajectory Optimization in Trust- and Fault-Aware Multi-UAV Edge IoT Networks  
**Target Journal:** IEEE Internet of Things Journal  

---

## 1. Simulation Setup & System Dimensions

The benchmark multi-UAV MEC environment is initialized with the following standardized parameters:

| Parameter | Symbol | Default Value | Unit / Range |
|---|---|---|---|
| Service Area Radius | $R_{\text{area}}$ | 500.0 | meters |
| Number of IoT Devices | $N$ | 100 (sweep: 60--180) | nodes |
| Number of UAV-MEC Servers | $M$ | 5 (sweep: 3--8) | aerial nodes |
| Number of Pre-deployment Clusters | $C$ | 5 | clusters |
| Mission Duration | $T_{\text{tot}}$ | 500 | seconds |
| Slot Duration | $\Delta t$ | 1.0 | seconds |
| UAV Altitude Range | $H_m$ | [50, 150] | meters |
| Safe Inter-UAV Distance | $d_{\text{safe}}$ | 10.0 | meters |
| Task Input Size | $D_i$ | [0.5, 2.5] | Megabits (Mb) |
| CPU Cycles per Bit | $C_i$ | [300, 700] | cycles/bit |
| Task Latency Deadline | $\delta_i$ | [1.0, 3.0] | seconds |
| Task Generation Probability | $p_{\text{gen}}$ | 0.70 | per slot |
| IoT Transmit Power | $P_i$ | 20.0 | dBm (0.1 W) |
| Carrier Frequency | $f_c$ | 2.0 | GHz |
| UAV Bandwidth | $B_m$ | 8.0 -- 16.0 | MHz |
| UAV CPU Capacity | $F_m^{\max}$ | 1.2 -- 3.2 | GHz |
| Switched Capacitance | $\kappa_m$ | $10^{-28}$ | $\text{J} \cdot \text{s}^2 / \text{cycle}^3$ |
| Certificate Duration | $T_{\text{cert}}$ | 1000 | slots |
| UAV Fault Probability | $p_{\text{fault}}$ | 0.001 | per slot |
| Battery Reserve Threshold | $E_{\min}$ | 50.0 | Joules |

---

## 2. Compared Baselines

All comparisons are executed under identical traffic traces, channel fading conditions, UAV initial placements, and hardware budgets:

1. **TF-MAPPO (Proposed):** Full action-masked MAPPO with factorized cluster/motion/CPU heads, dynamic cluster association, trust- and fault-aware eligibility filtering, and joint continuous trajectory safety shield.
2. **MAPPO-NoMask:** Standard unmasked MAPPO with identical network architecture and safety projection only applied at environment execution boundary without policy masking.
3. **MAPPO-NoTrust:** MAPPO without certificate trust $\tau_m(t)$ and operational availability $\phi_m(t)$ eligibility filtering (attempts offloading to faulty UAVs).
4. **Fixed-Cluster-Owner MAPPO:** Static association where UAV $m$ is permanently tied to cluster $c = m \pmod C$, adapting only internal CPU and trajectory.
5. **Dynamic Greedy Offloading:** Rule-based baseline where IoT devices dynamically offload to the nearest operational and eligible UAV, with circular orbit trajectories.
6. **Local Execution Baseline:** All IoT devices process computation tasks exclusively on local low-power microcontrollers ($f_{\text{local}} = 0.5$ GHz) with zero UAV offloading.

---

## 3. Statistical Rigor and Evaluation Rules

1. **Multi-Seed Evaluation:** All reported training curves and test evaluations use at least 5 independent random seeds (`seed \in {42, 43, 44, 45, 46}`).
2. **95% Confidence Interval (CI):** Reported as $\bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{K}}$, where $s$ is sample standard deviation and $K$ is number of seeds.
3. **Held-Out Evaluation:** Training checkpoints are frozen and evaluated on independent, freshly generated 100-episode runs without exploration noise.
4. **Reproducibility:** Every figure is directly generated from a corresponding CSV file in `outputs/` or `results/`.

