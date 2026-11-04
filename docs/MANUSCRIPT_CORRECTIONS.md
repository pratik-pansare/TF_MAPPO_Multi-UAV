# MANUSCRIPT CORRECTIONS AND PROPOSED LATEX AMENDMENTS
**Paper Title:** Joint Task Offloading and Adaptive Trajectory Optimization in Trust- and Fault-Aware Multi-UAV Edge IoT Networks  
**Target Journal:** IEEE Internet of Things Journal  

This document details the exact LaTeX corrections required to align the manuscript with sound physical, mathematical, and algorithmic principles.

---

## 1. Correction to Air-to-Ground Channel Path Loss (Section III.B)

### Issue
The original manuscript text listed $\eta_N = -20 \text{ dB}$ as the additional non-line-of-sight attenuation, which inadvertently treated non-line-of-sight propagation as an amplification rather than an attenuation.

### Proposed LaTeX Amendment
```latex
% In Section III.B, replace equation and description with:
The average path loss between IoT device $i$ and UAV $m$ at time slot $t$ is modeled by:
\begin{equation}
L_{i,m}(t) = L_{\mathrm{FS}}(d_{i,m}(t)) + P_{i,m}^{\mathrm{L}}(t)\,\eta_{\mathrm{L}} + \left(1 - P_{i,m}^{\mathrm{L}}(t)\right)\eta_{\mathrm{N}},
\end{equation}
where $L_{\mathrm{FS}}(d_{i,m}(t)) = 20\log_{10}\left(\frac{4\pi f_c d_{i,m}(t)}{c_0}\right)$ denotes the free-space path loss, and $\eta_{\mathrm{L}}$ and $\eta_{\mathrm{N}}$ represent the mean excessive path loss (in dB) incurred by LoS and NLoS propagation, respectively. For typical urban environments, $\eta_{\mathrm{L}} = 1.0\text{ dB}$ and $\eta_{\mathrm{N}} = 20.0\text{ dB}$, ensuring that severe shadowing attenuates the received power gain $g_{i,m}(t) = 10^{-L_{i,m}(t)/10}$.
```

---

## 2. Correction to Workload-Aware Clustering Formulation (Section III.A)

### Issue
The paper previously stated that the weighted K-means formulation inherently balances computational workloads across clusters. In Lloyd iterations, the objective minimizes spatial distortion weighted by static device importance; assignment is based strictly on spatial distance to the weighted centroid.

### Proposed LaTeX Amendment
```latex
% In Section III.A, replace clustering description with:
Before UAV fleet dispatch, the ground controller partitions the $N$ stationary IoT devices into $C$ localized spatial clusters $\{\mathcal{I}_c\}_{c=1}^C$ using a spatial workload-weighted Lloyd's algorithm. Each device transmits a nominal profiling report containing its estimated task generation rate $\lambda_i^0$ and mean workload parameters $(\bar{D}_i, \bar{C}_i)$. The controller computes the nominal computational demand:
\begin{equation}
W_i^0 = \lambda_i^0 \cdot \left(\bar{D}_i \cdot \bar{C}_i\right) \quad [\text{CPU cycles/s}],
\end{equation}
and defines the normalized workload weight as:
\begin{equation}
w_i = 1 + \lambda_w \frac{W_i^0}{\max_{j} W_j^0},
\end{equation}
where $\lambda_w \ge 0$ modulates the influence of high-demand hotspots. The clustering objective minimizes the weighted sum of squared spatial distances:
\begin{equation}
\min_{\{\mathcal{I}_c, z_c\}} \sum_{c=1}^C \sum_{i \in \mathcal{I}_c} w_i \|p_i - z_c\|^2.
\end{equation}
For fixed centroids $z_c$, because $w_i > 0$ depends only on device $i$, assignment is determined by nearest Euclidean distance: $\arg\min_c \|p_i - z_c\|$. The weight $w_i$ shifts the centroid $z_c$ toward computational hotspots during the centroid update step:
\begin{equation}
z_c = \frac{\sum_{i \in \mathcal{I}_c} w_i p_i}{\sum_{i \in \mathcal{I}_c} w_i}.
\end{equation}
Cluster workload imbalance is subsequently evaluated via the coefficient of variation (CV) of aggregate cluster demand across partitions.
```

---

## 3. Amendment to Task Completion Metric Definition (Section IV)

### Issue
Normalizing the deadline task completion ratio by $N \cdot T$ falsely assumed all devices generate tasks every slot, capping the observable completion ratio at the Bernoulli arrival probability.

### Proposed LaTeX Amendment
```latex
% In Section IV, replace the definition of eta^task with:
The overall mission deadline task completion ratio $\eta^{\mathrm{task}}$ is defined strictly with respect to the total number of computation tasks actually generated during the mission horizon:
\begin{equation}
\eta^{\mathrm{task}} = \frac{\sum_{t=1}^T \sum_{i=1}^N a_i(t) \cdot \mathbf{1}\left[T_{i,m}^{\mathrm{tot}}(t) \le \delta_i(t) \text{ and status} = \text{COMPLETED}\right]}{\sum_{t=1}^T \sum_{i=1}^N a_i(t)},
\end{equation}
where $a_i(t) \in \{0, 1\}$ is the Bernoulli task arrival indicator. Any admitted task remaining in an onboard UAV queue at the conclusion of mission slot $T$ is marked as incomplete.
```

---

## 4. Amendment to Continuous-Time Inter-UAV Collision Constraint (Section III.F)

### Issue
Discrete-endpoint separation $\|q_m(t+1) - q_n(t+1)\| \ge d_{\mathrm{safe}}$ permits trajectories that cross midway through the time step.

### Proposed LaTeX Amendment
```latex
% In Section III.F, amend Constraint C10 to:
To guarantee inter-UAV flight safety throughout the continuous duration of each synchronized interval $[t\Delta t, (t+1)\Delta t]$, we enforce continuous trajectory separation under piecewise-linear motion:
\begin{equation}
\min_{s \in [0, 1]} \left\| \left(q_m(t) - q_n(t)\right) + s \left(d_m(t) - d_n(t)\right) \right\| \ge d_{\mathrm{safe}}, \quad \forall m \neq n,
\end{equation}
where $d_m(t) = q_m(t+1) - q_m(t)$ denotes the 3D displacement vector of UAV $m$. A deterministic safety shield intercepts proposed velocity actions violating this condition, projecting them to collision-free maneuvers before physical execution.
```

