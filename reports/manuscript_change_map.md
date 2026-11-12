# Manuscript Change Map & Mathematical Amendments

**Manuscript Title:** Joint Task Offloading and Adaptive Trajectory Optimization in Trust- and Fault-Aware Multi-UAV Edge IoT Networks  
**Target Journal:** IEEE Internet of Things Journal  
**Document Purpose:** Exact line-by-line and equation-by-equation amendment mapping to align the IEEE IoT-J manuscript draft with supervisor instructions.

---

## 1. Summary of Changes

| Manuscript Section | Original Equation / Text | Amended Equation / Text | Rationale |
|---|---|---|---|
| **Section III.A (Clustering)** | Eq. (1)–(3): Workload-weighted Lloyd clustering with pre-deployment arrival estimates $\lambda_i^0, W_i^0, w_i$. | Eq. (1): Standard spatial geometric K-Means with unweighted centroids $z_c = \frac{1}{|\mathcal{I}_c|} \sum_{i \in \mathcal{I}_c} p_i$. | Eliminates unrealistic pre-deployment oracle workload assumptions. Clusters provide spatial regional organization and initial waypoints. |
| **Section III.B (Task Model)** | Eq. (14): Static Bernoulli task arrival with fixed per-device probability $p_{\text{gen}}$. | Dynamic environment state $z_i(t)$ and contextual event triggers $a_i(t) = \mathbf{1}[g_i(z_i(t), t) \text{ triggers job}]$. | Enables realistic heterogeneous and time-varying workload profiles (Routine, Hotspot, Emergency, Shift). |
| **Section III.C (Offloading)** | Offloading permitted only if device cluster matches UAV assigned cluster ($y_{im} \le x_{mc}$). | Dynamic coverage link set $\mathcal{I}_{m,\text{cov}}(t) = \{i : d_{im}(t) \le R_{m,\text{srv}} \wedge R_{im}(t) \ge R_{\min}\}$. | Devices communicate with any eligible UAV within physical line-of-sight/coverage range, regardless of cluster index. |
| **Section III.D (Concurrency)** | No constraint on simultaneous active communication links; full bandwidth divided across all cluster devices. | Hard radio concurrency cap $\sum_{i=1}^N u_{im}(t) \le K_m^{\text{comm}}$ per UAV, and $\sum_{m=1}^M u_{im}(t) \le 1$ per device. | Reflects physical transceiver and RF front-end hardware limits on simultaneous orthogonal channels. |
| **Section III.E (Multi-Slot Upload)** | One-slot upload abstraction ($T_{up} = D_i / R_{im}$). | Causal multi-slot progress: $D_i^{\text{rem}}(t+\Delta t) = \max(0, D_i^{\text{rem}}(t) - R_{im}(t)\Delta t)$. | Accurately models transmission across slot boundaries when data size exceeds single-slot throughput. |
| **Section IV.A (Actions)** | 3-head action: cluster $c_m$, displacement $\Delta p_m$, CPU $f_m$. | 4-head factorized action: $a_m = (a_m^{\text{clu}}, a_m^{\text{off}}, a_m^{\text{cpu}}, a_m^{\text{mov}})$. | $a_m^{\text{off}}$ dynamically schedules covered candidate devices under the $K_m^{\text{comm}}$ limit. |

---

## 2. LaTeX Snippets for Manuscript Updates

### A. Section III.A — Spatial K-Means Formulation
Replace Eq. (1)–(3) with:
```latex
\subsection{Pre-Deployment Spatial Clustering}
To provide macro-level spatial indexing and initial aerial dispatch coordinates without relying on fictitious pre-deployment workload estimations, the $N$ stationary ground IoT devices located at $\mathbf{p}_i = (x_i, y_i, 0)$ are partitioned into $C$ compact geometric clusters $\mathcal{I}_1, \dots, \mathcal{I}_C$ via spatial K-Means with seeded K-Means++ initialization:
\begin{equation}
    \min_{\{\mathbf{z}_c\}_{c=1}^C} J_{\text{clu}} = \sum_{c=1}^C \sum_{i \in \mathcal{I}_c} \|\mathbf{p}_i - \mathbf{z}_c\|^2
    \label{eq:spatial_kmeans}
\end{equation}
where the geometric cluster centroid $\mathbf{z}_c$ is updated iteratively as:
\begin{equation}
    \mathbf{z}_c = \frac{1}{|\mathcal{I}_c|} \sum_{i \in \mathcal{I}_c} \mathbf{p}_i, \quad \forall c \in \{1, \dots, C\}.
    \label{eq:unweighted_centroid}
\end{equation}
Crucially, these cluster boundaries serve exclusively for macro-level spatial coordination and visual indexing; real-time task offloading eligibility is governed dynamically by physical communication range and channel quality rather than cluster partition labels.
```

### B. Section III.C — Coverage-Driven Communication and Concurrency Cap
Replace the cluster-membership offloading constraints with:
```latex
\subsection{Coverage-Driven Communication and Link Concurrency Cap}
At each discrete time slot $t$, UAV $m$ at 3D location $\mathbf{q}_m(t) = (x_m(t), y_m(t), H_m(t))$ maintains a physical communication coverage disk of radius $R_{m,\text{srv}}$. The set of candidate IoT devices eligible to establish uplink transmission with UAV $m$ is defined dynamically as:
\begin{equation}
    \mathcal{I}_{m,\text{cov}}(t) = \left\{ i \in \{1, \dots, N\} : d_{im}(t) \le R_{m,\text{srv}}, \; R_{im}(t) \ge R_{i,\min}, \; e_m(t) = 1 \right\}
    \label{eq:coverage_candidate_set}
\end{equation}
where $d_{im}(t) = \sqrt{\|\mathbf{q}_m(t) - \mathbf{p}_i\|^2}$ is the 3D Euclidean distance, $R_{im}(t)$ is the instantaneous uplink data rate, and $e_m(t) = \tau_m(t)\phi_m(t)$ denotes UAV service eligibility.

Let $u_{im}(t) \in \{0, 1\}$ denote the binary indicator of an active communication link between device $i$ and UAV $m$ during time slot $t$. To respect physical RF transceiver constraints, each UAV can maintain at most $K_m^{\text{comm}}$ concurrent active uplinks:
\begin{equation}
    \sum_{i=1}^N u_{im}(t) \le K_m^{\text{comm}}, \quad \forall m \in \{1, \dots, M\} \quad (\text{Constraint C3a})
    \label{eq:concurrency_cap}
\end{equation}
Furthermore, to prevent redundant transmissions and packet collisions, each device can transmit to at most one UAV simultaneously:
\begin{equation}
    \sum_{m=1}^M u_{im}(t) \le 1, \quad \forall i \in \{1, \dots, N\} \quad (\text{Constraint C3b})
    \label{eq:device_single_link}
\end{equation}
Each scheduled link is allocated an orthogonal bandwidth sub-carrier $B_{im}(t) = B_{m,\text{total}} / \sum_{j=1}^N u_{jm}(t)$, satisfying $\sum_{i=1}^N u_{im}(t) B_{im}(t) \le B_{m,\text{total}}$.
```

### C. Section III.D — Multi-Slot Uplink Transmission
```latex
\subsection{Causal Multi-Slot Uplink Transmission Model}
Unlike idealized single-slot abstractions, tasks requiring substantial data transfer $D_i(t)$ progress across multiple discrete slots. The remaining transmission backlog $D_i^{\text{rem}}(t)$ evolves causally as:
\begin{equation}
    D_i^{\text{rem}}(t + \Delta t) = \max\left(0, \; D_i^{\text{rem}}(t) - \sum_{m=1}^M u_{im}(t) R_{im}(t) \Delta t\right)
    \label{eq:multislot_transmission}
\end{equation}
When $D_i^{\text{rem}} = 0$, the upload completes at timestamp $t_{\text{finish\_up}}$, and the task enters the onboard discrete-event FIFO/EDF computation queue of UAV $m$. If UAV $m$ moves out of range ($d_{im} > R_{m,\text{srv}}$) or experiences a hardware fault ($\phi_m = 0$) before upload completion, $u_{im}(t)$ drops to 0 and transmission halts.
```

---

## 3. Recommended Manuscript Text Edits by Section

1. **Abstract:** Remove any mention of "workload-weighted clustering" or "static task offloading." Add: *"We develop a coverage-driven communication framework subject to hard per-UAV simultaneous link concurrency caps ($K_m^{\text{comm}}$) and causal multi-slot discrete-event task progress."*
2. **Section II (Related Work):** Clarify that unlike existing heuristic offloading literature that assumes all nodes within a cluster are simultaneously served, our model explicitly enforces physical transceiver concurrency bounds and dynamic spatial coverage.
3. **Section V (Simulation Setup):** Specify the 4 environmental scenarios (Routine, Localized Hotspot, Multi-Event, Persistent Shift) and table of application profiles (data sizes 0.5–2.5 Mb, cycles 300–700 c/b, deadlines 1.0–3.0 s).
4. **Section VI (Numerical Results):** Include real learned trajectory plots comparing untrained vs early vs trained checkpoints, showing dynamic coverage circles and spatial adaptation to hotspot and fault events.

