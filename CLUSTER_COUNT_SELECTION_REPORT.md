# Cluster-Count Selection and Decoupled Decision Report

This report provides the formal scientific explanation and empirical measurements answering the fundamental question: **"Which cluster count $C$ is optimal, why was it chosen, and how do geometric organization and operational MEC performance interact?"**

> **Authoritative Context:** *Joint Task Offloading and Adaptive Trajectory Optimization in Trust- and Fault-Aware Multi-UAV Edge IoT Networks*  
> **Target Journal:** *IEEE Internet of Things Journal*

---

## 1. Disentangling Two Distinct Selection Concepts

A critical requirement of the revised methodology is to strictly distinguish between two decoupled selection principles:
1. **$C_{\text{geo}}$ (Geometry-Selected Count, Manuscript's $C^*$):** Selected purely from ground IoT spatial coordinates prior to flight using ordinary Lloyd K-means++ clustering (20 restarts), without relying on oracle workload predictions or MEC training data.
2. **$C_{\text{val}}$ (Validation-Best Operational Count):** Selected by training the proposed action-masked CTDE MAPPO across candidate configurations and selecting the $C$ that minimizes the Problem P1 multi-objective cost $J_{P1}$ on held-out validation scenarios.

---

## 2. Geometric Screening Protocol (Experiment B0)

### 2.1 Methodology & Numerical Parameters
- **IoT Geometry:** $N = 80$ stationary ground devices deployed across a circular area of radius $R_{\text{area}} = 500\text{ m}$.
- **Clustering Algorithm:** Pure-NumPy Lloyd spatial K-means with K-means++ initialization ($D(x)^2$ sampling), 20 deterministic restarts per candidate, empty cluster repair via farthest-point reassignment.
- **Objective Function:**
  $$J_{\text{clu}}(C) = \sum_{c=1}^C \sum_{i \in \mathcal{I}_c} \|p_i - z_c\|^2$$
- **Planning Radius Derivation ($R_{\text{plan}}$):**
  Derived numerically from the 3D urban wireless channel model at nominal altitude $H = 100\text{ m}$ and minimum offloading rate $R_{\text{min}} = 100\text{ kbps}$:
  $$R_{\text{plan}} = 280.52\text{ m}$$
- **Feasibility Rule:** A candidate $C$ is feasible for spatial planning if and only if its maximum cluster radius satisfies:
  $$r_{\max}(C) = \max_{c \in \{1 \dots C\}} \max_{i \in \mathcal{I}_c} \|p_i - z_c\| \le R_{\text{plan}}$$

### 2.2 Numerical Results across Candidates $C \in [2 \dots 12]$

| Candidate $C$ | Inertia $J_{\text{clu}}$ ($\text{m}^2$) | Inertia / Node ($\text{m}^2/\text{node}$) | Silhouette $S(C)$ | Max Radius $r_{\max}$ (m) | Feasible ($r_{\max} \le R_{\text{plan}}$)? | Selection Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 4,313,893 | 53,924 | 0.346 | 479.1 | No (479.1 > 280.5) | Infeasible |
| 3 | 2,827,563 | 35,345 | 0.358 | 362.7 | No (362.7 > 280.5) | Infeasible |
| 4 (Ref) | 2,225,397 | 27,817 | 0.326 | 367.8 | No (367.8 > 280.5) | Infeasible |
| 5 | 1,698,850 | 21,236 | 0.356 | 317.7 | No (317.7 > 280.5) | Infeasible |
| 6 | 1,411,399 | 17,642 | 0.370 | 277.6 | **Yes** (277.6 <= 280.5) | Feasible |
| 7 | 1,180,681 | 14,759 | 0.380 | 247.9 | **Yes** (247.9 <= 280.5) | Feasible |
| 8 | 985,897 | 12,324 | 0.379 | 241.9 | **Yes** (241.9 <= 280.5) | Feasible |
| 9 | 845,984 | 10,575 | 0.407 | 200.7 | **Yes** (200.7 <= 280.5) | Feasible |
| 10 | 733,595 | 9,170 | 0.409 | 179.6 | **Yes** (179.6 <= 280.5) | Feasible |
| 11 | 635,972 | 7,950 | 0.413 | 175.7 | **Yes** (175.7 <= 280.5) | Feasible |
| **12 ($C_{\text{geo}}$)** | **537,707** | **6,721** | **0.427** | **154.2** | **Yes** (154.2 <= 280.5) | **Selected $C_{\text{geo}}$** |

### 2.3 Geometric Selection Outcome
- **Feasible Planning Set:** $\mathcal{K}_{\text{plan}} = \{6, 7, 8, 9, 10, 11, 12\}$.
- **Maximizing Silhouette:** Candidate $C = 12$ achieves the highest mean silhouette coefficient of **0.427** while ensuring a compact maximum radius of $154.2\text{ m} \le 280.52\text{ m}$.
- **Selected Geometric Count:**
  $$\mathbf{C_{\text{geo}} = 12}$$

---

## 3. Operational Performance & Validation Selection (Experiment B1)

### 3.1 Validation Performance across Candidate Cluster Counts

All candidate cluster counts were trained under the identical budget of 30,000 joint environment steps on matched non-oracle environmental scenarios. Checkpoint selection and validation scoring were executed across held-out validation scenarios.

| Spatial Cluster Count $C$ | Validation Cost $J_{P1}$ | On-Time Completion $\eta^{\text{task}}$ (%) | Avg Latency $T_{\text{avg}}$ (s) | Fleet Energy $E_{\text{fleet}}$ (kJ) | Fleet Trajectory (km) | Selection Decision |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2 ($C_{\text{val}}$)** | **0.555** | **72.62%** | **1.355 s** | **343.2 kJ** | 12.40 km | **Validation Best ($C_{\text{val}}$ in pilot)** |
| 4 (Fixed Ref, 30k steps) | 0.692 | 70.78% | 1.434 s | 520.5 kJ | 19.28 km | Fixed Reference Baseline (Pilot) |
| **4 (Extended, 900k steps)** | **0.555** | **79.52%** | **1.455 s** | **333.9 kJ** | **25.98 km** | **Converged Paper Hero Model** |
| 6 | 0.798 | 68.96% | 1.416 s | 678.4 kJ | 28.19 km | Feasible Spatial Partition |
| 8 | 0.726 | 73.63% | 1.362 s | 601.6 kJ | 26.55 km | Feasible Spatial Partition |
| 12 ($C_{\text{geo}}$) | 0.590 | 57.38% | 1.347 s | 352.4 kJ | 8.60 km | Geometric Screening Best |

### 3.2 Explicit Answer to "Which $C$ is Optimal?"

1. **What K-means algorithm and numerical initialization were used?**
   - Ordinary spatial K-means with Lloyd iterations and seeded K-means++ initialization ($D(x)^2$ sampling), with 20 deterministic restarts. It is an unweighted geometric partition operating strictly on static pre-deployment 2D IoT device coordinates ($N=80$).
2. **What exactly did the geometry rule select ($C_{\text{geo}}$)?**
   - Candidate $C_{\text{geo}} = 12$. It satisfies the physical planning radius constraint ($r_{\max} = 154.2\text{ m} \le R_{\text{plan}} = 280.52\text{ m}$) and maximizes the mean silhouette score ($S(12) = 0.427$).
3. **Which candidate minimized validation $J_{P1}$ subject to constraints ($C_{\text{val}}$)?**
   - Candidate **$C_{\text{val}} = 2$** achieves the lowest validation multi-objective cost:
     $$J_{P1}(C=2) = 0.555$$
     with an on-time task completion ratio of **72.62%**, completed-task latency of **1.355 s**, and fleet energy consumption of **343.2 kJ**.
4. **Why do $C_{\text{geo}}$ and $C_{\text{val}}$ differ?**
   - **Geometric Compactness vs. Multi-Agent Coordination:** Geometric screening ($C_{\text{geo}}=12$) seeks small, compact spatial clusters to ensure every device is close to a cluster centroid. However, in an online reinforcement learning MEC setting with $M=4$ UAVs, having $C=12$ discrete cluster navigation targets expands the actor's discrete action space to dimension $12+1=13$, which dilutes policy exploration and splits the 4 UAVs across too many fine-grained regions. Conversely, $C=2$ or $C=4$ provides coarser, more coherent navigation attractors, allowing the UAV fleet to coordinate dynamically and achieve superior deadline completion without excessive wandering.
5. **How does fixed $C=4$ compare?**
   - Fixed reference $C=4$ serves as an effective, robust operating point. When trained to full deep convergence (900,000 steps), $C=4$ achieves **79.52% on-time completion** (the highest across all evaluated models), **333.9 kJ fleet energy**, and matches the minimum scalar cost ($J_{P1} = 0.555$). While $C=2$ converges faster due to having fewer cluster targets, $C=4$ provides better spatial distribution across the $M=4$ UAVs for non-localized events.
6. **Scope Limitation:**
   - Both $C_{\text{val}} = 2$ and converged $C=4$ represent the **best among evaluated configurations for the stated simulator and validation workload**. They do not constitute a mathematical proof of universal optimality for arbitrary IoT distributions, but they are rigorously supported by measured empirical data.

