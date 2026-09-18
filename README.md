# Joint Task Offloading and Adaptive Trajectory Optimization in Trust- and Fault-Aware Multi-UAV Edge IoT Networks (TF-MAPPO)

A multi-agent reinforcement learning (MARL) framework for joint UAV trajectory control, cooperative task offloading, and computation resource allocation in trust- and fault-aware Multi-UAV Mobile Edge Computing (MEC) networks.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Mathematical Formulation](#mathematical-formulation)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Paper Evaluation (100 Episodes)](#paper-evaluation-100-episodes)
- [Training](#training)
- [Evaluation & Visualization](#evaluation--visualization)
- [Configuration Reference](#configuration-reference)

---

## Overview

This repository implements **TF-MAPPO**, a hierarchical framework for cooperative multi-UAV MEC networks that addresses four coupled operational challenges:

1. **Scalability**: Pre-deployment workload-aware IoT clustering reduces the online association space before UAV deployment.
2. **Trust & Fault Awareness**: Explicitly separates certificate-based authentication trust $\tau_m(t)$ from operational availability $\phi_m(t)$ to derive joint service eligibility $e_m(t) = \tau_m(t) \cdot \phi_m(t)$.
3. **State-Dependent Action Masking**: Removes infeasible UAV service, energy, computation, deadline, safety, trust, and fault decisions before policy sampling.
4. **Cooperative Multi-Agent Control**: Centralized-Training Decentralized-Execution (CTDE) MAPPO captures inter-UAV coupling during training while enabling scalable local execution.

**Objective (Problem P1):**
$$\min_{\Psi} \quad \omega_E \frac{E_{UAV}^{tot}}{E_{ref}} + \omega_T \frac{T^{avg}}{T_{ref}}$$
subject to constraints on UAV CPU capacity, task deadlines, reserve energy, trust/fault eligibility, mobility speed limits, inter-UAV safety distance, and minimum deadline completion ratio.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     IoT Devices (N devices)                 │
│        Gauss-Markov mobility · variable task workloads      │
└────────────────────┬────────────────────────────────────────┘
                     │  Uplink (A2G Channel, Elevation-dependent LoS)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│             Pre-Deployment IoT Workload Clustering          │
│    Weight w_i = 1 + λ_w (W_i / W_max)  -->  Centroids z_c    │
└────────────────────┬────────────────────────────────────────┘
                     │  Initial Deployment & Dynamic Association
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Multi-UAV MEC Fleet (M Aerial Servers)         │
│  ┌────────────────────┐  ┌────────────────────────────────┐ │
│  │ Trust & Fault      │  │ Action-Masked MAPPO Policy     │ │
│  │ Manager            │  │ (Gaussian Actor + Centralized  │ │
│  │ e_m(t)=τ_m(t)φ_m(t)│  │  Critic)                       │ │
│  └────────────────────┘  └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

- **Pre-deployment Workload-Aware IoT Clustering**: Groups geographically distributed IoT devices into compact service clusters based on spatial distribution and nominal workload requirements ($\lambda_w = 0.5$).
- **Authentication Trust & Fault State Manager**:
  - Certificate-based authentication trust $\tau_m(t) \in \{0, 1\}$ via time-bounded digital credentials.
  - Independent operational availability $\phi_m(t) \in \{0, 1\}$ modeling battery depletion and hardware faults.
  - Overall service eligibility state $e_m(t) = \tau_m(t) \cdot \phi_m(t)$.
- **State-Dependent Action Masking**: Dynamically enforces $x_{m,c}(t), y_{i,m}(t) \le e_m(t)$ and safety/resource constraints before action sampling.
- **Air-to-Ground (A2G) Communication Model**: Elevation-angle-dependent Line-of-Sight (LoS) probability $P_{i,m}^L(t)$, path loss, and Rician fading.
- **Rotary-Wing UAV Energy Model**: Realistic propulsion power model accounting for forward velocity and hover power, plus computation and U2U coordination energy.

---

## Mathematical Formulation

### 1. Workload-Aware Clustering (Eqs. 1–3)
$$w_i = 1 + \lambda_w \frac{W_i^0}{W_{max}}, \quad z_c = \frac{\sum_{i \in \mathcal{I}_c} w_i p_i}{\sum_{i \in \mathcal{I}_c} w_i}$$

### 2. Service Eligibility (Eqs. 27–29)
$$\tau_m(t) = \mathbb{I}(\text{Valid Certificate}), \quad \phi_m(t) = \mathbb{I}(\text{Operational}), \quad e_m(t) = \tau_m(t) \cdot \phi_m(t)$$

### 3. Total Processing Latency (Eqs. 10, 19–21)
$$T_{i,m}^{tot}(t) = T_{i,m}^{up}(t) + T_m^{que}(t) + T_{i,m}^{exe}(t)$$

---

## Repository Structure

```
TF_MAPPO_Multi-UAV/
├── config.py              # System hyperparameters and simulation constants
├── train.py               # Training entry point (Action-Masked MAPPO)
├── run_100_episodes.py    # 100-episode paper evaluation script
├── test_custom.py         # Evaluation script on custom configurations
├── plot_ieee.py           # Evaluation figures generation script
├── requirements.txt       # Python package dependencies
└── src/
    ├── env.py             # Multi-UAV MEC environment (A2G channel, queues, energy)
    ├── clustering.py      # Workload-aware K-Means IoT clustering manager
    ├── trust_fault_manager.py # Trust, fault, and action-masking manager
    ├── actor.py           # Action-Masked Gaussian MAPPO actor
    ├── mappo_agent.py     # MAPPO agent with Centralized Critic (CTDE)
    ├── channel_model.py   # Air-to-Ground LoS/NLoS channel & fading model
    ├── buffer.py          # Rollout buffer with GAE advantage estimation
    ├── metrics.py         # Latency, energy, completion ratio tracking
    └── logger.py          # Training logger & trajectory recorder
```

---

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- ImageIO >= 2.31.0
- ImageIO-FFmpeg >= 0.4.8

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/TF_MAPPO_Multi-UAV.git
cd TF_MAPPO_Multi-UAV

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Paper Evaluation Framework (100 Episodes)
Runs 100 evaluation episodes ($100 \times 500 = 50,000$ steps) to compute statistical metrics ($\text{mean} \pm \text{std}$) for latency, UAV energy, task completion ratio, trajectory distance, and eligibility state:

```bash
python run_100_episodes.py
```
**Results saved to:** `results/test_results/paper_100_episodes/`

### 2. Training the MAPPO Agent
```bash
python train.py --device cpu
```

### 3. Custom Evaluation & Plotting
```bash
# Custom evaluation on 6 UAVs and 120 IoT devices
python test_custom.py --num-uavs 6 --num-iot 120

# Generate IEEE paper figures
python plot_ieee.py
```

---

## Configuration Reference

Key parameters in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `NUM_UAVS` | `5` | Number of aerial MEC servers |
| `NUM_USERS` | `100` | Number of ground IoT devices |
| `LAMBDA_W` | `0.5` | Workload weight coefficient for IoT clustering |
| `OMEGA_E` | `0.5` | Weight for energy consumption in objective P1 |
| `OMEGA_T` | `0.5` | Weight for task latency in objective P1 |
| `MIN_TASK_COMPLETION_RATIO` | `0.85` | Required deadline task completion ratio ($\eta_{min}$) |
| `SAFE_INTER_UAV_DISTANCE` | `10.0` | Minimum safe distance between UAVs ($d^{safe}$ in meters) |
