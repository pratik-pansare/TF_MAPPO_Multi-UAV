"""
Random Feasible Baseline Agent
==============================
Paper: Joint Task Offloading and Adaptive Trajectory Optimization in
       Trust- and Fault-Aware Multi-UAV Edge IoT Networks
Target Journal: IEEE Internet of Things Journal

Implements:
  - Uniform sampling of feasible cluster target, candidate priority scores,
    bounded 3D displacements, and CPU allocation fractions.
  - Passes through the exact same physical validator and execution shield.
"""

import numpy as np


class RandomFeasibleAgent:
    """Random feasible baseline policy."""

    METHOD = "RANDOM"

    def __init__(self, obs_dim=16, state_dim=92, action_dim=13, n_agents=4,
                 n_clusters=4, max_candidate_pool=8, device="cpu", **kwargs):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.n_clusters = n_clusters
        self.max_candidate_pool = max_candidate_pool
        self.device = device
        self.total_updates = 0

    def get_actions(self, obs_all, state, action_mask=None, num_active=None, deterministic=False):
        n_active = num_active or self.n_agents
        actions = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)

        for m in range(n_active):
            # Cluster choice: uniformly sample among feasible clusters from mask
            clu_dim = self.n_clusters + 1
            if action_mask is not None and len(action_mask[m]) >= clu_dim:
                clu_mask = action_mask[m, :clu_dim]
                valid_clusters = np.where(clu_mask > 0.5)[0]
                if len(valid_clusters) > 0:
                    c = int(np.random.choice(valid_clusters))
                else:
                    c = 0
            else:
                c = int(np.random.randint(0, self.n_clusters + 1))

            actions[m, 0] = float(c)

            # Candidate priorities: uniform [0, 1]
            idx_off_end = 1 + self.max_candidate_pool
            actions[m, 1:idx_off_end] = np.random.uniform(0.0, 1.0, size=self.max_candidate_pool)

            # 3D movement: uniform [-1, 1]
            actions[m, idx_off_end:idx_off_end + 3] = np.random.uniform(-1.0, 1.0, size=3)

            # CPU fraction: uniform [0.2, 1.0]
            actions[m, idx_off_end + 3] = np.random.uniform(0.2, 1.0)

        log_probs = np.zeros(self.n_agents, dtype=np.float32)
        values = np.zeros(self.n_agents, dtype=np.float32)
        return actions, log_probs, values

    def get_values(self, state):
        return np.zeros(self.n_agents, dtype=np.float32)

    def update(self, buffer):
        return {}

    def save(self, path):
        pass

    def load(self, path):
        pass

