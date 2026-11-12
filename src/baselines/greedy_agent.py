"""
Greedy Offloading Baseline Agent
=================================
Rule-based heuristic baseline:
  - Each operational UAV m is greedily assigned to cluster (m % C) + 1.
  - UAV flies directly towards the cluster centroid.
  - UAV allocates full CPU capacity (cpu_fraction = 1.0) to serve offloaded tasks.
  - No neural network training required.
"""

import numpy as np


class GreedyOffloadingAgent:
    """Greedy distance & full-capacity heuristic agent."""

    METHOD = "Greedy"

    def __init__(self, obs_dim=16, state_dim=92, action_dim=5, n_agents=5,
                 device="cpu", n_clusters=5, **kwargs):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.device = device
        self.n_clusters = n_clusters
        self.total_updates = 0

    def get_actions(self, obs_all, state, action_mask=None, num_active=None):
        """
        Returns greedy actions:
          - cluster_id: (m % n_clusters) + 1
          - movement: direction towards assigned cluster centroid from obs
          - cpu_fraction: 1.0
        """
        n_active = num_active or self.n_agents
        actions = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)

        for m in range(n_active):
            cluster_choice = (m % self.n_clusters) + 1
            if action_mask is not None and action_mask[m, cluster_choice] == 0:
                cluster_choice = 0  # Fallback to idle if masked

            # Extract relative centroid vector from observation if available (idx 10, 11)
            dx = float(obs_all[m][10]) if len(obs_all[m]) > 11 else 0.0
            dy = float(obs_all[m][11]) if len(obs_all[m]) > 11 else 0.0
            norm = np.sqrt(dx**2 + dy**2) + 1e-6
            dir_x = np.clip(dx / norm, -1.0, 1.0)
            dir_y = np.clip(dy / norm, -1.0, 1.0)

            actions[m, 0] = float(cluster_choice)
            # Prioritize all candidate offload slots with high score
            actions[m, 1:-4] = 1.0
            # Movement
            actions[m, -4:-1] = [dir_x, dir_y, 0.0]
            # Full CPU
            actions[m, -1] = 1.0

        log_probs = np.zeros(self.n_agents, dtype=np.float32)
        values = np.zeros(self.n_agents, dtype=np.float32)
        return actions, log_probs, values

    def get_values(self, state):
        return np.zeros(self.n_agents, dtype=np.float32)

    def update(self, buffer):
        return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0, 'approx_kl': 0.0}

    def save(self, path):
        pass

    def load(self, path):
        return True
