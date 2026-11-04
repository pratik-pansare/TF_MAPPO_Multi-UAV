"""
Local Execution Baseline Agent
==============================
Rule-based heuristic baseline:
  - All tasks executed locally at IoT devices; no UAV assistance.
  - UAV fleet idles at cluster centroids with zero compute offload.
  - Actions emitted: [cluster_id=0, dx=0, dy=0, dz=0, cpu=0].
"""

import numpy as np


class LocalExecutionAgent:
    """Local execution baseline agent."""

    METHOD = "LocalOnly"

    def __init__(self, obs_dim=16, state_dim=92, action_dim=5, n_agents=5,
                 device="cpu", **kwargs):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.device = device
        self.total_updates = 0

    def get_actions(self, obs_all, state, action_mask=None, num_active=None):
        actions = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)
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
