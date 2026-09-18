"""
Rollout Buffer for Action-Masked MAPPO
=======================================
Stores transitions from multi-agent rollouts (including feasible action masks)
and computes Generalised Advantage Estimation (GAE) for PPO updates.
"""

import numpy as np
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAPPO_GAMMA, MAPPO_GAE_LAMBDA


class RolloutBuffer:
    """
    Multi-agent rollout buffer with Action Masking support.

    Stores per-agent transitions, feasible action masks, and computes
    GAE advantages using the centralised value function.
    Data is consumed by MAPPOAgent.update() via get_batches().
    """

    def __init__(self, rollout_length, n_agents, obs_dim, state_dim,
                 action_dim, action_mask_dim=None, gamma=MAPPO_GAMMA, gae_lambda=MAPPO_GAE_LAMBDA):
        self.rollout_length   = rollout_length
        self.n_agents         = n_agents
        self.obs_dim          = obs_dim
        self.state_dim        = state_dim
        self.action_dim       = action_dim
        self.action_mask_dim  = action_mask_dim or action_dim
        self.gamma            = gamma
        self.gae_lambda       = gae_lambda

        self._allocate()

    def _allocate(self):
        T, N = self.rollout_length, self.n_agents

        self.obs          = np.zeros((T, N, self.obs_dim),         dtype=np.float32)
        self.states       = np.zeros((T, self.state_dim),          dtype=np.float32)
        self.actions      = np.zeros((T, N, self.action_dim),      dtype=np.float32)
        self.action_masks = np.ones((T, N, self.action_mask_dim), dtype=np.float32)
        self.log_probs    = np.zeros((T, N),                       dtype=np.float32)
        self.rewards      = np.zeros((T, N),                       dtype=np.float32)
        self.dones        = np.zeros((T,),                         dtype=np.float32)
        self.values       = np.zeros((T, N),                       dtype=np.float32)

        # Computed by compute_gae()
        self.advantages   = np.zeros((T, N), dtype=np.float32)
        self.returns      = np.zeros((T, N), dtype=np.float32)

        self.step = 0

    def reset(self):
        self._allocate()

    def insert(self, obs, state, actions, log_probs, rewards, done, values, action_masks=None):
        """
        Insert one timestep transition into rollout buffer.
        """
        t = self.step
        self.obs[t]       = obs
        self.states[t]    = state
        self.actions[t]   = actions
        self.log_probs[t] = log_probs
        self.rewards[t]   = rewards
        self.dones[t]     = done
        self.values[t]    = values

        if action_masks is not None:
            self.action_masks[t] = action_masks

        self.step += 1

    def compute_gae(self, last_values, last_done):
        """
        Compute Generalized Advantage Estimation (GAE) and discounted returns.
        """
        gae = np.zeros(self.n_agents, dtype=np.float32)

        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            mask  = 1.0 - self.dones[t]
            delta = (self.rewards[t]
                     + self.gamma * next_values * mask
                     - self.values[t])
            gae   = delta + self.gamma * self.gae_lambda * mask * gae

            self.advantages[t] = gae
            self.returns[t]    = self.advantages[t] + self.values[t]

    def get_batches(self, num_mini_batches):
        """
        Yield mini-batches for Action-Masked PPO training.
        """
        T, N  = self.rollout_length, self.n_agents
        total = T * N
        mini_batch_size = total // num_mini_batches

        # Flatten agent dimension into batch dimension
        obs_flat          = self.obs.reshape(total, -1)
        states_flat       = np.repeat(self.states, N, axis=0)
        actions_flat      = self.actions.reshape(total, -1)
        action_masks_flat = self.action_masks.reshape(total, -1)
        log_probs_flat    = self.log_probs.reshape(total)
        advantages_flat   = self.advantages.reshape(total)
        returns_flat      = self.returns.reshape(total)
        values_flat       = self.values.reshape(total)

        def _pin(arr):
            t = torch.from_numpy(arr)
            try:
                return t.pin_memory()
            except Exception:
                return t

        obs_t          = _pin(obs_flat)
        states_t       = _pin(states_flat)
        actions_t      = _pin(actions_flat)
        action_masks_t = _pin(action_masks_flat)
        log_probs_t    = _pin(log_probs_flat)
        advantages_t   = _pin(advantages_flat)
        returns_t      = _pin(returns_flat)
        values_t       = _pin(values_flat)

        indices = np.random.permutation(total)

        for start in range(0, total, mini_batch_size):
            end    = min(start + mini_batch_size, total)
            mb_idx = indices[start:end]
            agent_ids = torch.from_numpy((mb_idx % N).astype(np.int64))

            yield {
                'obs':           obs_t[mb_idx],
                'states':        states_t[mb_idx],
                'actions':       actions_t[mb_idx],
                'action_masks':  action_masks_t[mb_idx],
                'old_log_probs': log_probs_t[mb_idx],
                'advantages':    advantages_t[mb_idx],
                'returns':       returns_t[mb_idx],
                'old_values':    values_t[mb_idx],
                'agent_ids':     agent_ids,
            }
