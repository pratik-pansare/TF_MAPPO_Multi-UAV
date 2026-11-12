"""
Independent PPO (IPPO) Baseline Agent
======================================
Paper: Joint Task Offloading and Adaptive Trajectory Optimization in
       Trust- and Fault-Aware Multi-UAV Edge IoT Networks
Target Journal: IEEE Internet of Things Journal

Differences from Proposed MAPPO:
  - Local Critic V(o_m) per agent using local observations ONLY (no global centralized state).
  - Isolates the contribution of Centralized Training with Decentralized Execution (CTDE).
  - Uses the same action-masked hybrid actor architecture and common execution validator.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    MAPPO_LR_ACTOR, MAPPO_LR_CRITIC, MAPPO_CLIP_EPS,
    MAPPO_ENTROPY_COEF, MAPPO_VALUE_COEF, MAPPO_MAX_GRAD_NORM,
    MAPPO_PPO_EPOCHS, MAPPO_NUM_MINI_BATCHES,
    MAPPO_HIDDEN_DIM, MAPPO_CRITIC_HIDDEN_DIM,
)
from src.actor import HybridActionMaskedActor
from src.buffer import RolloutBuffer


class LocalCritic(nn.Module):
    """Local observation-based critic for Independent PPO (IPPO)."""
    def __init__(self, obs_dim, hidden_dim=MAPPO_CRITIC_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, obs):
        return self.net(obs)


class IPPOAgent:
    """
    Independent PPO (IPPO) agent: local actor and local critic.
    """
    METHOD = "IPPO"

    def __init__(self, obs_dim, state_dim, action_dim, n_agents,
                 device='cpu', share_actor=True, n_clusters=5, max_candidate_pool=8, **kwargs):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.n_clusters = n_clusters
        self.max_candidate_pool = max_candidate_pool
        self.device = device
        self.share_actor = share_actor

        self.actor = HybridActionMaskedActor(
            obs_dim, action_dim,
            n_clusters=n_clusters,
            max_candidate_pool=max_candidate_pool,
            hidden_dim=MAPPO_HIDDEN_DIM
        ).to(device)
        self.critic = LocalCritic(obs_dim, hidden_dim=MAPPO_CRITIC_HIDDEN_DIM).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=MAPPO_LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=MAPPO_LR_CRITIC)
        self.total_updates = 0

    @torch.inference_mode()
    def get_actions(self, obs_all, state, action_mask=None, num_active=None, deterministic=False):
        n_active = num_active or self.n_agents
        actions = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)
        log_probs = np.zeros(self.n_agents, dtype=np.float32)

        obs_batch = torch.from_numpy(np.ascontiguousarray(obs_all[:n_active], dtype=np.float32)).to(self.device)
        mask_batch = None
        if action_mask is not None:
            mask_batch = torch.from_numpy(np.ascontiguousarray(action_mask[:n_active], dtype=np.float32)).to(self.device)

        a_batch, lp_batch = self.actor.get_action_and_log_prob(obs_batch, action_mask=mask_batch, deterministic=deterministic)
        actions[:n_active] = a_batch.cpu().numpy()
        log_probs[:n_active] = lp_batch.cpu().numpy().flatten()

        # Local value estimates V(o_m)
        v_batch = self.critic(obs_batch).flatten().cpu().numpy()
        values = np.zeros(self.n_agents, dtype=np.float32)
        values[:n_active] = v_batch

        return actions, log_probs, values

    @torch.inference_mode()
    def get_values(self, state):
        # Fallback to local obs if state has local obs slices
        obs_dim = self.obs_dim
        values = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            start = i * obs_dim
            if start + obs_dim <= len(state):
                obs_t = torch.from_numpy(state[start:start+obs_dim].astype(np.float32)).unsqueeze(0).to(self.device)
                values[i] = float(self.critic(obs_t).cpu().item())
        return values

    def update(self, buffer):
        self.total_updates += 1
        actor_losses, critic_losses, entropies, kl_divs = [], [], [], []

        for _ in range(MAPPO_PPO_EPOCHS):
            for batch in buffer.get_batches(MAPPO_NUM_MINI_BATCHES):
                obs = batch['obs'].to(self.device, non_blocking=True)
                actions = batch['actions'].to(self.device, non_blocking=True)
                old_log_probs = batch['old_log_probs'].to(self.device, non_blocking=True)
                adv = batch['advantages'].to(self.device, non_blocking=True)
                ret = batch['returns'].to(self.device, non_blocking=True)
                masks = batch.get('action_masks', None)
                if masks is not None:
                    masks = masks.to(self.device, non_blocking=True)

                # Normalize advantages per mini-batch
                adv_std = adv.std()
                adv_mean = adv.mean()
                adv_norm = (adv - adv_mean) / (adv_std + 1e-8)

                # Actor update with IPPO
                new_log_probs, entropy = self.actor.evaluate_actions(obs, actions, action_mask=masks)
                new_log_probs = new_log_probs.squeeze(-1)
                entropy = entropy.squeeze(-1)
                ratios = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratios * adv_norm
                surr2 = torch.clamp(ratios, 1.0 - MAPPO_CLIP_EPS, 1.0 + MAPPO_CLIP_EPS) * adv_norm
                actor_loss = -torch.min(surr1, surr2).mean() - MAPPO_ENTROPY_COEF * entropy.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), MAPPO_MAX_GRAD_NORM)
                self.actor_optimizer.step()

                # Critic update with local observations
                values = self.critic(obs).squeeze(-1)
                critic_loss = F.mse_loss(values, ret)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAPPO_MAX_GRAD_NORM)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean().item()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())
                kl_divs.append(approx_kl)

        return {
            'actor_loss': float(np.mean(actor_losses)) if actor_losses else 0.0,
            'critic_loss': float(np.mean(critic_losses)) if critic_losses else 0.0,
            'entropy': float(np.mean(entropies)) if entropies else 0.0,
            'approx_kl': float(np.mean(kl_divs)) if kl_divs else 0.0,
        }

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
            'updates': self.total_updates,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor_state'])
        self.critic.load_state_dict(ckpt['critic_state'])
        if 'actor_opt' in ckpt:
            self.actor_optimizer.load_state_dict(ckpt['actor_opt'])
        if 'critic_opt' in ckpt:
            self.critic_optimizer.load_state_dict(ckpt['critic_opt'])
        self.total_updates = ckpt.get('updates', 0)
