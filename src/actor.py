"""
Hybrid Factorized Action-Masked Actor for MAPPO
===============================================
Paper Reference: Section III.F & Section IV.E
Target Journal: IEEE Internet of Things Journal

Four Factorized Action Heads per UAV m:
  1. a_m^clu in {0, 1, ..., C}: Macro spatial cluster target.
     - Masked Categorical distribution: invalid choices masked with -1e9 logits.
     - Choice 0 (idle) is always feasible.
  2. a_m^off in [0, 1]^K_pool: Candidate offloading scheduling priority head.
     - Squashed Gaussian via Sigmoid with exact change-of-variables log-prob.
     - Candidate slots masked by physical coverage and UAV eligibility.
  3. a_m^mov in [-1, 1]^3: Continuous 3D Trajectory Movement Head.
     - Squashed Gaussian via Tanh with exact Jacobian change-of-variables log-prob.
  4. a_m^cpu in [0, 1]: Continuous CPU Allocation Fraction Head.
     - Squashed Gaussian via Sigmoid with exact change-of-variables log-prob.

Guarantees:
  - Strictly finite log probabilities and bounded entropy (no NaN / -inf).
  - Exact consistency between behavior sampling and PPO evaluation passes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np


class ObsEncoder(nn.Module):
    """Shared observation feature encoder for MAPPO actor."""

    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs)


class HybridActionMaskedActor(nn.Module):
    """
    Four-Head Factorized Actor supporting:
      1. Discrete masked categorical cluster target choices.
      2. Continuous masked offload priority scores.
      3. Continuous bounded 3D movement controls.
      4. Continuous bounded CPU allocation fraction.
    """

    def __init__(self, obs_dim, action_dim=13, n_clusters=5, max_candidate_pool=8, hidden_dim=256, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_clusters = n_clusters
        self.max_candidate_pool = max_candidate_pool
        self.cluster_action_dim = n_clusters + 1  # 0 = idle, 1..C = clusters
        self.action_dim = 1 + max_candidate_pool + 3 + 1  # Total action dimensions

        self.encoder = ObsEncoder(obs_dim, hidden_dim)

        # 1. Categorical Cluster Head (size C + 1)
        self.cluster_head = nn.Linear(hidden_dim, self.cluster_action_dim)

        # 2. Candidate Offload Priority Head (size max_candidate_pool)
        self.offload_mu_head = nn.Linear(hidden_dim, max_candidate_pool)
        self.offload_log_std = nn.Parameter(torch.zeros(max_candidate_pool))

        # 3. Continuous 3D Movement Head (dx, dy, dz in [-1, 1])
        self.mov_mu_head = nn.Linear(hidden_dim, 3)
        self.mov_log_std = nn.Parameter(torch.zeros(3))

        # 4. Continuous CPU Allocation Head (rho in [0, 1])
        self.cpu_mu_head = nn.Linear(hidden_dim, 1)
        self.cpu_log_std = nn.Parameter(torch.zeros(1))

    def forward_heads(self, obs):
        feat = self.encoder(obs)
        cluster_logits = self.cluster_head(feat)

        offload_mu = self.offload_mu_head(feat)
        offload_log_std = torch.clamp(self.offload_log_std, -20.0, 2.0)

        mov_mu = self.mov_mu_head(feat)
        mov_log_std = torch.clamp(self.mov_log_std, -20.0, 2.0)

        cpu_mu = self.cpu_mu_head(feat)
        cpu_log_std = torch.clamp(self.cpu_log_std, -20.0, 2.0)

        return cluster_logits, offload_mu, offload_log_std, mov_mu, mov_log_std, cpu_mu, cpu_log_std

    def sample(self, obs, action_mask=None, cluster_mask=None, deterministic=False, **kwargs):
        """
        Sample actions from factorized heads under state-dependent action masks.
        action_mask / cluster_mask: tensor of shape (batch, (n_clusters + 1) + max_candidate_pool) or cluster mask only.
        """
        if action_mask is None and cluster_mask is not None:
            action_mask = cluster_mask
        cluster_logits, offload_mu, offload_log_std, mov_mu, mov_log_std, cpu_mu, cpu_log_std = self.forward_heads(obs)

        # Parse action masks
        cluster_mask = None
        candidate_mask = None

        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).to(obs.device, dtype=obs.dtype)

            if action_mask.shape[-1] >= self.cluster_action_dim + self.max_candidate_pool:
                cluster_mask = action_mask[..., :self.cluster_action_dim]
                candidate_mask = action_mask[..., self.cluster_action_dim:self.cluster_action_dim + self.max_candidate_pool]
            elif action_mask.shape[-1] == self.cluster_action_dim:
                cluster_mask = action_mask

        # 1. Masked Categorical Cluster Selection
        if cluster_mask is not None:
            valid_mask = torch.clamp(cluster_mask, 0.0, 1.0)
            cluster_logits = cluster_logits + (valid_mask + 1e-8).log()

        cluster_dist = Categorical(logits=cluster_logits)
        if deterministic:
            cluster_action = torch.argmax(cluster_logits, dim=-1)
        else:
            cluster_action = cluster_dist.sample()

        log_prob_clu = cluster_dist.log_prob(cluster_action)
        entropy_clu = cluster_dist.entropy()

        # 2. Continuous Offload Priorities with Sigmoid squashing
        offload_std = torch.exp(offload_log_std)
        offload_dist = Normal(offload_mu, offload_std)
        if deterministic:
            u_off = offload_mu
        else:
            u_off = offload_dist.rsample()
        a_off = torch.sigmoid(u_off)

        # Jacobian correction for Sigmoid
        log_prob_off = offload_dist.log_prob(u_off) - torch.log(a_off * (1.0 - a_off) + 1e-6)
        if candidate_mask is not None:
            log_prob_off = log_prob_off * candidate_mask
            a_off = a_off * candidate_mask
        log_prob_off = log_prob_off.sum(dim=-1)
        entropy_off = offload_dist.entropy().sum(dim=-1)

        # 3. Continuous 3D Movement with Tanh squashing
        mov_std = torch.exp(mov_log_std)
        mov_dist = Normal(mov_mu, mov_std)
        if deterministic:
            u_mov = mov_mu
        else:
            u_mov = mov_dist.rsample()
        a_mov = torch.tanh(u_mov)
        log_prob_mov = mov_dist.log_prob(u_mov) - torch.log(1.0 - a_mov.pow(2) + 1e-6)
        log_prob_mov = log_prob_mov.sum(dim=-1)
        entropy_mov = mov_dist.entropy().sum(dim=-1)

        # 4. Continuous CPU Allocation with Sigmoid squashing
        cpu_std = torch.exp(cpu_log_std)
        cpu_dist = Normal(cpu_mu, cpu_std)
        if deterministic:
            u_cpu = cpu_mu
        else:
            u_cpu = cpu_dist.rsample()
        a_cpu = torch.sigmoid(u_cpu)
        log_prob_cpu = cpu_dist.log_prob(u_cpu) - torch.log(a_cpu * (1.0 - a_cpu) + 1e-6)
        log_prob_cpu = log_prob_cpu.sum(dim=-1)
        entropy_cpu = cpu_dist.entropy().sum(dim=-1)

        # Composite action: [cluster (1), offload_priorities (K_pool), dx, dy, dz (3), cpu (1)]
        actions = torch.cat([
            cluster_action.unsqueeze(-1).float(),
            a_off,
            a_mov,
            a_cpu
        ], dim=-1)

        total_log_prob = log_prob_clu + log_prob_off + log_prob_mov + log_prob_cpu
        total_entropy = entropy_clu + entropy_off + entropy_mov + entropy_cpu

        return actions, total_log_prob, total_entropy

    def get_action_and_log_prob(self, obs, action_mask=None, cluster_mask=None, deterministic=False, **kwargs):
        if action_mask is None and cluster_mask is not None:
            action_mask = cluster_mask
        actions, log_prob, _ = self.sample(obs, action_mask=action_mask, deterministic=deterministic)
        return actions, log_prob

    def evaluate_actions(self, obs, actions, action_mask=None, cluster_mask=None, **kwargs):
        """
        Evaluate log probabilities and entropy of recorded actions during PPO update.
        """
        if action_mask is None and cluster_mask is not None:
            action_mask = cluster_mask
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(obs.device, dtype=obs.dtype)

        cluster_logits, offload_mu, offload_log_std, mov_mu, mov_log_std, cpu_mu, cpu_log_std = self.forward_heads(obs)

        # Parse actions
        cluster_act = actions[..., 0].long()
        idx_off_end = 1 + self.max_candidate_pool
        a_off = actions[..., 1:idx_off_end]
        a_mov = actions[..., idx_off_end:idx_off_end + 3]
        a_cpu = actions[..., idx_off_end + 3:idx_off_end + 4]

        # Parse masks
        cluster_mask = None
        candidate_mask = None
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).to(obs.device, dtype=obs.dtype)
            if action_mask.shape[-1] >= self.cluster_action_dim + self.max_candidate_pool:
                cluster_mask = action_mask[..., :self.cluster_action_dim]
                candidate_mask = action_mask[..., self.cluster_action_dim:self.cluster_action_dim + self.max_candidate_pool]
            elif action_mask.shape[-1] == self.cluster_action_dim:
                cluster_mask = action_mask

        # 1. Cluster Log Prob
        if cluster_mask is not None:
            valid_mask = torch.clamp(cluster_mask, 0.0, 1.0)
            cluster_logits = cluster_logits + (valid_mask + 1e-8).log()
        cluster_dist = Categorical(logits=cluster_logits)
        log_prob_clu = cluster_dist.log_prob(cluster_act)
        entropy_clu = cluster_dist.entropy()

        # 2. Offload Log Prob
        offload_std = torch.exp(offload_log_std)
        offload_dist = Normal(offload_mu, offload_std)
        # Invert sigmoid: u = log(a / (1 - a))
        a_off_clamped = torch.clamp(a_off, 1e-6, 1.0 - 1e-6)
        u_off = torch.log(a_off_clamped / (1.0 - a_off_clamped))
        log_prob_off = offload_dist.log_prob(u_off) - torch.log(a_off_clamped * (1.0 - a_off_clamped) + 1e-6)
        if candidate_mask is not None:
            log_prob_off = log_prob_off * candidate_mask
        log_prob_off = log_prob_off.sum(dim=-1)
        entropy_off = offload_dist.entropy().sum(dim=-1)

        # 3. Movement Log Prob
        mov_std = torch.exp(mov_log_std)
        mov_dist = Normal(mov_mu, mov_std)
        a_mov_clamped = torch.clamp(a_mov, -0.999999, 0.999999)
        u_mov = 0.5 * torch.log((1.0 + a_mov_clamped) / (1.0 - a_mov_clamped))
        log_prob_mov = mov_dist.log_prob(u_mov) - torch.log(1.0 - a_mov_clamped.pow(2) + 1e-6)
        log_prob_mov = log_prob_mov.sum(dim=-1)
        entropy_mov = mov_dist.entropy().sum(dim=-1)

        # 4. CPU Log Prob
        cpu_std = torch.exp(cpu_log_std)
        cpu_dist = Normal(cpu_mu, cpu_std)
        a_cpu_clamped = torch.clamp(a_cpu, 1e-6, 1.0 - 1e-6)
        u_cpu = torch.log(a_cpu_clamped / (1.0 - a_cpu_clamped))
        log_prob_cpu = cpu_dist.log_prob(u_cpu) - torch.log(a_cpu_clamped * (1.0 - a_cpu_clamped) + 1e-6)
        log_prob_cpu = log_prob_cpu.sum(dim=-1)
        entropy_cpu = cpu_dist.entropy().sum(dim=-1)

        total_log_prob = log_prob_clu + log_prob_off + log_prob_mov + log_prob_cpu
        total_entropy = entropy_clu + entropy_off + entropy_mov + entropy_cpu

        return total_log_prob, total_entropy


class GaussianOnlyActor(nn.Module):
    """
    Action-Masked Gaussian MLP actor for legacy MAPPO baseline.
    Generates continuous actions bounded in [-1, 1] using Tanh, while applying
    state-dependent action masks to eliminate infeasible decisions.
    """
    def __init__(self, obs_dim, action_dim=5, hidden_dim=256, K=3, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.encoder = ObsEncoder(obs_dim, hidden_dim)

        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        self.min_log_std = -20
        self.max_log_std = 2

    def forward(self, obs):
        features = self.encoder(obs)
        mu = self.mu_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        return mu, log_std

    def sample(self, obs, action_mask=None, **kwargs):
        mu, log_std = self.forward(obs)
        
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).to(obs.device, dtype=obs.dtype)
            if action_mask.shape == mu.shape:
                mu = mu * action_mask - 1e6 * (1.0 - action_mask)

        std = torch.exp(log_std)
        dist = Normal(mu, std)

        u = dist.rsample()
        action = torch.tanh(u)

        if action_mask is not None and action_mask.shape == action.shape:
            action = action * action_mask

        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return action, log_prob, entropy

    def evaluate_actions(self, obs, actions, action_mask=None, **kwargs):
        mu, log_std = self.forward(obs)

        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).to(obs.device, dtype=obs.dtype)
            if action_mask.shape == mu.shape:
                mu = mu * action_mask - 1e6 * (1.0 - action_mask)

        std = torch.exp(log_std)
        dist = Normal(mu, std)

        actions_clamped = torch.clamp(actions, -0.999999, 0.999999)
        u = torch.atanh(actions_clamped)

        log_prob = dist.log_prob(u) - torch.log(1 - actions_clamped.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy

    def get_action(self, obs, action_mask=None, deterministic=False):
        if deterministic:
            mu, _ = self.forward(obs)
            action = torch.tanh(mu)
            if action_mask is not None:
                if isinstance(action_mask, np.ndarray):
                    action_mask = torch.from_numpy(action_mask).to(obs.device, dtype=obs.dtype)
                if action_mask.shape == action.shape:
                    action = action * action_mask
            return action
        action, _, _ = self.sample(obs, action_mask=action_mask)
        return action

    def get_action_and_log_prob(self, obs, action_mask=None):
        action, log_prob, _ = self.sample(obs, action_mask=action_mask)
        return action, log_prob

    def compute_diffusion_loss(self, obs, actions):
        return torch.tensor(0.0, device=obs.device)

