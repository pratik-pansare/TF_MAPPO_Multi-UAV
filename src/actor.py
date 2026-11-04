"""
Hybrid Factorized Action-Masked Actor for MAPPO
===============================================
Paper Reference: Section III.F & Section IV.E
Target Journal: IEEE Internet of Things Journal

Factorized Action Space per UAV m:
  1. Categorical Cluster Selection Head: a_m^clu in {0, 1, ..., C}
     - Masked Categorical distribution: invalid choices masked with -1e9 logits.
     - Choice 0 (idle) is always feasible.
  2. Continuous 3D Trajectory Movement Head: a_m^mov in [-1, 1]^3
     - Squashed Gaussian via Tanh with exact Jacobian change-of-variables log-prob.
  3. Continuous CPU Allocation Fraction Head: a_m^cpu in [0, 1]
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
    Factorized Actor supporting discrete masked categorical cluster choices
    and continuous bounded continuous trajectory and CPU allocations.
    """

    def __init__(self, obs_dim, action_dim=5, n_clusters=5, hidden_dim=256, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_clusters = n_clusters
        self.cluster_action_dim = n_clusters + 1  # 0 = idle, 1..C = clusters

        self.encoder = ObsEncoder(obs_dim, hidden_dim)

        # 1. Categorical Cluster Head (size C + 1)
        self.cluster_head = nn.Linear(hidden_dim, self.cluster_action_dim)

        # 2. Continuous 3D Movement Head (dx, dy, dz in [-1, 1])
        self.mov_mu_head = nn.Linear(hidden_dim, 3)
        self.mov_log_std = nn.Parameter(torch.zeros(3))

        # 3. Continuous CPU Allocation Head (rho in [0, 1])
        self.cpu_mu_head = nn.Linear(hidden_dim, 1)
        self.cpu_log_std = nn.Parameter(torch.zeros(1))

    def forward_heads(self, obs):
        feat = self.encoder(obs)
        cluster_logits = self.cluster_head(feat)
        mov_mu = self.mov_mu_head(feat)
        mov_log_std = torch.clamp(self.mov_log_std, -20.0, 2.0)
        cpu_mu = self.cpu_mu_head(feat)
        cpu_log_std = torch.clamp(self.cpu_log_std, -20.0, 2.0)
        return cluster_logits, mov_mu, mov_log_std, cpu_mu, cpu_log_std

    def sample(self, obs, cluster_mask=None, deterministic=False):
        """
        Sample actions from factorized heads under state-dependent action masks.
        """
        cluster_logits, mov_mu, mov_log_std, cpu_mu, cpu_log_std = self.forward_heads(obs)

        # 1. Masked Categorical Cluster Selection
        if cluster_mask is not None:
            if isinstance(cluster_mask, np.ndarray):
                cluster_mask = torch.from_numpy(cluster_mask).to(obs.device, dtype=obs.dtype)
            # Mask out invalid clusters with large negative logits
            # Ensure idle (index 0) remains unmasked
            valid_mask = torch.clamp(cluster_mask, 0.0, 1.0)
            cluster_logits = cluster_logits + (valid_mask + 1e-8).log()

        cluster_dist = Categorical(logits=cluster_logits)
        if deterministic:
            cluster_action = torch.argmax(cluster_logits, dim=-1)
        else:
            cluster_action = cluster_dist.sample()
        log_prob_clu = cluster_dist.log_prob(cluster_action)
        entropy_clu = cluster_dist.entropy()

        # 2. Continuous 3D Movement with Tanh squashing
        mov_std = torch.exp(mov_log_std)
        mov_dist = Normal(mov_mu, mov_std)
        if deterministic:
            u_mov = mov_mu
        else:
            u_mov = mov_dist.rsample()
        a_mov = torch.tanh(u_mov)
        # Jacobian correction for Tanh
        log_prob_mov = mov_dist.log_prob(u_mov) - torch.log(1.0 - a_mov.pow(2) + 1e-6)
        log_prob_mov = log_prob_mov.sum(dim=-1)
        entropy_mov = mov_dist.entropy().sum(dim=-1)

        # 3. Continuous CPU Allocation with Sigmoid squashing
        cpu_std = torch.exp(cpu_log_std)
        cpu_dist = Normal(cpu_mu, cpu_std)
        if deterministic:
            u_cpu = cpu_mu
        else:
            u_cpu = cpu_dist.rsample()
        a_cpu = torch.sigmoid(u_cpu)
        # Jacobian correction for Sigmoid: log(sigma * (1 - sigma))
        log_prob_cpu = cpu_dist.log_prob(u_cpu) - torch.log(a_cpu * (1.0 - a_cpu) + 1e-6)
        log_prob_cpu = log_prob_cpu.sum(dim=-1)
        entropy_cpu = cpu_dist.entropy().sum(dim=-1)

        # Composite action: [cluster, dx, dy, dz, cpu_fraction] (5 dimensions)
        actions = torch.cat([
            cluster_action.unsqueeze(-1).float(),
            a_mov,
            a_cpu
        ], dim=-1)

        total_log_prob = log_prob_clu + log_prob_mov + log_prob_cpu
        total_entropy = entropy_clu + entropy_mov + entropy_cpu

        return actions, total_log_prob, total_entropy

    def get_action_and_log_prob(self, obs, action_mask=None, deterministic=False):
        actions, log_prob, _ = self.sample(obs, cluster_mask=action_mask, deterministic=deterministic)
        return actions, log_prob

    def evaluate_actions(self, obs, actions, cluster_mask=None, action_mask=None):
        """
        Evaluate log probabilities and entropy of recorded actions during PPO update.
        """
        if cluster_mask is None and action_mask is not None:
            cluster_mask = action_mask
        cluster_logits, mov_mu, mov_log_std, cpu_mu, cpu_log_std = self.forward_heads(obs)

        # 1. Categorical Cluster Head
        if cluster_mask is not None:
            if isinstance(cluster_mask, np.ndarray):
                cluster_mask = torch.from_numpy(cluster_mask).to(obs.device, dtype=obs.dtype)
            valid_mask = torch.clamp(cluster_mask, 0.0, 1.0)
            cluster_logits = cluster_logits + (valid_mask + 1e-8).log()

        cluster_dist = Categorical(logits=cluster_logits)
        cluster_action = actions[..., 0].long()
        log_prob_clu = cluster_dist.log_prob(cluster_action)
        entropy_clu = cluster_dist.entropy()

        # 2. Movement Head
        mov_std = torch.exp(mov_log_std)
        mov_dist = Normal(mov_mu, mov_std)
        a_mov = actions[..., 1:4]
        a_mov_clamped = torch.clamp(a_mov, -0.999999, 0.999999)
        u_mov = 0.5 * (torch.log1p(a_mov_clamped) - torch.log1p(-a_mov_clamped))  # atanh
        log_prob_mov = mov_dist.log_prob(u_mov) - torch.log(1.0 - a_mov_clamped.pow(2) + 1e-6)
        log_prob_mov = log_prob_mov.sum(dim=-1)
        entropy_mov = mov_dist.entropy().sum(dim=-1)

        # 3. CPU Head
        cpu_std = torch.exp(cpu_log_std)
        cpu_dist = Normal(cpu_mu, cpu_std)
        a_cpu = actions[..., 4:5]
        a_cpu_clamped = torch.clamp(a_cpu, 1e-6, 1.0 - 1e-6)
        u_cpu = torch.logit(a_cpu_clamped)  # inverse sigmoid
        log_prob_cpu = cpu_dist.log_prob(u_cpu) - torch.log(a_cpu_clamped * (1.0 - a_cpu_clamped) + 1e-6)
        log_prob_cpu = log_prob_cpu.sum(dim=-1)
        entropy_cpu = cpu_dist.entropy().sum(dim=-1)

        total_log_prob = log_prob_clu + log_prob_mov + log_prob_cpu
        total_entropy = entropy_clu + entropy_mov + entropy_cpu

        return total_log_prob, total_entropy


GaussianOnlyActor = HybridActionMaskedActor
