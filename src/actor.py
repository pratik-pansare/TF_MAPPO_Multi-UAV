"""
Action-Masked Gaussian Actor for MAPPO
=======================================
Standard Gaussian MLP actor for multi-agent policy optimization with
State-Dependent Action Masking support.
Maps local observations to continuous mean and std parameters, sampling
feasible actions bounded by Tanh and action masks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class ObsEncoder(nn.Module):
    """Shared observation feature encoder for MAPPO."""
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class GaussianOnlyActor(nn.Module):
    """
    Action-Masked Gaussian MLP actor for MAPPO.
    Generates continuous actions bounded in [-1, 1] using Tanh, while applying
    state-dependent action masks to eliminate infeasible decisions.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=256, K=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.encoder = ObsEncoder(obs_dim, hidden_dim)

        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Initialize std range bounds
        self.min_log_std = -20
        self.max_log_std = 2

    def forward(self, obs):
        features = self.encoder(obs)
        mu = self.mu_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        return mu, log_std

    def sample(self, obs, action_mask=None):
        mu, log_std = self.forward(obs)
        
        # Apply action mask to mean if provided
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).to(obs.device, dtype=obs.dtype)
            # Mask infeasible action components
            if action_mask.shape == mu.shape:
                mu = mu * action_mask - 1e6 * (1.0 - action_mask)

        std = torch.exp(log_std)
        dist = Normal(mu, std)

        u = dist.rsample()
        action = torch.tanh(u)

        # Apply hard mask to sampled output if mask shape matches
        if action_mask is not None and action_mask.shape == action.shape:
            action = action * action_mask

        # Log prob calculation with Tanh correction formula
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return action, log_prob, entropy

    def evaluate_actions(self, obs, actions, action_mask=None):
        mu, log_std = self.forward(obs)

        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).to(obs.device, dtype=obs.dtype)
            if action_mask.shape == mu.shape:
                mu = mu * action_mask - 1e6 * (1.0 - action_mask)

        std = torch.exp(log_std)
        dist = Normal(mu, std)

        # Inverse Tanh mapping
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
        """No-op fallback for backwards compatibility."""
        return torch.tensor(0.0, device=obs.device)
