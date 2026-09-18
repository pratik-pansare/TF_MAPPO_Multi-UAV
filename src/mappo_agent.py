"""
Action-Masked MAPPO Agent
===========================
Multi-Agent PPO with:
  - Decentralized Action-Masked Actor (Paper Section III.F & IV.E)
  - Centralised Critic with global state (CTDE paradigm)
  - GAE advantage estimation
  - PPO clipped objective with KL early stopping and entropy bonus
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MAPPO_LR_ACTOR, MAPPO_LR_CRITIC, MAPPO_CLIP_EPS,
    MAPPO_ENTROPY_COEF, MAPPO_VALUE_COEF, MAPPO_MAX_GRAD_NORM,
    MAPPO_PPO_EPOCHS, MAPPO_NUM_MINI_BATCHES,
    MAPPO_HIDDEN_DIM, MAPPO_CRITIC_HIDDEN_DIM,
)
from src.actor import GaussianOnlyActor
from src.buffer import RolloutBuffer

_ActorClass = GaussianOnlyActor


class CentralisedCritic(nn.Module):
    """
    Centralised critic for CTDE: global state -> V(s) per agent.
    Shared across all agents (parameter sharing).
    """
    def __init__(self, state_dim, n_agents, hidden_dim=MAPPO_CRITIC_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_agents),
        )

    def forward(self, state):
        return self.net(state)


class MAPPOAgent:
    """
    Action-Masked MAPPO agent under CTDE paradigm.
    """

    METHOD = "MAPPO"

    def __init__(self, obs_dim, state_dim, action_dim, n_agents,
                 device='cpu', share_actor=True):
        self.obs_dim     = obs_dim
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.n_agents    = n_agents
        self.device      = device
        self.share_actor = share_actor

        if share_actor:
            self.actor  = _ActorClass(
                obs_dim, action_dim,
                hidden_dim=MAPPO_HIDDEN_DIM,
            ).to(device)
            self.actors    = [self.actor] * n_agents
            actor_params   = list(self.actor.parameters())
        else:
            self.actors = [
                _ActorClass(
                    obs_dim, action_dim,
                    hidden_dim=MAPPO_HIDDEN_DIM,
                ).to(device)
                for _ in range(n_agents)
            ]
            self.actor   = self.actors[0]
            actor_params = []
            for a in self.actors:
                actor_params.extend(list(a.parameters()))

        self.critic = CentralisedCritic(
            state_dim, n_agents, hidden_dim=MAPPO_CRITIC_HIDDEN_DIM
        ).to(device)

        if share_actor:
            self.actor_optimizer  = optim.Adam(actor_params, lr=MAPPO_LR_ACTOR)
            self.actor_optimizers = [self.actor_optimizer]
        else:
            self.actor_optimizers = [
                optim.Adam(list(a.parameters()), lr=MAPPO_LR_ACTOR)
                for a in self.actors
            ]
            self.actor_optimizer = self.actor_optimizers[0]

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=MAPPO_LR_CRITIC)

        self.total_updates = 0

    @torch.inference_mode()
    def get_actions(self, obs_all, state, action_mask=None, num_active=None):
        """
        Select feasible actions for all agents using action masks.
        """
        n_active  = num_active or self.n_agents
        actions   = np.zeros((self.n_agents, self.action_dim))
        log_probs = np.zeros(self.n_agents)

        if self.share_actor:
            obs_batch = torch.from_numpy(
                np.ascontiguousarray(obs_all[:n_active], dtype=np.float32)
            ).to(self.device, non_blocking=True)

            mask_batch = None
            if action_mask is not None:
                mask_batch = torch.from_numpy(
                    np.ascontiguousarray(action_mask[:n_active], dtype=np.float32)
                ).to(self.device, non_blocking=True)

            a_batch, lp_batch    = self.actor.get_action_and_log_prob(obs_batch, action_mask=mask_batch)
            actions[:n_active]   = a_batch.cpu().numpy()
            log_probs[:n_active] = lp_batch.cpu().numpy().flatten()
        else:
            for i in range(n_active):
                obs_t = torch.from_numpy(
                    np.ascontiguousarray(obs_all[i], dtype=np.float32)
                ).unsqueeze(0).to(self.device, non_blocking=True)
                
                mask_t = None
                if action_mask is not None:
                    mask_t = torch.from_numpy(
                        np.ascontiguousarray(action_mask[i], dtype=np.float32)
                    ).unsqueeze(0).to(self.device, non_blocking=True)

                a, lp        = self.actors[i % len(self.actors)].get_action_and_log_prob(obs_t, action_mask=mask_t)
                actions[i]   = a.cpu().numpy().flatten()
                log_probs[i] = lp.cpu().numpy().item()

        state_t = torch.from_numpy(
            np.ascontiguousarray(state, dtype=np.float32)
        ).unsqueeze(0).to(self.device, non_blocking=True)
        values = self.critic(state_t).cpu().numpy().flatten()

        return actions, log_probs, values

    @torch.inference_mode()
    def get_values(self, state):
        """Return V(s) from the centralised critic."""
        state_t = torch.from_numpy(
            np.ascontiguousarray(state, dtype=np.float32)
        ).unsqueeze(0).to(self.device, non_blocking=True)
        return self.critic(state_t).cpu().numpy().flatten()

    def update(self, buffer: RolloutBuffer):
        """
        Perform Action-Masked MAPPO PPO update.
        """
        self.total_updates += 1
        metrics = {
            'actor_loss':  0.0,
            'critic_loss': 0.0,
            'entropy':     0.0,
            'approx_kl':   0.0,
        }
        n_updates = 0

        for epoch in range(MAPPO_PPO_EPOCHS):
            kl_exceeded = False

            for batch in buffer.get_batches(MAPPO_NUM_MINI_BATCHES):
                obs          = batch['obs'].to(self.device, non_blocking=True)
                states       = batch['states'].to(self.device, non_blocking=True)
                actions      = batch['actions'].to(self.device, non_blocking=True)
                masks        = batch.get('action_masks', None)
                if masks is not None:
                    masks = masks.to(self.device, non_blocking=True)
                old_log_probs= batch['old_log_probs'].to(self.device, non_blocking=True)
                advantages   = batch['advantages'].to(self.device, non_blocking=True)
                returns      = batch['returns'].to(self.device, non_blocking=True)
                old_values   = batch['old_values'].to(self.device, non_blocking=True)

                # Normalize advantages per mini-batch
                adv_std  = advantages.std()
                adv_mean = advantages.mean()
                advantages_norm = (advantages - adv_mean) / (adv_std + 1e-8)

                # Evaluate actions under actor and masks
                log_probs, entropy = self.actor.evaluate_actions(obs, actions, action_mask=masks)
                log_probs = log_probs.squeeze(-1)
                entropy   = entropy.squeeze(-1)

                # PPO clipped surrogate ratio
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages_norm
                surr2 = torch.clamp(ratio, 1.0 - MAPPO_CLIP_EPS, 1.0 + MAPPO_CLIP_EPS) * advantages_norm
                actor_loss = -torch.min(surr1, surr2).mean() - MAPPO_ENTROPY_COEF * entropy.mean()

                # Centralised critic loss with value clipping
                v_pred      = self.critic(states)
                agent_ids   = batch['agent_ids']
                v_pred_agent= v_pred.gather(1, agent_ids.unsqueeze(1).to(self.device)).squeeze(1)

                v_clipped   = old_values + torch.clamp(
                    v_pred_agent - old_values, -MAPPO_CLIP_EPS, MAPPO_CLIP_EPS
                )
                vf_loss1    = F.mse_loss(v_pred_agent, returns)
                vf_loss2    = F.mse_loss(v_clipped, returns)
                critic_loss = 0.5 * torch.max(vf_loss1, vf_loss2)

                # Optimize Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), MAPPO_MAX_GRAD_NORM)
                self.actor_optimizer.step()

                # Optimize Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAPPO_MAX_GRAD_NORM)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs - log_probs).mean().item()
                    if approx_kl > 0.03:
                        kl_exceeded = True

                metrics['actor_loss']  += actor_loss.item()
                metrics['critic_loss'] += critic_loss.item()
                metrics['entropy']     += entropy.mean().item()
                metrics['approx_kl']   += approx_kl
                n_updates              += 1

                if kl_exceeded:
                    break

        for k in metrics:
            metrics[k] /= max(n_updates, 1)
        return metrics

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'critic':        self.critic.state_dict(),
            'critic_opt':    self.critic_optimizer.state_dict(),
            'actor_opt':     self.actor_optimizers[0].state_dict(),
            'total_updates': self.total_updates,
        }
        if self.share_actor:
            state['actor'] = self.actor.state_dict()
        else:
            for i, (a, opt) in enumerate(zip(self.actors, self.actor_optimizers)):
                state[f'actor_{i}']     = a.state_dict()
                state[f'actor_opt_{i}'] = opt.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.critic.load_state_dict(ckpt['critic'])
            self.critic_optimizer.load_state_dict(ckpt['critic_opt'])
            self.total_updates = ckpt.get('total_updates', 0)
            if self.share_actor:
                self.actor.load_state_dict(ckpt['actor'])
                self.actor_optimizers[0].load_state_dict(ckpt['actor_opt'])
            else:
                for i, (a, opt) in enumerate(zip(self.actors, self.actor_optimizers)):
                    a.load_state_dict(ckpt[f'actor_{i}'])
                    key = f'actor_opt_{i}' if f'actor_opt_{i}' in ckpt else ('actor_opt' if i == 0 else None)

                    if key:
                        opt.load_state_dict(ckpt[key])
            return True
        except Exception as e:
            print(f"  [Warning] Skipping incompatible checkpoint '{path}': {e}")
            return False