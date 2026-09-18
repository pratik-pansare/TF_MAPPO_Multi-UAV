"""
Agent Factory — Single-Point Switcher for Baseline Methods
===========================================================
Paper: Trust- and Fault-Aware Action-Masked MAPPO for Dynamic Multi-UAV
       Enabled IoT Mobile Edge Computing with Adaptive Task Offloading
       and Trajectory Optimization
Target Journal: IEEE Internet of Things Journal

Supported Comparison Baselines:
  1. MAPPO / TF-MAPPO   - Proposed Trust- and Fault-Aware Action-Masked MAPPO
  2. MAPPO_NoMask       - MAPPO without action masking
  3. MAPPO_NoTrust      - MAPPO without trust and fault awareness
  4. MADDPG             - Multi-Agent Deep Deterministic Policy Gradient
  5. MATD3              - Multi-Agent Twin Delayed DDPG
  6. GreedyOffloading   - Nearest-UAV Greedy Offloading (No Learning)
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mappo_agent import MAPPOAgent
from src.baselines.maddpg_agent import MADDPGAgent
from src.baselines.matd3_agent import MATD3Agent
from src.baselines.greedy_agent import GreedyOffloadingAgent


def make_agent(method, obs_dim, state_dim, action_dim, n_agents, device='cpu'):
    """
    Factory function instantiating agent specified by method string.
    """
    method_upper = method.upper()

    if method_upper in ["MAPPO", "TF-MAPPO", "DMJO"]:
        return MAPPOAgent(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            device=device,
            share_actor=True
        )

    elif method_upper in ["MAPPO_NOMASK", "NOMASK"]:
        agent = MAPPOAgent(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            device=device,
            share_actor=True
        )
        agent.METHOD = "MAPPO_NoMask"
        return agent

    elif method_upper in ["MADDPG"]:
        return MADDPGAgent(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            device=device
        )

    elif method_upper in ["MATD3"]:
        return MATD3Agent(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            device=device
        )

    elif method_upper in ["GREEDY", "GREEDYOFFLOADING"]:
        return GreedyOffloadingAgent(n_agents=n_agents, action_dim=action_dim)

    else:
        # Fallback default to Action-Masked MAPPO
        return MAPPOAgent(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            device=device,
            share_actor=True
        )
