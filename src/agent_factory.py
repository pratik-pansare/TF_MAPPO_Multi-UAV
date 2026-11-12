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
from src.baselines.local_agent import LocalExecutionAgent
from src.baselines.ippo_agent import IPPOAgent
from src.baselines.random_agent import RandomFeasibleAgent


def make_agent(method, obs_dim, state_dim, action_dim, n_agents, device='cpu', **kwargs):
    """
    Factory function instantiating agent specified by method string.
    """
    method_upper = method.upper()

    if any(k in method_upper for k in ["NOMASK", "NO_MASK"]):
        agent = MAPPOAgent(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            device=device,
            share_actor=True,
            **kwargs
        )
        agent.METHOD = "MAPPO_NoMask"
        return agent

    elif "IPPO" in method_upper:
        return IPPOAgent(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            device=device,
            share_actor=True,
            **kwargs
        )

    elif "RANDOM" in method_upper:
        return RandomFeasibleAgent(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            device=device,
            **kwargs
        )

    elif any(k in method_upper for k in ["PROPOSED", "MAPPO", "TF-MAPPO", "DMJO"]):
        return MAPPOAgent(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            device=device,
            share_actor=True,
            **kwargs
        )

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
        return GreedyOffloadingAgent(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            device=device
        )

    elif method_upper in ["LOCAL", "LOCALONLY", "LOCALEXECUTION"]:
        return LocalExecutionAgent(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            device=device
        )

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
