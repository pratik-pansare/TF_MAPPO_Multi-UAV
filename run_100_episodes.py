"""
100-Episode Evaluation Framework for Paper Validation
======================================================
Paper Title: "Trust- and Fault-Aware Action-Masked MAPPO for Dynamic Multi-UAV Enabled
              IoT Mobile Edge Computing with Adaptive Task Offloading and Trajectory Optimization"
Target Journal: IEEE Internet of Things Journal

Runs 100 complete evaluation episodes (100 x 500 = 50,000 steps) using:
  1. Pre-deployment Workload-Aware IoT Clustering (Eq. 1-3)
  2. Certificate Authentication Trust tau_m(t), Operational Availability phi_m(t), Eligibility e_m(t) (Eq. 27-29)
  3. Action-Masked MAPPO Policy
  4. Multi-Objective Cost Minimisation (Problem P1)

Outputs saved to: results/test_results/paper_100_episodes/
"""

import sys
import os
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from src.env import MultiUAVMECEnv
from src.agent_factory import make_agent
from src.metrics import MetricTracker


def run_100_episodes(num_episodes=100, max_steps_per_episode=500, device='cpu'):
    """
    Run 100 evaluation episodes and calculate statistical mean +/- std and 95% CI
    for Trajectories, Energy Consumption, Task Latency, and Completion Ratio.
    """
    print("=" * 75)
    print("      100-EPISODE PAPER EVALUATION FRAMEWORK FOR MULTI-UAV MEC      ")
    print("=" * 75)
    print(f"  Configuration : {NUM_UAVS} UAVs, {NUM_USERS} IoT Devices, {NUM_CLUSTERS} Clusters")
    print(f"  Clustering    : Pre-deployment Workload-Aware K-Means (lambda_w={LAMBDA_W})")
    print(f"  Policy Model  : Action-Masked Gaussian MAPPO")
    print(f"  Evaluation    : {num_episodes} Episodes x {max_steps_per_episode} Slots")
    print("=" * 75)

    out_dir = os.path.join("results", "test_results", "paper_100_episodes")
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    env = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)

    agent = make_agent(
        method="MAPPO",
        obs_dim=env.local_obs_dim,
        state_dim=env.global_state_dim,
        action_dim=env.agent_action_dim,
        n_agents=env.n_agents,
        device=device
    )

    # Auto-load trained model checkpoint if available
    ckpt_candidates = [
        "results/checkpoints/MAPPO_seed42/best_model.pt",
        "results/checkpoints/DMJO/full/best_model.pt",
        "results/checkpoints/MAPPO/full/best_model.pt",
        "results/checkpoints/best_model.pt",
        "results/checkpoints/final_model.pt"
    ]
    loaded_ckpt = None
    for candidate in ckpt_candidates:
        if os.path.exists(candidate):
            if agent.load(candidate):
                loaded_ckpt = candidate
                print(f"  Loaded trained checkpoint: {candidate}")
                break

    if not loaded_ckpt:
        print("  Notice: No trained checkpoint found in results/checkpoints/. Using initial policy.")

    episode_rewards = []
    episode_latencies = []
    episode_energies = []
    episode_completion_ratios = []
    episode_traj_costs = []
    episode_eligibilities = []

    start_time = time.time()

    for ep in range(1, num_episodes + 1):
        obs, state = env.reset()
        ep_reward = 0.0
        ep_latency_sum = 0.0
        ep_energy_sum = 0.0
        ep_tasks_completed = 0
        ep_tasks_generated = 0
        ep_traj_dist = 0.0
        ep_eligibility_sum = 0.0

        prev_uav_pos = np.array([u.position.copy() for u in env.uavs[:env.num_active]])

        for t in range(max_steps_per_episode):
            action_mask = env.get_action_mask()
            actions, _, _ = agent.get_actions(obs, state, action_mask=action_mask)

            next_obs, next_state, rewards, done, info = env.step(actions)

            ep_reward += float(np.mean(rewards[:env.num_active]))

            slot_lat = info.get('total_latency', 0.0) / max(env.num_users, 1)
            ep_latency_sum += slot_lat

            slot_energy = info.get('total_energy', 0.0)
            ep_energy_sum += slot_energy

            curr_uav_pos = np.array([u.position.copy() for u in env.uavs[:env.num_active]])
            ep_traj_dist += np.sum(np.linalg.norm(curr_uav_pos[:, :2] - prev_uav_pos[:, :2], axis=1))
            prev_uav_pos = curr_uav_pos

            eligibility_ratio = np.mean(env.trust_manager.eligibility[:env.num_active])
            ep_eligibility_sum += eligibility_ratio

            ep_tasks_completed += info.get('completed_tasks', 0)
            ep_tasks_generated += max(info.get('completed_tasks', 0) + info.get('deadline_violations', 0), 1)

            obs, state = next_obs, next_state

        avg_ep_latency = ep_latency_sum / max_steps_per_episode
        avg_ep_energy = ep_energy_sum / max_steps_per_episode
        completion_ratio = ep_tasks_completed / max(ep_tasks_generated, 1)
        avg_eligibility = ep_eligibility_sum / max_steps_per_episode

        episode_rewards.append(ep_reward)
        episode_latencies.append(avg_ep_latency)
        episode_energies.append(avg_ep_energy)
        episode_completion_ratios.append(completion_ratio)
        episode_traj_costs.append(ep_traj_dist)
        episode_eligibilities.append(avg_eligibility)

        if ep % 10 == 0 or ep == 1:
            print(f"  Episode {ep:3d}/{num_episodes} | "
                  f"Reward: {ep_reward:7.2f} | "
                  f"Latency: {avg_ep_latency:5.3f}s | "
                  f"UAV Energy: {avg_ep_energy:7.1f}J | "
                  f"Task Ratio: {completion_ratio*100:5.1f}% | "
                  f"Eligibility: {avg_eligibility*100:5.1f}%")

    elapsed = time.time() - start_time
    print("=" * 75)
    print(f"  Completed 100 Episodes in {elapsed:.2f} seconds!")
    print("=" * 75)

    results_summary = {
        "num_episodes": num_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "metrics": {
            "reward": MetricTracker.compute_confidence_interval(episode_rewards),
            "task_latency_s": MetricTracker.compute_confidence_interval(episode_latencies),
            "uav_energy_j": MetricTracker.compute_confidence_interval(episode_energies),
            "completion_ratio_pct": MetricTracker.compute_confidence_interval(np.array(episode_completion_ratios) * 100),
            "trajectory_distance_m": MetricTracker.compute_confidence_interval(episode_traj_costs),
            "uav_eligibility_pct": MetricTracker.compute_confidence_interval(np.array(episode_eligibilities) * 100),
        }
    }

    print("\n  SUMMARY OF RESULTS ACROSS 100 EPISODES (Mean +/- 95% CI):")
    print(f"    - Task Completion Latency   : {results_summary['metrics']['task_latency_s']['mean']:.4f} +/- {results_summary['metrics']['task_latency_s']['ci95']:.4f} s")
    print(f"    - UAV Fleet Energy          : {results_summary['metrics']['uav_energy_j']['mean']:.2f} +/- {results_summary['metrics']['uav_energy_j']['ci95']:.2f} J")
    print(f"    - Task Completion Ratio     : {results_summary['metrics']['completion_ratio_pct']['mean']:.2f}% +/- {results_summary['metrics']['completion_ratio_pct']['ci95']:.2f}%")
    print(f"    - UAV Trajectory Distance   : {results_summary['metrics']['trajectory_distance_m']['mean']:.2f} +/- {results_summary['metrics']['trajectory_distance_m']['ci95']:.2f} m")
    print(f"    - UAV Eligibility State     : {results_summary['metrics']['uav_eligibility_pct']['mean']:.2f}% +/- {results_summary['metrics']['uav_eligibility_pct']['ci95']:.2f}%")

    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(results_summary, f, indent=4)
    print(f"\n  Saved summary metrics to: {json_path}")

    # Render Summary Chart
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    episodes_axis = np.arange(1, num_episodes + 1)

    axs[0, 0].plot(episodes_axis, episode_latencies, color='#E07020', linewidth=1.5, label='Latency (s)')
    axs[0, 0].axhline(np.mean(episode_latencies), color='#888888', linestyle='--', label=f"Mean: {np.mean(episode_latencies):.3f}s")
    axs[0, 0].set_title('Task Completion Latency (s)', fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Latency (seconds)')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    axs[0, 1].plot(episodes_axis, episode_energies, color='#2CA02C', linewidth=1.5, label='Energy (J)')
    axs[0, 1].axhline(np.mean(episode_energies), color='#888888', linestyle='--', label=f"Mean: {np.mean(episode_energies):.1f}J")
    axs[0, 1].set_title('UAV Fleet Energy Expenditure (J)', fontsize=11, fontweight='bold')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Energy (Joules)')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    axs[1, 0].plot(episodes_axis, np.array(episode_completion_ratios) * 100, color='#1F77B4', linewidth=1.5, label='Completion Ratio (%)')
    axs[1, 0].axhline(np.mean(episode_completion_ratios) * 100, color='#888888', linestyle='--', label=f"Mean: {np.mean(episode_completion_ratios)*100:.1f}%")
    axs[1, 0].set_title('Deadline Task Completion Ratio (%)', fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Completion Rate (%)')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    axs[1, 1].plot(episodes_axis, episode_traj_costs, color='#9467BD', linewidth=1.5, label='Traj Distance (m)')
    axs[1, 1].axhline(np.mean(episode_traj_costs), color='#888888', linestyle='--', label=f"Mean: {np.mean(episode_traj_costs):.1f}m")
    axs[1, 1].set_title('UAV Fleet Trajectory Distance (m)', fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Total Distance (m)')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "paper_100_episodes_summary.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved 4-panel summary plot to: {plot_path}\n")

    return results_summary


if __name__ == "__main__":
    run_100_episodes(100, max_steps_per_episode=500)
