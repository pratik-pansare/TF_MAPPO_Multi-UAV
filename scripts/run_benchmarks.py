"""
Baseline Benchmark Comparison Runner
====================================
Paper Reference: Section IV.E & Table I
Target Journal: IEEE Internet of Things Journal

Evaluates all compared methods on identical traffic realizations:
  1. TF-MAPPO (Proposed Action-Masked MAPPO)
  2. MAPPO_NoMask (Unmasked MAPPO)
  3. MAPPO_NoTrust (MAPPO without trust eligibility)
  4. Greedy (Dynamic Nearest-UAV Full-Capacity Heuristic)
  5. LocalOnly (Local-only compute floor)

Outputs saved to:
  - outputs/Milestone_4_baselines/benchmark_results.csv
  - results/logs/benchmark_comparison.json
"""

import sys
import os
import json
import csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.env import MultiUAVMECEnv
from src.agent_factory import make_agent
from src.metrics import MetricTracker


def evaluate_method(method_name, num_episodes=5, steps_per_episode=50, seed=42, device="cpu"):
    print(f"\n--- Evaluating Baseline: {method_name} ---")
    env = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=seed)
    agent = make_agent(
        method=method_name,
        obs_dim=env.local_obs_dim,
        state_dim=env.global_state_dim,
        action_dim=env.agent_action_dim,
        n_agents=env.n_agents,
        device=device
    )

    # Try loading best checkpoint for MAPPO if available
    if "MAPPO" in method_name:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"MAPPO_seed{seed}", "best_model.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"MAPPO_seed{seed}", "final_model.pt")
        if os.path.exists(ckpt_path):
            agent.load(ckpt_path)

    tracker = MetricTracker()
    episode_latencies = []
    episode_energies = []
    episode_ratios = []
    episode_distances = []

    for ep in range(1, num_episodes + 1):
        obs, state = env.reset(seed=seed + ep * 10)
        tracker.reset()

        for t in range(steps_per_episode):
            if method_name == "MAPPO_NoMask":
                # Unmasked: provide all-ones mask
                action_mask = np.ones((env.num_uavs, env.action_mask_dim), dtype=np.float32)
            elif method_name == "MAPPO_NoTrust":
                # No trust awareness: assume all UAVs eligible
                action_mask = np.ones((env.num_uavs, env.action_mask_dim), dtype=np.float32)
            else:
                action_mask = env.get_action_mask()

            actions, _, _ = agent.get_actions(obs, state, action_mask=action_mask)
            next_obs, next_state, rewards, done, info = env.step(actions)
            tracker.update_step(env, rewards, info)

            obs, state = next_obs, next_state
            if done:
                break

        summary = tracker.get_episode_summary()
        episode_latencies.append(summary['avg_latency'])
        episode_energies.append(summary['total_energy'])
        episode_ratios.append(summary['completion_ratio_pct'])
        episode_distances.append(summary['trajectory_distance_m'])

    lat_ci = MetricTracker.compute_confidence_interval(episode_latencies)
    eng_ci = MetricTracker.compute_confidence_interval(episode_energies)
    rat_ci = MetricTracker.compute_confidence_interval(episode_ratios)
    dist_ci = MetricTracker.compute_confidence_interval(episode_distances)

    res = {
        'method': method_name,
        'latency_s_mean': lat_ci['mean'],
        'latency_s_ci': lat_ci['ci95'],
        'energy_j_mean': eng_ci['mean'],
        'energy_j_ci': eng_ci['ci95'],
        'completion_ratio_mean': rat_ci['mean'],
        'completion_ratio_ci': rat_ci['ci95'],
        'flight_dist_m_mean': dist_ci['mean'],
    }
    print(f"  Latency: {res['latency_s_mean']:.3f} +/- {res['latency_s_ci']:.3f} s")
    print(f"  Energy : {res['energy_j_mean']:.1f} +/- {res['energy_j_ci']:.1f} J")
    print(f"  Ratio  : {res['completion_ratio_mean']:.2f}% +/- {res['completion_ratio_ci']:.2f}%")
    return res


def run_all_benchmarks(episodes=5, steps=50):
    methods = ["MAPPO", "MAPPO_NoMask", "MAPPO_NoTrust", "Greedy", "LocalOnly"]
    all_results = []

    print("=" * 70)
    print("      RUNNING IEEE MULTI-BASELINE BENCHMARK COMPARISONS      ")
    print("=" * 70)

    for m in methods:
        res = evaluate_method(m, num_episodes=episodes, steps_per_episode=steps)
        all_results.append(res)

    out_csv = "outputs/Milestone_4_baselines/benchmark_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved benchmark results to: {out_csv}")

    out_json = "results/logs/benchmark_comparison.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=4)

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    run_all_benchmarks(episodes=args.episodes, steps=args.steps)

