"""
Training Script — Trust- and Fault-Aware Action-Masked MAPPO
============================================================
Target Journal: IEEE Internet of Things Journal
Paper Title: Trust- and Fault-Aware Action-Masked MAPPO for Dynamic Multi-UAV Enabled
             IoT Mobile Edge Computing with Adaptive Task Offloading and Trajectory Optimization

Runs multi-seed training (5 random seeds by default) and computes
mean +/- 95% Confidence Intervals for all paper metrics.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as _cfg
from src.env import MultiUAVMECEnv
from src.agent_factory import make_agent
from src.buffer import RolloutBuffer
from src.metrics import MetricTracker


def train_single_seed(args, seed, method="MAPPO"):
    """
    Train agent on a single random seed.
    """
    print(f"\n{'='*70}")
    print(f"  STARTING TRAINING [Seed {seed}] | Method: {method}")
    print(f"  Configuration : {args.num_uavs} UAVs, {args.num_iot} IoT Devices, {args.total_steps:,} Steps")
    print(f"{'='*70}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = MultiUAVMECEnv(num_users=args.num_iot, num_uavs=args.num_uavs, seed=seed)
    agent = make_agent(
        method=method,
        obs_dim=env.local_obs_dim,
        state_dim=env.global_state_dim,
        action_dim=env.agent_action_dim,
        n_agents=env.n_agents,
        device=args.device
    )

    buffer = RolloutBuffer(
        rollout_length=_cfg.MAPPO_ROLLOUT_LENGTH,
        n_agents=env.n_agents,
        obs_dim=env.local_obs_dim,
        state_dim=env.global_state_dim,
        action_dim=env.agent_action_dim,
        action_mask_dim=env.agent_action_dim
    )

    tracker = MetricTracker()
    obs, state = env.reset(seed=seed)

    out_dir = os.path.join(_cfg.CHECKPOINT_DIR, f"{method}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    global_step = 0
    episode_count = 0
    best_reward = -1e9
    start_time = time.time()

    while global_step < args.total_steps:
        # Collect rollout of MAPPO_ROLLOUT_LENGTH steps
        for step in range(_cfg.MAPPO_ROLLOUT_LENGTH):
            global_step += 1

            # Generate feasible state-dependent action mask M_m(t)
            action_mask = env.get_action_mask()

            # Sample feasible actions
            actions, log_probs, values = agent.get_actions(obs, state, action_mask=action_mask)

            # Step environment
            next_obs, next_state, rewards, done, info = env.step(actions)

            # Store transition in Rollout Buffer
            buffer.insert(obs, state, actions, log_probs, rewards, float(done), values, action_masks=action_mask)

            # Track step metrics
            tracker.update_step(env, rewards, info)

            obs, state = next_obs, next_state

            if done:
                episode_count += 1
                obs, state = env.reset()

            if global_step >= args.total_steps:
                break

        # Compute GAE and update policy
        with torch.no_grad():
            last_values = agent.get_values(state)
        buffer.compute_gae(last_values, float(done))

        update_info = agent.update(buffer)
        buffer.reset()

        # Logging & Checkpointing
        if global_step % _cfg.LOG_INTERVAL < _cfg.MAPPO_ROLLOUT_LENGTH:
            summary = tracker.get_episode_summary()
            mean_ep_reward = update_info.get('actor_loss', 0.0)
            elapsed = time.time() - start_time
            sps = global_step / max(elapsed, 1.0)

            print(f"  [Seed {seed} | Step {global_step:7,d}/{args.total_steps:,d}] "
                  f"SPS: {sps:4.0f} | Latency: {summary['avg_latency']:5.3f}s | "
                  f"Task Ratio: {summary['completion_ratio_pct']:5.1f}% | "
                  f"Energy: {summary['total_energy']:7.1f}J")

            if mean_ep_reward > best_reward:
                best_reward = mean_ep_reward
                agent.save(os.path.join(out_dir, "best_model.pt"))

    agent.save(os.path.join(out_dir, "final_model.pt"))
    final_summary = tracker.get_episode_summary()
    print(f"  Finished Seed {seed}! Final Completion Ratio: {final_summary['completion_ratio_pct']:.2f}%\n")
    return final_summary


def train_multi_seeds(args):
    """
    Train across multiple random seeds and compute 95% Confidence Intervals.
    """
    n_seeds = args.seeds
    seed_list = [args.seed + i for i in range(n_seeds)]

    print("=" * 75)
    print("   TRUST- AND FAULT-AWARE ACTION-MASKED MAPPO MULTI-SEED TRAINING   ")
    print("=" * 75)
    print(f"  Method            : {args.method}")
    print(f"  Seeds ({n_seeds})       : {seed_list}")
    print(f"  Total Steps/Seed  : {args.total_steps:,}")
    print("=" * 75)

    all_seed_results = []

    for seed in seed_list:
        summary = train_single_seed(args, seed, method=args.method)
        all_seed_results.append(summary)

    # Compute 95% Confidence Intervals across seeds
    latencies = [r['avg_latency'] for r in all_seed_results]
    energies = [r['total_energy'] for r in all_seed_results]
    ratios = [r['completion_ratio_pct'] for r in all_seed_results]
    distances = [r['trajectory_distance_m'] for r in all_seed_results]

    lat_ci = MetricTracker.compute_confidence_interval(latencies)
    eng_ci = MetricTracker.compute_confidence_interval(energies)
    rat_ci = MetricTracker.compute_confidence_interval(ratios)
    dist_ci = MetricTracker.compute_confidence_interval(distances)

    summary_results = {
        'method': args.method,
        'num_seeds': n_seeds,
        'total_steps_per_seed': args.total_steps,
        'metrics_ci95': {
            'latency_s': lat_ci,
            'energy_j': eng_ci,
            'completion_ratio_pct': rat_ci,
            'trajectory_distance_m': dist_ci,
        }
    }

    out_json = os.path.join(_cfg.LOG_DIR, f"{args.method}_multi_seed_results.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary_results, f, indent=4)

    print("\n" + "=" * 75)
    print("  FINAL MULTI-SEED PERFORMANCE SUMMARY (Mean +/- 95% CI):")
    print(f"    - Task Completion Latency   : {lat_ci['mean']:.4f} +/- {lat_ci['ci95']:.4f} s")
    print(f"    - UAV Fleet Energy          : {eng_ci['mean']:.2f} +/- {eng_ci['ci95']:.2f} J")
    print(f"    - Task Completion Ratio     : {rat_ci['mean']:.2f}% +/- {rat_ci['ci95']:.2f}%")
    print(f"    - UAV Trajectory Distance   : {dist_ci['mean']:.2f} +/- {dist_ci['ci95']:.2f} m")
    print(f"  Results saved to: {out_json}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Action-Masked MAPPO for IEEE IoT-J Paper")
    parser.add_argument("--method", type=str, default="MAPPO", help="MAPPO, MAPPO_NoMask, MADDPG, MATD3, Greedy")
    parser.add_argument("--total-steps", type=int, default=500000, help="Total environment steps per seed")
    parser.add_argument("--num-uavs", type=int, default=_cfg.NUM_UAVS, help="Number of UAVs")
    parser.add_argument("--num-iot", type=int, default=_cfg.NUM_USERS, help="Number of IoT devices")
    parser.add_argument("--seeds", type=int, default=1, help="Number of random seeds for 95%% CI")

    parser.add_argument("--seed", type=int, default=42, help="Initial random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device: cpu or cuda")
    parser.add_argument("--ablation", type=str, default="full", help="Ablation mode")

    args = parser.parse_args()
    train_multi_seeds(args)