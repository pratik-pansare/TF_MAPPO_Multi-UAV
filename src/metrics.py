"""
Metrics & Performance Evaluation Manager
=========================================
Target Journal: IEEE Internet of Things Journal
Paper: Trust- and Fault-Aware Action-Masked MAPPO for Dynamic Multi-UAV Enabled
       IoT Mobile Edge Computing with Adaptive Task Offloading and Trajectory Optimization

Tracks all 8 paper performance metrics across evaluation episodes:
  1. Average Task Processing Latency (seconds)
  2. UAV Fleet Energy Consumption (Joules)
  3. Deadline Task Completion Ratio eta^{task} (%)
  4. Total UAV Fleet Trajectory Distance (meters)
  5. Number of UAV Failures Recovered
  6. Average Failure Recovery Time (slots)
  7. Trust Violation Events Blocked by Action Masking
  8. Reward Convergence & 95% Confidence Interval
"""

import numpy as np
import json
import os


class MetricTracker:
    """Tracks and computes statistical summaries for paper evaluation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.episode_latencies = []
        self.episode_energies = []
        self.tasks_generated = 0
        self.tasks_completed = 0
        self.trajectory_distances = []
        self.failures_recovered = 0
        self.recovery_times = []
        self.trust_violations_blocked = 0
        self.episode_rewards = []
        self.prev_uav_positions = None

    def update_step(self, env, rewards, info):
        """Update metric counters for one environment timestep."""
        step_lat = info.get('total_latency', 0.0) / max(env.num_users, 1)
        self.episode_latencies.append(step_lat)

        step_energy = info.get('total_energy', 0.0)
        self.episode_energies.append(step_energy)

        self.tasks_completed += info.get('completed_tasks', 0)
        self.tasks_generated += max(info.get('completed_tasks', 0) + info.get('deadline_violations', 0), 1)

        # UAV Trajectory Distance
        curr_positions = np.array([u.position[:2].copy() for u in env.uavs])
        if self.prev_uav_positions is not None:
            step_dist = np.sum(np.linalg.norm(curr_positions - self.prev_uav_positions, axis=1))
            self.trajectory_distances.append(step_dist)
        self.prev_uav_positions = curr_positions

        # Trust & Fault Metrics
        if hasattr(env, 'trust_manager'):
            tm = env.trust_manager
            self.failures_recovered = tm.failures_recovered
            self.recovery_times = tm.recovery_times
            self.trust_violations_blocked = tm.trust_violations_blocked

    def get_episode_summary(self):
        """Compute statistical episode metrics."""
        avg_lat = np.mean(self.episode_latencies) if len(self.episode_latencies) > 0 else 0.0
        tot_energy = np.sum(self.episode_energies) if len(self.episode_energies) > 0 else 0.0
        comp_ratio = (self.tasks_completed / max(self.tasks_generated, 1)) * 100.0
        traj_dist = np.sum(self.trajectory_distances) if len(self.trajectory_distances) > 0 else 0.0
        mean_rec_time = np.mean(self.recovery_times) if len(self.recovery_times) > 0 else 0.0

        return {
            'avg_latency': float(avg_lat),
            'total_energy': float(tot_energy),
            'completion_ratio_pct': float(comp_ratio),
            'trajectory_distance_m': float(traj_dist),
            'failures_recovered': int(self.failures_recovered),
            'avg_recovery_time_slots': float(mean_rec_time),
            'trust_violations_blocked': int(self.trust_violations_blocked),
        }

    @staticmethod
    def compute_confidence_interval(data, confidence=0.95):
        """Compute mean and 95% Confidence Interval bounds (mean +/- 1.96 * std / sqrt(N))."""
        arr = np.array(data, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        n = max(len(arr), 1)
        ci95 = float(1.96 * std / np.sqrt(n))
        return {
            'mean': mean,
            'std': std,
            'ci95': ci95,
            'low_ci': mean - ci95,
            'high_ci': mean + ci95,
        }
