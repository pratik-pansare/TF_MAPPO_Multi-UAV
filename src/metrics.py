"""
Metrics & Performance Evaluation Manager
=========================================
Target Journal: IEEE Internet of Things Journal
Paper: Trust- and Fault-Aware Action-Masked MAPPO for Dynamic Multi-UAV Enabled
       IoT Mobile Edge Computing with Adaptive Task Offloading and Trajectory Optimization

Tracks all required paper evaluation metrics:
  1. Average Task Processing Latency (seconds)
  2. UAV Fleet Energy Consumption & Components (Propulsion, CPU, U2U in Joules)
  3. Deadline Task Completion Ratio eta^{task} = N_ddl / N_gen (%)
  4. Total UAV Fleet Trajectory Distance (meters)
  5. Safety Shield Interventions (continuous collision overrides)
  6. UAV Failures Recovered & Mean Recovery Time (slots)
  7. Trust Violation Events Blocked by Action Masking
  8. Step Reward Components Decomposition (reconciled in reward_components.csv)
  9. Mean +/- 95% Confidence Intervals across seeds
"""

import numpy as np
import json
import csv
import os


class MetricTracker:
    """Tracks and computes statistical summaries for paper evaluation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_latencies = []
        self.step_energies = []
        self.step_propulsion = []
        self.step_computation = []
        self.step_u2u = []
        self.step_rewards = []
        self.reward_components_log = []

        self.tasks_generated = 0
        self.tasks_completed = 0
        self.tasks_expired = 0
        self.active_links_history = []

        self.trajectory_distances = []
        self.shield_interventions = 0
        self.failures_recovered = 0
        self.recovery_times = []
        self.trust_violations_blocked = 0

        self.completed_task_latencies = []
        self.all_wait_delays = []
        self.all_upload_delays = []
        self.all_queue_delays = []
        self.all_execution_delays = []

        self.prev_uav_positions = None
        self.last_env = None

    def update_step(self, env, rewards, info):
        """Update metric counters for one environment transition."""
        self.last_env = env
        # 1. Latency & Tasks
        slot_comp = info.get('completed_tasks_step', 0)
        slot_exp = info.get('expired_tasks_step', 0)
        self.tasks_completed += slot_comp
        self.tasks_expired += slot_exp
        self.active_links_history.append(info.get('active_links_count', 0))

        # Count generated tasks from devices
        newly_gen = info.get('generated_tasks_step', sum(1 for d in env.devices if d.current_task is not None))
        self.tasks_generated += newly_gen

        if slot_comp > 0:
            self.step_latencies.append(info.get('mean_slot_latency', 0.0))
            for task in info.get('slot_completed_tasks', []):
                decomp = task.get_latency_decomposition() if hasattr(task, 'get_latency_decomposition') else {}
                self.completed_task_latencies.append(float(getattr(task, 'total_latency', decomp.get('total_latency', 0.0))))
                self.all_wait_delays.append(float(decomp.get('t_wait', getattr(task, 'wait_delay', 0.0))))
                self.all_upload_delays.append(float(decomp.get('t_up', getattr(task, 'upload_delay', 0.0))))
                self.all_queue_delays.append(float(decomp.get('t_que', getattr(task, 'queue_delay', 0.0))))
                self.all_execution_delays.append(float(decomp.get('t_exe', getattr(task, 'execution_delay', 0.0))))

        # 2. Energy
        self.step_energies.append(info.get('total_energy', 0.0))
        self.step_propulsion.append(info.get('propulsion_energy', 0.0))
        self.step_computation.append(info.get('computation_energy', 0.0))
        self.step_u2u.append(info.get('u2u_energy', 0.0))

        # 3. Trajectory Distance
        curr_positions = np.array([u.position.copy() for u in env.uavs])
        if self.prev_uav_positions is not None:
            step_dist = np.sum(np.linalg.norm(curr_positions - self.prev_uav_positions, axis=1))
            self.trajectory_distances.append(step_dist)
        self.prev_uav_positions = curr_positions

        # 4. Shield Interventions
        if info.get('shield_intervened', False):
            self.shield_interventions += 1

        # 5. Trust & Faults
        if hasattr(env, 'trust_manager'):
            self.trust_violations_blocked = env.trust_manager.trust_violations_blocked
        if hasattr(env, 'fault_manager'):
            self.failures_recovered = env.fault_manager.failures_recovered
            self.recovery_times = env.fault_manager.recovery_times

        # 6. Reward Components
        rc = info.get('reward_components', {})
        if rc:
            self.reward_components_log.append(rc)
            self.step_rewards.append(rc.get('total_step_reward', 0.0))

    def get_episode_summary(self):
        """Compute statistical summary for completed episode."""
        # Individual completed-task latency mean (NOT unweighted mean of per-slot averages)
        if len(self.completed_task_latencies) > 0:
            avg_lat = float(np.mean(self.completed_task_latencies))
        elif len(self.step_latencies) > 0:
            avg_lat = float(np.mean(self.step_latencies))
        else:
            avg_lat = 0.0

        tot_energy = float(np.sum(self.step_energies)) if len(self.step_energies) > 0 else 0.0
        tot_prop = float(np.sum(self.step_propulsion)) if len(self.step_propulsion) > 0 else 0.0
        tot_comp = float(np.sum(self.step_computation)) if len(self.step_computation) > 0 else 0.0
        tot_u2u = float(np.sum(self.step_u2u)) if len(self.step_u2u) > 0 else 0.0

        n_gen = max(self.tasks_generated, 1)
        comp_ratio = (self.tasks_completed / n_gen) * 100.0

        traj_dist = float(np.sum(self.trajectory_distances)) if len(self.trajectory_distances) > 0 else 0.0
        mean_rec_time = float(np.mean(self.recovery_times)) if len(self.recovery_times) > 0 else 0.0
        mean_reward = float(np.mean(self.step_rewards)) if len(self.step_rewards) > 0 else 0.0

        # Exact latency decomposition with zero guessed synthetic fallbacks
        avg_wait = float(np.mean(self.all_wait_delays)) if len(self.all_wait_delays) > 0 else 0.0
        avg_uplink = float(np.mean(self.all_upload_delays)) if len(self.all_upload_delays) > 0 else 0.0
        avg_queue = float(np.mean(self.all_queue_delays)) if len(self.all_queue_delays) > 0 else 0.0
        avg_cpu = float(np.mean(self.all_execution_delays)) if len(self.all_execution_delays) > 0 else 0.0

        # Mission-level Problem P1 objective from ledger if available
        t_svc = avg_lat
        eta_task = comp_ratio / 100.0
        if self.last_env is not None and hasattr(self.last_env, 'task_ledger') and len(self.last_env.task_ledger.records) > 0:
            t_svc, eta_task = self.last_env.task_ledger.compute_mission_service_latency_and_completion(t_miss=2.0)

        n_uavs = len(self.last_env.uavs) if self.last_env is not None else 4
        e_ref = max(n_uavs * 200.0 * 500.0, 1.0)  # Reference mission fleet energy
        t_ref = 2.0
        omega_e = 0.5
        omega_t = 0.5
        omega_d = 0.5
        p1_objective = omega_e * (tot_energy / e_ref) + omega_t * (t_svc / t_ref) + omega_d * (1.0 - eta_task)

        return {
            'avg_latency': avg_lat,
            'avg_wait_latency': avg_wait,
            'avg_uplink_latency': avg_uplink,
            'avg_queue_latency': avg_queue,
            'avg_cpu_latency': avg_cpu,
            't_svc': float(t_svc),
            'p1_objective': float(p1_objective),
            'total_energy': tot_energy,
            'propulsion_energy': tot_prop,
            'computation_energy': tot_comp,
            'u2u_energy': tot_u2u,
            'completion_ratio_pct': float(comp_ratio),
            'tasks_generated': int(self.tasks_generated),
            'tasks_completed': int(self.tasks_completed),
            'tasks_expired': int(self.tasks_expired),
            'avg_active_links': float(np.mean(self.active_links_history)) if self.active_links_history else 0.0,
            'trajectory_distance_m': traj_dist,
            'shield_interventions': int(self.shield_interventions),
            'failures_recovered': int(self.failures_recovered),
            'avg_recovery_time_slots': mean_rec_time,
            'trust_violations_blocked': int(self.trust_violations_blocked),
            'mean_step_reward': mean_reward,
        }

    def export_reward_components_csv(self, file_path):
        """Export reward reconciliation components to CSV."""
        if not self.reward_components_log:
            return
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        keys = list(self.reward_components_log[0].keys())
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.reward_components_log)

    @staticmethod
    def compute_confidence_interval(data, confidence=0.95):
        """Compute sample mean and 95% Confidence Interval bounds."""
        arr = np.array(data, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else float(np.std(arr))
        n = max(len(arr), 1)
        ci95 = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
        return {
            'mean': mean,
            'std': std,
            'ci95': ci95,
            'low_ci': mean - ci95,
            'high_ci': mean + ci95,
        }
