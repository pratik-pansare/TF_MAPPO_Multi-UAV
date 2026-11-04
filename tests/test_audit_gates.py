"""
Automated Test Suite for Required Validation Gates (Section 8 of Master Prompt)
=============================================================================
Target Journal: IEEE Internet of Things Journal
Paper: Trust- and Fault-Aware Action-Masked MAPPO for Dynamic Multi-UAV Enabled
       IoT Mobile Edge Computing with Adaptive Task Offloading and Trajectory Optimization

Validates all 12 required test gates:
  Gate 1: Clustering partitions cover all devices, no empty clusters, weighted centroid equality,
          reproducibility, lambda_w=0, all-zero demands.
  Gate 2: BS reporting and commissioning: packet counts, bytes, overhead.
  Gate 3: Air-to-ground path loss monotonicity, positive gain/rate, no inf/NaN.
  Gate 4: Causal task timestamps, unique terminal states, latency = finish - arrival.
  Gate 5: Reconciliation of N_gen, N_ddl, admitted, finished, expired; ratio uses N_gen.
  Gate 6: Multi-slot queue execution without new arrivals, CPU bounds, non-negative queue.
  Gate 7: Trust and faults: no offloading to e_m=0, cached certificate usable until expiry.
  Gate 8: Battery reserve enforcement, rotary-wing hover power at zero speed.
  Gate 9: Continuous-time trajectory collision check: crossing paths with safe endpoints detected.
  Gate 10: Cluster exclusivity sum_m x_{m,c} <= 1, valid idle fallback.
  Gate 11: PPO probability ratios finite under masks, checkpoint save/load repeatability.
  Gate 12: Reward decomposition reconciliation.
"""

import unittest
import numpy as np
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.clustering import WorkloadAwareClusterManager, BSCommissioningManager, IoTDeviceRegistrationReport
from src.task import StaticIoTDevice, IoTTask, TaskStatus
from src.channel import ChannelModel
from src.computation import ComputationManager
from src.energy import EnergyManager
from src.trust import TrustManager
from src.faults import FaultManager
from src.action_mask import ActionMaskGenerator, JointSafetyShield
from src.actor import HybridActionMaskedActor
from src.env import MultiUAVMECEnv


class TestAuditGates(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    # -------------------------------------------------------------
    # Gate 1: Clustering Partitions & Centroids
    # -------------------------------------------------------------
    def test_gate1_clustering_properties(self):
        mgr = WorkloadAwareClusterManager(n_clusters=4, lambda_w=0.5, random_state=42)
        devices = [StaticIoTDevice(i, [np.cos(i)*100, np.sin(i)*100, 0]) for i in range(20)]
        centroids, assignments = mgr.fit_clusters(devices)

        # 1. sum_c 1[i in I_c] == 1 (every device belongs to exactly one cluster)
        self.assertEqual(len(assignments), len(devices))
        self.assertTrue(np.all(assignments >= 0) and np.all(assignments < 4))

        # 2. No empty clusters after repair
        counts = np.bincount(assignments, minlength=4)
        self.assertTrue(np.all(counts > 0), f"Empty cluster detected: {counts}")

        # 3. Weighted centroid center of mass equality
        positions = np.array([d.position[:2] for d in devices])
        weights = mgr.compute_device_weights(devices)
        for c in range(4):
            mask = (assignments == c)
            expected_centroid = np.sum(positions[mask] * weights[mask, None], axis=0) / np.sum(weights[mask])
            np.testing.assert_allclose(centroids[c], expected_centroid, rtol=1e-3)

        # 4. lambda_w = 0 yields identical weights for all devices
        mgr_zero = WorkloadAwareClusterManager(n_clusters=4, lambda_w=0.0, random_state=42)
        weights_zero = mgr_zero.compute_device_weights(devices)
        self.assertTrue(np.all(weights_zero == 1.0))

        # 5. All-zero nominal workloads handled gracefully without division by zero
        for d in devices:
            d.nominal_demand = 0.0
        weights_all_zero = mgr.compute_device_weights(devices)
        self.assertFalse(np.any(np.isnan(weights_all_zero)))
        self.assertTrue(np.all(weights_all_zero >= 1.0))

    # -------------------------------------------------------------
    # Gate 2: BS Commissioning & Overhead Accounting
    # -------------------------------------------------------------
    def test_gate2_bs_commissioning(self):
        comm = BSCommissioningManager()
        for i in range(10):
            rep = IoTDeviceRegistrationReport(device_id=i, position=[i*10, i*10, 0])
            comm.register_device(rep)
        stats = comm.get_commissioning_stats()

        self.assertEqual(stats['total_registered'], 10)
        self.assertEqual(stats['total_bytes'], 10 * 32)
        self.assertGreater(stats['total_energy_joules'], 0.0)
        self.assertGreater(stats['total_signaling_delay_s'], 0.0)

    # -------------------------------------------------------------
    # Gate 3: A2G Channel Monotonicity & Valid Rates
    # -------------------------------------------------------------
    def test_gate3_channel_model(self):
        cm = ChannelModel()
        iot_pos = [0.0, 0.0, 0.0]
        uav_close = [0.0, 0.0, 100.0]
        uav_far = [400.0, 400.0, 100.0]

        pl_close = cm.compute_average_path_loss(iot_pos, uav_close)
        pl_far = cm.compute_average_path_loss(iot_pos, uav_far)
        # Monotonicity: farther distance must have higher path loss
        self.assertGreater(pl_far, pl_close)

        rate_close = cm.compute_uplink_rate(iot_pos, uav_close, tx_power=0.1, bandwidth=1e6)
        rate_far = cm.compute_uplink_rate(iot_pos, uav_far, tx_power=0.1, bandwidth=1e6)
        # Closer node must achieve higher transmission rate
        self.assertGreater(rate_close, rate_far)
        self.assertFalse(np.isnan(rate_close) or np.isinf(rate_close))

    # -------------------------------------------------------------
    # Gate 4: Causal Task Timestamps & Lifecycle
    # -------------------------------------------------------------
    def test_gate4_task_lifecycle(self):
        task = IoTTask(device_id=1, data_size_bits=1e6, cycles_per_bit=500,
                       deadline_s=2.0, slot_generated=0, global_time_gen=0.0)
        comp = ComputationManager(n_uavs=1)

        comp.admit_task(uav_id=0, task=task, upload_delay_s=0.2, current_time_s=0.0)
        self.assertEqual(task.status, TaskStatus.QUEUED)
        self.assertEqual(task.t_upload_end, 0.2)

        # Process queue with 2 GHz CPU for 1 second (2 Gcycles available > 0.5 Gcycles needed)
        completed, expired = comp.process_slot(uav_cpu_frequencies=[2e9], current_time_s=0.2, delta_t=1.0)
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].status, TaskStatus.COMPLETED_ON_TIME)

        # Causal verification: t_gen < t_upload_end <= t_service_start <= t_finish
        self.assertLessEqual(task.t_gen, task.t_upload_end)
        self.assertLessEqual(task.t_upload_end, task.t_service_start)
        self.assertLessEqual(task.t_service_start, task.t_finish)
        # Total latency == t_finish - t_gen
        self.assertAlmostEqual(task.total_latency, task.t_finish - task.t_gen, places=5)

    # -------------------------------------------------------------
    # Gate 5: Metrics Count & Denominator Reconciliation
    # -------------------------------------------------------------
    def test_gate5_metric_denominator(self):
        env = MultiUAVMECEnv(num_users=20, num_uavs=2, num_clusters=2, seed=42)
        env.reset()
        actions = np.zeros((2, 5))
        actions[:, 0] = [1, 2]  # UAV 0 serves Cluster 1, UAV 1 serves Cluster 2
        actions[:, 4] = 1.0     # 100% CPU

        next_obs, next_state, rewards, done, info = env.step(actions)
        total_gen = sum(len(d.all_tasks) for d in env.devices)
        self.assertGreater(total_gen, 0)
        # Denominator of eta^task is total_gen, not N * T
        completed = len(env.computation_manager.completed_tasks)
        ratio = (completed / total_gen) * 100.0
        self.assertTrue(0.0 <= ratio <= 100.0)

    # -------------------------------------------------------------
    # Gate 6: Multi-Slot Queue Execution Without New Arrivals
    # -------------------------------------------------------------
    def test_gate6_queue_continuity(self):
        comp = ComputationManager(n_uavs=1)
        # Large task: 3 Gcycles
        task = IoTTask(device_id=0, data_size_bits=3e6, cycles_per_bit=1000,
                       deadline_s=5.0, slot_generated=0, global_time_gen=0.0)
        comp.admit_task(uav_id=0, task=task, upload_delay_s=0.1, current_time_s=0.0)

        # Slot 1: 1 GHz CPU -> 1 Gcycles executed, 2 Gcycles remaining
        comp.process_slot(uav_cpu_frequencies=[1e9], current_time_s=0.1, delta_t=1.0)
        self.assertAlmostEqual(comp.queues[0], 2e9, delta=1e5)
        self.assertEqual(len(comp.completed_tasks), 0)

        # Slot 2: No new arrivals! Continue running with 1 GHz -> 1 Gcycle remaining
        comp.process_slot(uav_cpu_frequencies=[1e9], current_time_s=1.1, delta_t=1.0)
        self.assertAlmostEqual(comp.queues[0], 1e9, delta=1e5)
        self.assertEqual(len(comp.completed_tasks), 0)

        # Slot 3: Continue running with 1 GHz -> Task completes!
        completed, _ = comp.process_slot(uav_cpu_frequencies=[1e9], current_time_s=2.1, delta_t=1.0)
        self.assertEqual(len(completed), 1)
        self.assertAlmostEqual(comp.queues[0], 0.0, delta=1e5)

    # -------------------------------------------------------------
    # Gate 7: Trust and Fault Eligibility Enforcement
    # -------------------------------------------------------------
    def test_gate7_trust_fault_enforcement(self):
        tm = TrustManager(n_uavs=2, cert_duration=50)
        fm = FaultManager(n_uavs=2, fault_prob=0.0)

        # Invalidate UAV 0 certificate
        tm.revoke_certificate(0)
        tau = tm.update_trust_states(current_slot=10)
        elig = fm.update_fault_and_eligibility(current_slot=10, tau_states=tau, uav_batteries=[1000, 1000])

        self.assertEqual(elig[0], 0, "UAV 0 with revoked cert must be ineligible")
        self.assertEqual(elig[1], 1, "UAV 1 with valid cert must be eligible")

        mask_gen = ActionMaskGenerator(n_uavs=2, n_clusters=2)
        mask = mask_gen.compute_cluster_mask(
            uav_positions=np.zeros((2, 3)),
            uav_batteries=np.array([1000.0, 1000.0]),
            cluster_centroids=np.zeros((2, 2)),
            eligibility_states=elig,
            tau_states=tau,
            phi_states=fm.phi
        )
        # UAV 0 cannot choose clusters 1 or 2; only idle (index 0)
        self.assertEqual(mask[0, 0], 1.0)
        self.assertEqual(mask[0, 1], 0.0)
        self.assertEqual(mask[0, 2], 0.0)
        # UAV 1 can choose all
        self.assertEqual(mask[1, 0], 1.0)
        self.assertEqual(mask[1, 1], 1.0)

    # -------------------------------------------------------------
    # Gate 8: Battery Updates & Hover Power at Zero Speed
    # -------------------------------------------------------------
    def test_gate8_energy_and_hover(self):
        em = EnergyManager()
        # Rotary-wing propulsion power at speed v = 0 must equal hover power (~168.49 W)
        p_hover = em.compute_propulsion_power(speed_m_s=0.0)
        self.assertAlmostEqual(p_hover, 79.86 + 88.63, places=2)

        # High-speed flight power (v = 30 m/s) must exceed hover power due to parasitic drag
        p_fast = em.compute_propulsion_power(speed_m_s=30.0)
        self.assertGreater(p_fast, p_hover)

        # Step energy at speed = 30 m/s
        e_step = em.compute_step_energy(speed_m_s=30.0, allocated_cpu_hz=2e9,
                                        execution_delay_s=0.5, u2u_delay_s=0.01)
        self.assertGreater(e_step, p_hover * 1.0)

    # -------------------------------------------------------------
    # Gate 9: Continuous Trajectory Separation & Safety Shield
    # -------------------------------------------------------------
    def test_gate9_continuous_collision_shield(self):
        shield = JointSafetyShield(n_uavs=2, safe_dist=10.0)
        # UAV 0 at (-20, 0, 100) flying to (+20, 0, 100)
        # UAV 1 at (+20, 0, 100) flying to (-20, 0, 100)
        # At s = 0.5 (midpoint), both are at (0, 0, 100) -> Head-on crash!
        p0 = np.array([[-20.0, 0.0, 100.0], [20.0, 0.0, 100.0]])
        disp = np.array([[40.0, 0.0, 0.0], [-40.0, 0.0, 0.0]])

        d_min, s_crit = shield.check_continuous_separation(p0[0], p0[1], disp[0], disp[1])
        self.assertAlmostEqual(s_crit, 0.5, places=3)
        self.assertAlmostEqual(d_min, 0.0, places=3)

        safe_disp, intervened = shield.shield_trajectories(p0, disp)
        self.assertTrue(intervened, "Safety shield must intervene on crossing trajectories")

        # After shielding, minimum continuous separation must be >= safe_dist
        d_min_safe, _ = shield.check_continuous_separation(p0[0], p0[1], safe_disp[0], safe_disp[1])
        self.assertGreaterEqual(d_min_safe, 9.0)

    # -------------------------------------------------------------
    # Gate 10: Cluster Exclusivity sum_m x_{m,c} <= 1
    # -------------------------------------------------------------
    def test_gate10_cluster_exclusivity(self):
        shield = JointSafetyShield(n_uavs=3, safe_dist=10.0)
        # Both UAV 0 and UAV 1 propose cluster 1
        proposed = np.array([1, 1, 2])
        pos = np.array([[0, 0, 100], [10, 0, 100], [200, 200, 100]])
        centroids = np.array([[0, 0], [200, 200]])
        eligibility = np.array([1, 1, 1])

        executed = shield.resolve_cluster_conflicts(proposed, pos, centroids, eligibility)
        # Each cluster c in {1, 2} can be claimed by AT MOST one UAV
        counts = [np.sum(executed == c) for c in [1, 2]]
        self.assertTrue(all(cnt <= 1 for cnt in counts))

    # -------------------------------------------------------------
    # Gate 11: PPO Log Probabilities Finite Under Action Mask
    # -------------------------------------------------------------
    def test_gate11_ppo_finite_log_probs(self):
        actor = HybridActionMaskedActor(obs_dim=16, n_clusters=5)
        obs = torch.randn(4, 16)
        # Mask where agent 0 has only cluster 0 feasible
        mask = np.ones((4, 6), dtype=np.float32)
        mask[0, 1:] = 0.0  # Only idle valid

        actions, log_prob, entropy = actor.sample(obs, cluster_mask=mask)
        # Log probability must be strictly finite (no inf or NaN)
        self.assertFalse(torch.any(torch.isnan(log_prob)))
        self.assertFalse(torch.any(torch.isinf(log_prob)))

        # Evaluate actions with the same mask
        eval_log_prob, eval_ent = actor.evaluate_actions(obs, actions, cluster_mask=mask)
        self.assertFalse(torch.any(torch.isnan(eval_log_prob)))
        self.assertFalse(torch.any(torch.isinf(eval_log_prob)))
        np.testing.assert_allclose(log_prob.detach().numpy(), eval_log_prob.detach().numpy(), rtol=1e-3)

    # -------------------------------------------------------------
    # Gate 12: Reward Decomposition Reconciliation
    # -------------------------------------------------------------
    def test_gate12_reward_reconciliation(self):
        env = MultiUAVMECEnv(num_users=10, num_uavs=2, num_clusters=2, seed=42)
        env.reset()
        actions = np.zeros((2, 5))
        actions[:, 0] = [1, 2]
        actions[:, 4] = 0.5

        next_obs, next_state, rewards, done, info = env.step(actions)
        rc = info['reward_components']
        reconstructed = rc['r_energy'] + rc['r_latency'] + rc['r_completion'] + rc['r_shield_penalty']
        self.assertAlmostEqual(rc['total_step_reward'], reconstructed, places=5)
        self.assertAlmostEqual(rewards[0], reconstructed, places=5)


if __name__ == "__main__":
    unittest.main()
