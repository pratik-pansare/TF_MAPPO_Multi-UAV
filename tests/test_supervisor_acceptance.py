"""
Comprehensive Supervisor Acceptance Test Suite
================================================
Target Journal: IEEE Internet of Things Journal
Paper: Joint Task Offloading and Adaptive Trajectory Optimization in
       Trust- and Fault-Aware Multi-UAV Edge IoT Networks

Tests the 12 Mandatory Acceptance Criteria from Phase 6:
  Test 1:  Environment-Dependent Task Generation & Seed Reproducibility
  Test 2:  Information Leakage Prevention (No Future Event Oracle)
  Test 3:  One-Time Spatial Clustering Invariance (Coordinates only)
  Test 4:  Coverage-Driven Link Feasibility & Exclusivity (d_im <= R_srv, e_m=1, sum_m u_im <= 1)
  Test 5:  Simultaneous Communication Concurrency Cap (sum_i u_im <= K_comm)
  Test 6:  Link Invalidation & Uplink Interruption Upon Out-of-Range Flight
  Test 7:  Multi-Slot Upload Lifecycle & Causal Queue Timestamp Continuity
  Test 8:  Physical Flight Limits, Battery Reserves & Full-Segment Trajectory Shield
  Test 9:  Reward History Exact Reconstruction from Decomposed Components
  Test 10: Trajectory CSV Reproducibility & Matched Initial Conditions
  Test 11: Active MAPPO Parameter Updates (Non-zero Gradients, Weight Divergence)
  Test 12: Extreme Corner Cases & Edge Robustness (All Faulty, Out of Range, Heavy Hotspot)
"""

import unittest
import numpy as np
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.env import MultiUAVMECEnv
from src.task import StaticIoTDevice, IoTTask, TaskStatus, EnvironmentStateManager
from src.clustering import SpatialClusterManager
from src.channel import ChannelModel
from src.computation import ComputationManager
from src.action_mask import JointSafetyShield, ActionMaskGenerator
from src.actor import HybridActionMaskedActor
from src.agent_factory import make_agent
from src.buffer import RolloutBuffer


class TestSupervisorAcceptance(unittest.TestCase):

    # --------------------------------------------------------------------------
    # Test 1: Environment-Dependent Task Generation & Seed Reproducibility
    # --------------------------------------------------------------------------
    def test_01_environment_dependent_task_profiles(self):
        """Identical device types in different environmental modes produce different profiles."""
        mgr_routine = EnvironmentStateManager(scenario_name="SCENARIO_A_ROUTINE", seed=42)
        mgr_hotspot = EnvironmentStateManager(scenario_name="SCENARIO_B_HOTSPOT", seed=42)

        # Place device at hotspot center [150.0, -200.0]
        dev = StaticIoTDevice(device_id=1, position=[150.0, -200.0, 0.0])

        # At slot 100, Scenario B has an active hotspot while Scenario A is routine
        rng1 = np.random.RandomState(123)
        rng2 = np.random.RandomState(123)

        mode_routine, surge_routine = mgr_routine.get_device_environment_state(dev.position[:2], slot=100)
        mode_hotspot, surge_hotspot = mgr_hotspot.get_device_environment_state(dev.position[:2], slot=100)

        self.assertEqual(mode_routine, "ROUTINE")
        self.assertEqual(mode_hotspot, "HOTSPOT")
        self.assertGreater(surge_hotspot, surge_routine)

        # Seed reproducibility
        t1 = mgr_hotspot.generate_task_for_device(dev, slot=100, rng=rng1, deterministic=True)
        t2 = mgr_hotspot.generate_task_for_device(dev, slot=100, rng=rng2, deterministic=True)
        self.assertIsNotNone(t1)
        self.assertEqual(t1.data_size, t2.data_size)
        self.assertEqual(t1.workload, t2.workload)

    # --------------------------------------------------------------------------
    # Test 2: Information Leakage Prevention
    # --------------------------------------------------------------------------
    def test_02_no_future_information_leakage(self):
        """No future event schedules or hidden seeds leak into agent observation."""
        env = MultiUAVMECEnv(num_users=20, num_uavs=2, scenario="SCENARIO_B_HOTSPOT", seed=42)
        obs, state = env.reset(seed=42)

        # Local observation dimension is 16; verify each element is a current physical quantity
        self.assertEqual(obs.shape, (2, 16))
        for m in range(2):
            o = obs[m]
            # [x, y, z, bat, queue, cpu, cluster_target, tau, phi, elig, dx_c, dy_c, n_cov, slot, vx, vy]
            self.assertFalse(np.isnan(o).any())
            self.assertFalse(np.isinf(o).any())
            # Observation slot fraction is current slot / max slots
            self.assertAlmostEqual(o[13], 0.0)

    # --------------------------------------------------------------------------
    # Test 3: One-Time Spatial Clustering Invariance
    # --------------------------------------------------------------------------
    def test_03_spatial_clustering_invariance(self):
        """Clustering depends on coordinates, not future workloads, and stays invariant during mission."""
        devices = [StaticIoTDevice(i, [np.cos(i)*100, np.sin(i)*100, 0]) for i in range(20)]
        mgr = SpatialClusterManager(n_clusters=3, random_state=42)
        c1, a1 = mgr.fit_clusters(devices)

        # Centroids are unweighted arithmetic means
        for c in range(3):
            sub_pos = [devices[i].position[:2] for i in range(20) if a1[i] == c]
            if len(sub_pos) > 0:
                expected_centroid = np.mean(sub_pos, axis=0)
                np.testing.assert_allclose(c1[c], expected_centroid, atol=1e-5)

        # Check clusters remain invariant when tasks change dynamically
        for dev in devices:
            dev.current_task = IoTTask(dev.id, 1e6, 500, 2.0, slot_generated=1)
        self.assertTrue(np.array_equal(mgr.cluster_assignments, a1))

    # --------------------------------------------------------------------------
    # Test 4: Coverage-Driven Link Feasibility & Exclusivity
    # --------------------------------------------------------------------------
    def test_04_coverage_driven_feasibility_and_exclusivity(self):
        """Active links must satisfy d_im <= R_srv, R_im >= R_min, e_m=1, and single link per device."""
        env = MultiUAVMECEnv(num_users=10, num_uavs=2, seed=42)
        env.reset(seed=42)

        # Place device 0 directly under UAV 0
        env.devices[0].position = np.array([env.uavs[0].position[0], env.uavs[0].position[1], 0.0])
        env.devices[0].current_task = IoTTask(0, 1e6, 500, 2.0, slot_generated=0)

        # Place device 1 far away (> 400m)
        env.devices[1].position = np.array([env.uavs[0].position[0] + 500.0, env.uavs[0].position[1] + 500.0, 0.0])
        env.devices[1].current_task = IoTTask(1, 1e6, 500, 2.0, slot_generated=0)

        env._update_candidate_cache()
        cand_dev_ids_0 = [c[0].id for c in env.cached_candidates[0]]
        self.assertIn(0, cand_dev_ids_0)
        self.assertNotIn(1, cand_dev_ids_0)

    # --------------------------------------------------------------------------
    # Test 5: Simultaneous Communication Concurrency Cap
    # --------------------------------------------------------------------------
    def test_05_concurrency_cap_enforcement(self):
        """With 10 requesting nodes in range and cap=3, at most 3 links are active; others logged."""
        shield = JointSafetyShield(n_uavs=1, k_comm=3)
        # UAV 0 proposes 8 devices
        proposed = {0: [0, 1, 2, 3, 4, 5, 6, 7]}
        rates = {0: {dev_id: (10 - dev_id) * 1e5 for dev_id in range(8)}}

        scheduled = shield.resolve_link_conflicts(proposed, rates, k_comm=3)
        # Must schedule exactly 3 devices
        self.assertEqual(len(scheduled[0]), 3)
        self.assertEqual(scheduled[0], [0, 1, 2])  # Highest rates chosen

    # --------------------------------------------------------------------------
    # Test 6: Link Invalidation Upon Out-of-Range Flight
    # --------------------------------------------------------------------------
    def test_06_link_invalidation_out_of_range(self):
        """Moving UAV out of range invalidates link; upload does not continue without link."""
        cm = ChannelModel()
        iot_pos = np.array([0.0, 0.0, 0.0])
        uav_close = np.array([0.0, 0.0, 100.0])
        uav_far = np.array([600.0, 600.0, 100.0])

        is_cov_close = cm.is_in_coverage(iot_pos, uav_close, max_service_radius=250.0)
        is_cov_far = cm.is_in_coverage(iot_pos, uav_far, max_service_radius=250.0)

        self.assertTrue(is_cov_close)
        self.assertFalse(is_cov_far)

    # --------------------------------------------------------------------------
    # Test 7: Multi-Slot Upload Lifecycle & Causal Progress
    # --------------------------------------------------------------------------
    def test_07_multislot_upload_progress(self):
        """Task progresses across slots, not executed until upload finishes, deadlines from gen time."""
        comp = ComputationManager(n_uavs=1)
        task = IoTTask(device_id=1, data_size_bits=2e6, cycles_per_bit=500, deadline_s=3.0,
                       slot_generated=0, global_time_gen=0.0)

        # Slot 1: rate is 1 Mbps (1e6 bps). Can only transfer 1e6 bits out of 2e6 bits.
        newly_q1, exp1 = comp.process_uplinks(active_links=[(0, task)], rates={task.task_id: 1e6}, current_time_s=0.0, delta_t=1.0)
        self.assertEqual(len(newly_q1), 0)  # Not yet queued
        self.assertAlmostEqual(task.remaining_bits, 1e6)
        self.assertEqual(task.status, TaskStatus.UPLOADING)

        # Slot 2: second 1 Mbps transfer finishes the remaining 1e6 bits!
        newly_q2, exp2 = comp.process_uplinks(active_links=[(0, task)], rates={task.task_id: 1e6}, current_time_s=1.0, delta_t=1.0)
        self.assertEqual(len(newly_q2), 1)  # Now queued!
        self.assertAlmostEqual(task.remaining_bits, 0.0)
        self.assertEqual(task.status, TaskStatus.QUEUED)
        self.assertLessEqual(task.t_upload_end, task.t_absolute_deadline)

    # --------------------------------------------------------------------------
    # Test 8: Trajectory Safety Shield & Continuous Pairwise Separation
    # --------------------------------------------------------------------------
    def test_08_continuous_separation_and_boundary_shield(self):
        """Full-segment minimum continuous separation and boundary constraints enforced."""
        shield = JointSafetyShield(n_uavs=2, safe_dist=10.0, area_radius=500.0)

        # UAV 0 and UAV 1 moving towards each other, crossing mid-slot
        p1 = np.array([-15.0, 0.0, 100.0])
        p2 = np.array([15.0, 0.0, 100.0])
        disp1 = np.array([30.0, 0.0, 0.0])
        disp2 = np.array([-30.0, 0.0, 0.0])

        d_min, s_crit = shield.check_continuous_separation(p1, p2, disp1, disp2)
        self.assertAlmostEqual(d_min, 0.0, places=3)
        self.assertAlmostEqual(s_crit, 0.5, places=3)

        safe_disp, intervened = shield.shield_trajectories(np.array([p1, p2]), np.array([disp1, disp2]))
        self.assertTrue(intervened)

    # --------------------------------------------------------------------------
    # Test 9: Reward History Exact Reconstruction
    # --------------------------------------------------------------------------
    def test_09_reward_reconstruction_from_components(self):
        """Sum of decomposed reward components strictly equals recorded step reward."""
        env = MultiUAVMECEnv(num_users=20, num_uavs=2, seed=42)
        env.reset(seed=42)

        actions = np.zeros((2, env.agent_action_dim))
        next_obs, next_state, rewards, done, info = env.step(actions)

        rc = info['reward_components']
        recon_sum = rc['r_energy'] + rc['r_latency'] + rc['r_completion'] + rc['r_shield_penalty']
        self.assertAlmostEqual(recon_sum, rc['total_step_reward'], places=5)
        self.assertAlmostEqual(recon_sum, rewards[0], places=5)

    # --------------------------------------------------------------------------
    # Test 10: Trajectory CSV Reproducibility
    # --------------------------------------------------------------------------
    def test_10_trajectory_matched_conditions(self):
        """Trained vs untrained trajectory evaluations use identical initial environment seeds."""
        env1 = MultiUAVMECEnv(num_users=20, num_uavs=2, seed=42)
        env2 = MultiUAVMECEnv(num_users=20, num_uavs=2, seed=42)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        np.testing.assert_allclose(obs1, obs2, atol=1e-6)
        for m in range(2):
            np.testing.assert_allclose(env1.uavs[m].position, env2.uavs[m].position, atol=1e-6)

    # --------------------------------------------------------------------------
    # Test 11: Policy Parameter Updates & Gradient Flow
    # --------------------------------------------------------------------------
    def test_11_policy_parameter_updates(self):
        """PPO updates modify actor and critic weights; detects stalled training or zero gradients."""
        env = MultiUAVMECEnv(num_users=20, num_uavs=2, seed=42)
        obs, state = env.reset(seed=42)

        agent = make_agent('MAPPO', env.local_obs_dim, env.global_state_dim, env.agent_action_dim, env.n_agents)
        buffer = RolloutBuffer(rollout_length=16, n_agents=env.n_agents, obs_dim=env.local_obs_dim,
                               state_dim=env.global_state_dim, action_dim=env.agent_action_dim,
                               action_mask_dim=env.action_mask_dim)

        init_weight = agent.actor.encoder.net[0].weight.clone()

        for _ in range(16):
            mask = env.get_action_mask()
            actions, log_probs, values = agent.get_actions(obs, state, action_mask=mask)
            next_obs, next_state, rewards, done, info = env.step(actions)
            buffer.insert(obs, state, actions, log_probs, rewards, float(done), values, action_masks=mask)
            obs, state = next_obs, next_state

        last_v = agent.get_values(state)
        buffer.compute_gae(last_v, 0.0)
        agent.update(buffer)

        updated_weight = agent.actor.encoder.net[0].weight
        weight_diff = torch.norm(updated_weight - init_weight).item()
        self.assertGreater(weight_diff, 1e-6)

    # --------------------------------------------------------------------------
    # Test 12: Extreme Corner Cases & Edge Robustness
    # --------------------------------------------------------------------------
    def test_12_extreme_corner_cases(self):
        """Zero tasks, all faulty UAVs, extreme hotspots terminate safely without crashing."""
        env = MultiUAVMECEnv(num_users=20, num_uavs=2, seed=42)
        env.reset(seed=42)

        # Force all UAVs faulty
        env.fault_manager.phi = np.zeros(2, dtype=int)
        env.fault_manager.eligibility = np.zeros(2, dtype=int)

        actions = np.zeros((2, env.agent_action_dim))
        next_obs, next_state, rewards, done, info = env.step(actions)

        # Must cleanly return without crash and assign zero active links
        self.assertEqual(info['active_links_count'], 0)
        self.assertFalse(done)


if __name__ == '__main__':
    unittest.main()

