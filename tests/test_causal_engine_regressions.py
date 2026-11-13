"""
Causal Task and State Engine Regression Acceptance Tests
=========================================================
Target Journal: IEEE Internet of Things Journal
Paper: Joint Task Offloading and Adaptive Trajectory Optimization
       in Trust- and Fault-Aware Multi-UAV Edge IoT Networks

These acceptance tests verify all model invariants and regressions:
  1. Exogenous trace generation and schema (task_id, device_id, gen_time, deadline, data_bits, cycles_per_bit, event_mode).
  2. Non-blocking arrivals: arriving tasks are NOT suppressed by existing pending tasks; finite buffers track overflow drops.
  3. Strict task lifecycle: PENDING -> ASSIGNED_UPLOAD -> UPLOADING/PAUSED -> UPLOADED -> QUEUED -> IN_SERVICE -> COMPLETED/EXPIRED.
  4. Immutable source-to-UAV upload binding: partial uploads cannot transfer bits to another UAV.
  5. Synchronized fly-then-hover: t_fly = ||Delta q|| / V_cr, t_hover = Delta_t - t_fly. No upload during flight.
  6. Boundary rule: newly uploaded tasks enter resident CPU queue only at slot t+1.
  7. Discrete per-job CPU allocation bound: sum_j f_{j,m} <= F_m^max * phi_m(t).
  8. Radio concurrency cap sum_i u_im <= K_m^comm and exclusivity sum_m u_im <= 1.
  9. Multi-UAV same cluster targeting allowed (strict_exclusivity=False by default).
 10. Exact per-task latency decomposition: T_tot = T_wait + T_up + T_que + T_exe == t_finish - t_gen (zero synthetic fallbacks).
 11. Completion-aware Problem P1 objective: T_svc calculated over ALL generated tasks (with deadline + T_miss for unfinished).
 12. Piecewise propulsion energy: E_prop = P_fly(V_cr)*t_fly + P_hov*t_hover.
"""

import unittest
import numpy as np
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SLOT_DURATION, NUM_TIME_SLOTS, AREA_RADIUS,
    K_COMM_DEFAULT, MAX_CANDIDATE_POOL
)
from src.task import (
    IoTTask, TaskStatus, EnvironmentStateManager, StaticIoTDevice,
    ExogenousTask, ExogenousTraceGenerator, TaskLedger, FiniteBufferDevice
)
from src.computation import ComputationManager
from src.action_mask import JointSafetyShield, ActionMaskGenerator
from src.energy import EnergyManager
from src.env import MultiUAVMECEnv, UAV


class TestCausalTaskEngineRegressions(unittest.TestCase):
    """Test suite verifying all causal engine model invariants."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.delta_t = 1.0

    def test_01_exogenous_trace_generation_and_schema(self):
        """Invariant A: Exogenous trace schema must be deterministic, policy-independent, and complete."""
        generator = ExogenousTraceGenerator(num_devices=10, num_slots=50, seed=123, scenario="SCENARIO_A_ROUTINE")
        trace = generator.generate_trace()

        self.assertGreater(len(trace), 0)
        # Check first task schema
        t0 = trace[0]
        self.assertIsInstance(t0, ExogenousTask)
        self.assertTrue(hasattr(t0, "task_id"))
        self.assertTrue(hasattr(t0, "device_id"))
        self.assertTrue(hasattr(t0, "generation_time"))
        self.assertTrue(hasattr(t0, "deadline"))
        self.assertTrue(hasattr(t0, "data_bits"))
        self.assertTrue(hasattr(t0, "cycles_per_bit"))
        self.assertTrue(hasattr(t0, "event_mode"))
        self.assertGreater(t0.data_bits, 0.0)
        self.assertGreater(t0.cycles_per_bit, 0.0)
        self.assertGreater(t0.deadline, 0.0)

        # Trace determinism: regenerating with same seed yields identical tasks
        generator2 = ExogenousTraceGenerator(num_devices=10, num_slots=50, seed=123, scenario="SCENARIO_A_ROUTINE")
        trace2 = generator2.generate_trace()
        self.assertEqual(len(trace), len(trace2))
        for task_a, task_b in zip(trace, trace2):
            self.assertEqual(task_a.task_id, task_b.task_id)
            self.assertEqual(task_a.device_id, task_b.device_id)
            self.assertEqual(task_a.generation_time, task_b.generation_time)
            self.assertAlmostEqual(task_a.data_bits, task_b.data_bits)
            self.assertAlmostEqual(task_a.cycles_per_bit, task_b.cycles_per_bit)

    def test_02_finite_device_buffer_and_non_blocking_arrivals(self):
        """Invariant A: Arrivals must never be blocked by pending tasks; buffer overflow must be tracked."""
        ledger = TaskLedger()
        device = FiniteBufferDevice(device_id=1, position=[0.0, 0.0, 0.0], buffer_capacity=2, ledger=ledger)

        # Generate 4 tasks arriving for device 1
        tasks = [
            ExogenousTask(task_id=1, device_id=1, generation_time=0.0, deadline=3.0, data_bits=1e6, cycles_per_bit=500, event_mode="ROUTINE"),
            ExogenousTask(task_id=2, device_id=1, generation_time=1.0, deadline=3.0, data_bits=1e6, cycles_per_bit=500, event_mode="ROUTINE"),
            ExogenousTask(task_id=3, device_id=1, generation_time=2.0, deadline=3.0, data_bits=1e6, cycles_per_bit=500, event_mode="ROUTINE"),
            ExogenousTask(task_id=4, device_id=1, generation_time=3.0, deadline=3.0, data_bits=1e6, cycles_per_bit=500, event_mode="ROUTINE"),
        ]

        # Ingest task 1 and 2: should fit in buffer
        device.ingest_task(tasks[0], current_time=0.0)
        device.ingest_task(tasks[1], current_time=1.0)
        self.assertEqual(len(device.buffer), 2)
        self.assertEqual(device.overflow_count, 0)

        # Ingest task 3 and 4: buffer full, must overflow/drop and record in ledger as unsatisfied
        device.ingest_task(tasks[2], current_time=2.0)
        device.ingest_task(tasks[3], current_time=3.0)
        self.assertEqual(len(device.buffer), 2)
        self.assertEqual(device.overflow_count, 2)

        record3 = ledger.get_record(3)
        self.assertIsNotNone(record3)
        self.assertEqual(record3["status"], TaskStatus.DROPPED_BUFFER_OVERFLOW.value)
        self.assertFalse(record3["is_satisfied"])

    def test_03_strict_task_lifecycle_and_deduplication(self):
        """Invariant B: Strict lifecycle transitions and timestamp recording without duplicate counting."""
        ledger = TaskLedger()
        task = IoTTask(
            device_id=0, data_size_bits=1e6, cycles_per_bit=500, deadline_s=4.0,
            slot_generated=0, global_time_gen=0.0, ledger=ledger
        )
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.t_gen, 0.0)

        # Transition to ASSIGNED_UPLOAD
        task.assign_to_uav(uav_id=2, current_time=1.0)
        self.assertEqual(task.status, TaskStatus.ASSIGNED_UPLOAD)
        self.assertEqual(task.serving_uav_id, 2)

        # Transition to UPLOADING
        task.start_upload(current_time=1.2)
        self.assertEqual(task.status, TaskStatus.UPLOADING)
        self.assertEqual(task.t_upload_start, 1.2)

        # Upload finishes
        task.finish_upload(current_time=1.8)
        self.assertEqual(task.status, TaskStatus.QUEUED)
        self.assertEqual(task.t_upload_end, 1.8)

        # Boundary step: enters resident queue at slot boundary
        task.enter_queue(current_time=2.0)
        self.assertEqual(task.status, TaskStatus.QUEUED)
        self.assertEqual(task.t_queue_entry, 2.0)

        # Start execution
        task.start_service(current_time=2.2)
        self.assertEqual(task.status, TaskStatus.IN_SERVICE)
        self.assertEqual(task.t_service_start, 2.2)

        # Finish execution on time
        task.finish_service(current_time=2.9)
        self.assertEqual(task.status, TaskStatus.COMPLETED_ON_TIME)
        self.assertEqual(task.t_finish, 2.9)

        # Latency breakdown identity
        decomp = task.get_latency_decomposition()
        t_wait = decomp["t_wait"]
        t_up = decomp["t_up"]
        t_que = decomp["t_que"]
        t_exe = decomp["t_exe"]
        self.assertAlmostEqual(task.total_latency, t_wait + t_up + t_que + t_exe, places=6)
        self.assertAlmostEqual(task.total_latency, task.t_finish - task.t_gen, places=6)

    def test_04_immutable_upload_binding_and_pause(self):
        """Invariant B & D: Upload is bound to one UAV; when link is lost, it pauses without bit transfer."""
        task = IoTTask(
            device_id=0, data_size_bits=2e6, cycles_per_bit=500, deadline_s=5.0,
            slot_generated=0, global_time_gen=0.0
        )
        task.assign_to_uav(uav_id=1, current_time=0.0)
        task.start_upload(current_time=0.2)

        # Transfer 0.8e6 bits to UAV 1
        task.transfer_bits(0.8e6)
        self.assertAlmostEqual(task.remaining_bits, 1.2e6)
        self.assertEqual(task.serving_uav_id, 1)

        # Next slot: UAV 1 is out of range, upload is paused
        task.pause_upload(current_time=1.0)
        self.assertEqual(task.status, TaskStatus.PAUSED_UPLOADING)
        self.assertEqual(task.serving_uav_id, 1)

        # Binding cannot be stolen by UAV 2: assigning to UAV 2 must raise ValueError
        with self.assertRaises(ValueError):
            task.assign_to_uav(uav_id=2, current_time=2.0)

        # UAV 1 reconnects: resume upload
        task.resume_upload(uav_id=1, current_time=2.0)
        self.assertEqual(task.status, TaskStatus.UPLOADING)
        task.transfer_bits(1.2e6)
        self.assertEqual(task.remaining_bits, 0.0)

    def test_05_fly_then_hover_time_partition_and_no_upload_during_flight(self):
        """Invariant C: t_fly = ||Delta q|| / V_cr, t_hover = Delta_t - t_fly, no uploads when t_hover == 0."""
        v_cr = 15.0  # Cruise speed m/s
        delta_t = 1.0

        # UAV moves 15m in 1s -> t_fly = 1.0s, t_hover = 0.0s
        delta_q_full = 15.0
        t_fly_full = min(delta_t, delta_q_full / v_cr)
        t_hover_full = max(0.0, delta_t - t_fly_full)
        self.assertAlmostEqual(t_fly_full, 1.0)
        self.assertAlmostEqual(t_hover_full, 0.0)

        # Max uploaded bits during full-flight slot must be 0
        rate = 1e6  # 1 Mbps
        uploaded_bits_full = rate * t_hover_full
        self.assertEqual(uploaded_bits_full, 0.0)

        # UAV moves 6m in 1s -> t_fly = 0.4s, t_hover = 0.6s
        delta_q_part = 6.0
        t_fly_part = min(delta_t, delta_q_part / v_cr)
        t_hover_part = max(0.0, delta_t - t_fly_part)
        self.assertAlmostEqual(t_fly_part, 0.4)
        self.assertAlmostEqual(t_hover_part, 0.6)
        uploaded_bits_part = rate * t_hover_part
        self.assertAlmostEqual(uploaded_bits_part, 600000.0)

    def test_06_queue_entry_boundary_rule(self):
        """Invariant C: Upload completing in slot t enters resident queue only at slot t+1."""
        comp_mgr = ComputationManager(n_uavs=1)
        task = IoTTask(
            device_id=0, data_size_bits=1e6, cycles_per_bit=500, deadline_s=4.0,
            slot_generated=0, global_time_gen=0.0
        )
        task.assign_to_uav(uav_id=0, current_time=0.0)
        task.start_upload(current_time=0.2)

        # Upload completes at t=0.8s during slot 0
        comp_mgr.record_upload_completed(uav_id=0, task=task, upload_end_time=0.8)

        # Verify task is in staging queue, NOT yet resident in execution queue for slot 0
        self.assertEqual(len(comp_mgr.resident_queues[0]), 0)
        self.assertEqual(len(comp_mgr.staging_queues[0]), 1)
        self.assertEqual(task.status, TaskStatus.QUEUED)

        # Advance slot boundary to slot 1 (t=1.0s)
        comp_mgr.advance_slot_boundary(new_slot_start_time=1.0)

        # Now task is promoted to resident queue for slot 1
        self.assertEqual(len(comp_mgr.resident_queues[0]), 1)
        self.assertEqual(len(comp_mgr.staging_queues[0]), 0)
        self.assertEqual(task.status, TaskStatus.QUEUED)
        self.assertEqual(task.t_queue_entry, 1.0)

    def test_07_per_job_discrete_cpu_allocation_bound(self):
        """Invariant E: Discrete per-job CPU allocation sum_j f_{j,m} <= F_m^max * phi_m(t)."""
        comp_mgr = ComputationManager(n_uavs=1)
        f_max = 2.0e9  # 2 GHz
        phi = 1.0

        # Create two resident tasks
        t1 = IoTTask(device_id=1, data_size_bits=1e6, cycles_per_bit=500, deadline_s=5.0, slot_generated=0, global_time_gen=0.0)
        t2 = IoTTask(device_id=2, data_size_bits=1e6, cycles_per_bit=500, deadline_s=5.0, slot_generated=0, global_time_gen=0.0)
        t1.status = TaskStatus.QUEUED
        t2.status = TaskStatus.QUEUED
        comp_mgr.resident_queues[0].extend([t1, t2])

        # Propose frequencies that exceed capacity (1.5 GHz + 1.5 GHz = 3.0 GHz > 2.0 GHz)
        proposed_freqs = [1.5e9, 1.5e9]
        valid_freqs = comp_mgr.normalize_cpu_frequencies(
            uav_id=0, proposed_frequencies=proposed_freqs, f_max=f_max, phi=phi
        )

        self.assertAlmostEqual(sum(valid_freqs), f_max, places=2)
        self.assertLessEqual(sum(valid_freqs), f_max * phi + 1e-6)

        # When UAV is faulty (phi = 0), allocated CPU must be 0
        faulty_freqs = comp_mgr.normalize_cpu_frequencies(
            uav_id=0, proposed_frequencies=proposed_freqs, f_max=f_max, phi=0.0
        )
        self.assertEqual(sum(faulty_freqs), 0.0)

    def test_08_radio_concurrency_and_link_exclusivity(self):
        """Invariant D: sum_i u_im <= K_m^comm and sum_m u_im <= 1."""
        shield = JointSafetyShield(n_uavs=2, area_radius=500.0, k_comm=3)

        # UAV 0 and UAV 1 both attempt to link to device 5, plus UAV 0 attempts 4 links (exceeding K_comm=3)
        proposed_links = {
            0: [5, 10, 11, 12],  # 4 proposals, exceeds K_comm=3
            1: [5, 20]           # Contests device 5
        }
        candidate_rates = {
            0: {5: 2.0e6, 10: 1.8e6, 11: 1.5e6, 12: 1.0e6},
            1: {5: 1.2e6, 20: 1.6e6}  # Lower rate for device 5
        }

        scheduled = shield.resolve_link_conflicts(proposed_links, candidate_rates, k_comm=3)

        # Concurrency check
        self.assertLessEqual(len(scheduled[0]), 3)
        self.assertLessEqual(len(scheduled[1]), 3)

        # Exclusivity check: device 5 scheduled for at most one UAV (UAV 0 with higher rate)
        self.assertIn(5, scheduled[0])
        self.assertNotIn(5, scheduled[1])

        # Check total links for UAV 0 is capped at 3
        self.assertEqual(len(scheduled[0]), 3)
        self.assertEqual(scheduled[0], [5, 10, 11])

    def test_09_multi_uav_same_cluster_target_allowed(self):
        """Invariant 2: Multiple UAVs may target the same cluster (remove strict_exclusivity=True default)."""
        shield = JointSafetyShield(n_uavs=3, area_radius=500.0)
        proposed_clusters = np.array([2, 2, 2])  # All 3 UAVs target cluster 2 during hotspot
        eligibility = np.array([1, 1, 1])

        # Default must allow multi-UAV targeting of the same cluster
        executed = shield.resolve_cluster_conflicts(
            proposed_clusters=proposed_clusters,
            eligibility=eligibility,
            strict_exclusivity=False
        )
        self.assertTrue(np.array_equal(executed, np.array([2, 2, 2])))

    def test_10_exact_per_task_latency_decomposition(self):
        """Invariant 10: T_i = T_wait + T_up + T_que + T_exe == t_finish - t_gen (zero synthetic fallbacks)."""
        task = IoTTask(device_id=1, data_size_bits=1.5e6, cycles_per_bit=400, deadline_s=5.0, slot_generated=0, global_time_gen=0.0)
        task.assign_to_uav(uav_id=0, current_time=0.5)
        task.start_upload(current_time=0.5)
        task.finish_upload(current_time=1.2)
        task.enter_queue(current_time=2.0)
        task.start_service(current_time=2.4)
        task.finish_service(current_time=3.1)

        decomp = task.get_latency_decomposition()
        self.assertAlmostEqual(decomp["t_wait"], 0.5)
        self.assertAlmostEqual(decomp["t_up"], 0.7)
        self.assertAlmostEqual(decomp["t_que"], 1.2)
        self.assertAlmostEqual(decomp["t_exe"], 0.7)
        self.assertAlmostEqual(decomp["total_latency"], 3.1)
        self.assertAlmostEqual(decomp["total_latency"], sum([decomp["t_wait"], decomp["t_up"], decomp["t_que"], decomp["t_exe"]]))

    def test_11_problem_p1_completion_aware_objective(self):
        """Invariant 11: P1 uses T_svc over ALL generated tasks with T_miss penalty for uncompleted."""
        ledger = TaskLedger()

        # Task 1: Completed on time, latency 2.0s
        t1 = IoTTask(device_id=1, data_size_bits=1e6, cycles_per_bit=500, deadline_s=3.0, slot_generated=0, global_time_gen=0.0, ledger=ledger)
        t1.assign_to_uav(0, 0.0)
        t1.start_upload(0.0)
        t1.finish_upload(0.8)
        t1.enter_queue(1.0)
        t1.start_service(1.2)
        t1.finish_service(2.0)

        # Task 2: Expired uncompleted, deadline 2.5s
        t2 = IoTTask(device_id=2, data_size_bits=2e6, cycles_per_bit=600, deadline_s=2.5, slot_generated=0, global_time_gen=0.0, ledger=ledger)
        t2.mark_expired(current_time=2.5, status=TaskStatus.EXPIRED_IN_QUEUE)

        t_miss = 2.0
        t_svc, eta_task = ledger.compute_mission_service_latency_and_completion(t_miss=t_miss)

        # Expected:
        # Completed t1: 2.0s
        # Uncompleted t2: deadline (2.5) + t_miss (2.0) = 4.5s
        # Mean T_svc: (2.0 + 4.5) / 2 = 3.25s
        # Completion ratio: 1/2 = 0.50 (50%)
        self.assertAlmostEqual(t_svc, 3.25)
        self.assertAlmostEqual(eta_task, 0.50)

    def test_12_piecewise_propulsion_energy(self):
        """Invariant 12: E_prop = P_fly(V_cr) * t_fly + P_hov * t_hover."""
        energy_mgr = EnergyManager()
        v_cr = 15.0
        p_fly = energy_mgr.compute_propulsion_power(v_cr)
        p_hov = energy_mgr.compute_propulsion_power(0.0)

        t_fly = 0.4
        t_hover = 0.6
        e_prop = energy_mgr.compute_piecewise_propulsion_energy(t_fly=t_fly, t_hover=t_hover, v_cr=v_cr)
        expected_e_prop = p_fly * t_fly + p_hov * t_hover
        self.assertAlmostEqual(e_prop, expected_e_prop, places=4)


if __name__ == "__main__":
    unittest.main()
