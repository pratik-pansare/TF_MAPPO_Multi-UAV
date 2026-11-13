"""
Computation & Onboard Discrete-Event Queue Manager
===================================================
Paper Reference: Section III.C (Eq. 16–21)
Target Journal: IEEE Internet of Things Journal

Models:
  - Synchronized Fly-Then-Hover Boundary Rule:
    Newly uploaded tasks in slot t enter the resident CPU queue only at slot t+1.
  - Per-Resident-Job CPU Allocation Constraint:
    sum_j f_{j,m}(t) <= F_m^max * phi_m(t)
  - Causal Timestamping & Exact Latency Decomposition:
    total_latency = (t_finish - t_gen) == T_wait + T_up + T_que + T_exe
"""

from collections import deque
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.task import TaskStatus


class ComputationManager:
    """Manages UAV onboard computation staging and resident queues with per-job CPU allocation."""

    def __init__(self, n_uavs):
        self.n_uavs = int(n_uavs)
        # Resident queues: jobs available to execute in current slot
        self.resident_queues = [deque() for _ in range(n_uavs)]
        # Staging queues: jobs that finished upload during current slot (promoted at slot boundary)
        self.staging_queues = [deque() for _ in range(n_uavs)]

        self.completed_tasks = []
        self.expired_tasks = []
        self.total_cycles_executed = np.zeros(n_uavs, dtype=float)

    @property
    def uav_queues(self):
        """Backward-compatible alias for resident queues."""
        return self.resident_queues

    @uav_queues.setter
    def uav_queues(self, val):
        self.resident_queues = val

    def reset(self):
        """Reset all queues and execution tracking."""
        self.resident_queues = [deque() for _ in range(self.n_uavs)]
        self.staging_queues = [deque() for _ in range(self.n_uavs)]
        self.completed_tasks.clear()
        self.expired_tasks.clear()
        self.total_cycles_executed = np.zeros(self.n_uavs, dtype=float)

    @property
    def queues(self):
        """Returns array of current backlog Q_m(t) in CPU cycles across all UAVs."""
        backlogs = np.zeros(self.n_uavs, dtype=float)
        for m in range(self.n_uavs):
            backlogs[m] = sum(task.remaining_cycles for task in self.resident_queues[m])
            backlogs[m] += sum(task.remaining_cycles for task in self.staging_queues[m])
        return backlogs

    def record_upload_completed(self, uav_id, task, upload_end_time):
        """
        Record task upload completion and place into staging queue.
        Task remains staged until advance_slot_boundary() promotes it to resident queue.
        """
        task.finish_upload(upload_end_time)
        self.staging_queues[uav_id].append(task)

    def advance_slot_boundary(self, new_slot_start_time):
        """
        Boundary transition: promote all staged tasks into resident execution queues.
        Enforces causality: newly uploaded work in slot t executes only from slot t+1.
        """
        for m in range(self.n_uavs):
            while self.staging_queues[m]:
                task = self.staging_queues[m].popleft()
                task.enter_queue(new_slot_start_time)
                self.resident_queues[m].append(task)

    def normalize_cpu_frequencies(self, uav_id, proposed_frequencies, f_max, phi=1.0):
        """
        Projects proposed per-job CPU frequencies to satisfy:
          sum_j f_{j,m} <= F_m^max * phi_m(t)
        """
        phi_val = max(0.0, float(phi))
        allowed_total = float(f_max) * phi_val

        if allowed_total <= 0.0 or len(proposed_frequencies) == 0:
            return [0.0] * len(proposed_frequencies)

        props = [max(0.0, float(f)) for f in proposed_frequencies]
        tot_prop = sum(props)

        if tot_prop <= 0.0:
            # Default equal share among resident jobs
            n = len(props)
            return [allowed_total / n] * n

        if tot_prop > allowed_total:
            scale = allowed_total / tot_prop
            return [p * scale for p in props]

        return props

    def process_resident_jobs(self, uav_id, allocated_frequencies, slot_start_time, delta_t=1.0,
                              phi=1.0, max_cpu_freq=3.0e9):
        """
        Executes resident jobs for UAV m during slot [slot_start_time, slot_start_time + delta_t].
        """
        slot_completed = []
        slot_expired = []
        queue = self.resident_queues[uav_id]

        if phi <= 0.0 or len(queue) == 0:
            return slot_completed, slot_expired

        # Normalize frequencies to capacity
        if len(allocated_frequencies) < len(queue):
            # Pad with zeros or equal share
            rem_cap = max(0.0, float(max_cpu_freq) * float(phi) - sum(allocated_frequencies))
            n_unalloc = len(queue) - len(allocated_frequencies)
            pad_share = rem_cap / max(1, n_unalloc)
            allocated_frequencies = list(allocated_frequencies) + [pad_share] * n_unalloc

        valid_freqs = self.normalize_cpu_frequencies(
            uav_id, allocated_frequencies[:len(queue)], max_cpu_freq, phi
        )

        remaining_queue = deque()
        for idx, task in enumerate(queue):
            f_j = valid_freqs[idx]

            # Check if deadline already passed while waiting in queue
            if slot_start_time >= task.t_absolute_deadline:
                task.mark_expired(slot_start_time, TaskStatus.EXPIRED_IN_QUEUE)
                self.expired_tasks.append(task)
                slot_expired.append(task)
                continue

            if f_j <= 0.0:
                remaining_queue.append(task)
                continue

            if task.t_service_start is None:
                task.start_service(slot_start_time)

            cycles_needed = task.remaining_cycles
            cycles_avail = f_j * float(delta_t)

            if cycles_avail >= cycles_needed:
                # Finished during this slot
                exec_dur = cycles_needed / f_j
                task.execute_cycles(cycles_needed, slot_start_time, f_j)
                t_fin = slot_start_time + exec_dur
                self.total_cycles_executed[uav_id] += cycles_needed

                if t_fin <= task.t_absolute_deadline + 1e-9:
                    task.finish_service(t_fin)
                    self.completed_tasks.append(task)
                    slot_completed.append(task)
                else:
                    task.mark_expired(t_fin, TaskStatus.EXPIRED_IN_SERVICE)
                    self.expired_tasks.append(task)
                    slot_expired.append(task)
            else:
                # Partial execution
                task.execute_cycles(cycles_avail, slot_start_time, f_j)
                self.total_cycles_executed[uav_id] += cycles_avail
                # Check if deadline expired during slot
                if (slot_start_time + delta_t) >= task.t_absolute_deadline:
                    task.mark_expired(slot_start_time + delta_t, TaskStatus.EXPIRED_IN_SERVICE)
                    self.expired_tasks.append(task)
                    slot_expired.append(task)
                else:
                    remaining_queue.append(task)

        self.resident_queues[uav_id] = remaining_queue
        return slot_completed, slot_expired

    def compute_execution_delay(self, task_workload_cycles, allocated_cpu_hz):
        safe_cpu = max(float(allocated_cpu_hz), 1.0)
        return float(task_workload_cycles) / safe_cpu

    def compute_queueing_delay(self, uav_id, aggregate_cpu_hz):
        if aggregate_cpu_hz <= 0:
            return 0.0
        return float(self.queues[uav_id]) / float(aggregate_cpu_hz)

    def compute_total_latency(self, uplink_delay, queueing_delay, execution_delay):
        return float(uplink_delay + queueing_delay + execution_delay)

    def admit_task(self, uav_id, task, upload_delay_s, current_time_s):
        """Backward-compatible immediate admit."""
        task.serving_uav_id = uav_id
        if task.t_upload_start is None:
            task.t_upload_start = current_time_s
        task.finish_upload(current_time_s + upload_delay_s)
        task.enter_queue(current_time_s + upload_delay_s)
        self.resident_queues[uav_id].append(task)

    def process_uplinks(self, active_links, rates, current_time_s, delta_t=1.0):
        """
        Progress multi-slot active uplinks during hover time.
        active_links: list of (uav_id, task)
        rates: dict {task_id: rate_bps}
        """
        newly_queued = []
        upload_expired = []

        for uav_id, task in active_links:
            rate = max(rates.get(task.task_id, 1e5), 1e3)

            if task.t_upload_start is None:
                task.assign_to_uav(uav_id, current_time_s)
                task.start_upload(current_time_s)

            bits_can_transfer = rate * float(delta_t)
            if task.remaining_bits <= bits_can_transfer:
                t_tx = task.remaining_bits / rate
                t_end = current_time_s + t_tx
                task.transfer_bits(task.remaining_bits)

                if t_end <= task.t_absolute_deadline:
                    self.record_upload_completed(uav_id, task, t_end)
                    newly_queued.append(task)
                else:
                    task.mark_expired(t_end, TaskStatus.EXPIRED_IN_UPLOADING)
                    self.expired_tasks.append(task)
                    upload_expired.append(task)
            else:
                task.transfer_bits(bits_can_transfer)
                if (current_time_s + delta_t) > task.t_absolute_deadline:
                    task.mark_expired(current_time_s + delta_t, TaskStatus.EXPIRED_IN_UPLOADING)
                    self.expired_tasks.append(task)
                    upload_expired.append(task)

        return newly_queued, upload_expired

    def process_slot(self, uav_cpu_frequencies, current_time_s, delta_t=1.0):
        """
        Process resident computation queues across the fleet during time slot Delta_t.
        """
        all_completed = []
        all_expired = []

        for m in range(self.n_uavs):
            cpu_freq = max(float(uav_cpu_frequencies[m]), 0.0)
            if cpu_freq <= 0.0 or len(self.resident_queues[m]) == 0:
                continue
            # Single aggregate CPU frequency distributed across resident jobs
            n_jobs = len(self.resident_queues[m])
            job_freqs = [cpu_freq / n_jobs] * n_jobs
            comp, exp = self.process_resident_jobs(
                uav_id=m, allocated_frequencies=job_freqs,
                slot_start_time=current_time_s, delta_t=delta_t,
                phi=1.0, max_cpu_freq=cpu_freq
            )
            all_completed.extend(comp)
            all_expired.extend(exp)

        return all_completed, all_expired
