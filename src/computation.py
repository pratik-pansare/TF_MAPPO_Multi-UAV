"""
Computation & Onboard Discrete-Event Queue Manager
===================================================
Paper Reference: Section III.C (Eq. 16–21)
Target Journal: IEEE Internet of Things Journal

Models:
  - UAV CPU Capacity Constraint: sum_i y_{i,m}(t) f_{i,m}(t) <= F_m^max
  - Discrete-event FIFO/EDF job scheduling queue on each UAV
  - Macroscopic Workload Backlog:
      Q_m(t) = sum_{k in Queue_m} W_k^{rem}(t)  [CPU cycles]
      Q_m(t+1) = [Q_m(t) - F_m(t) * Delta_t]^+ + sum_i y_{i,m}(t) W_i(t)
  - Causal timestamping & physical delay tracking:
      t_finish - t_gen = T_{i,m}^{up} + T_{i,m}^{que} + T_{i,m}^{exe}
  - Continues processing existing queued work across slots even without new arrivals.
"""

from collections import deque
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.task import TaskStatus


class ComputationManager:
    """Manages UAV onboard computation queues, CPU allocation, and task execution."""

    def __init__(self, n_uavs):
        self.n_uavs = n_uavs
        # Onboard job queues for each UAV
        self.uav_queues = [deque() for _ in range(n_uavs)]
        self.completed_tasks = []
        self.expired_tasks = []
        self.total_cycles_executed = np.zeros(n_uavs, dtype=float)

    def reset(self):
        """Reset all queues and execution tracking."""
        self.uav_queues = [deque() for _ in range(self.n_uavs)]
        self.completed_tasks.clear()
        self.expired_tasks.clear()
        self.total_cycles_executed = np.zeros(self.n_uavs, dtype=float)

    @property
    def queues(self):
        """Returns array of current backlog Q_m(t) in CPU cycles for all UAVs."""
        backlogs = np.zeros(self.n_uavs, dtype=float)
        for m in range(self.n_uavs):
            backlogs[m] = sum(task.remaining_cycles for task in self.uav_queues[m])
        return backlogs

    def compute_execution_delay(self, task_workload_cycles, allocated_cpu_hz):
        """Compute task execution delay T_{i,m}^{exe}(t) = W_i / f_{i,m} (Eq. 19)."""
        safe_cpu = max(float(allocated_cpu_hz), 1.0)
        return float(task_workload_cycles) / safe_cpu

    def compute_queueing_delay(self, uav_id, aggregate_cpu_hz):
        """Compute expected queueing delay T_m^{que}(t) = Q_m(t) / F_m(t) (Eq. 20)."""
        if aggregate_cpu_hz <= 0:
            return 0.0
        return float(self.queues[uav_id]) / float(aggregate_cpu_hz)

    def compute_total_latency(self, uplink_delay, queueing_delay, execution_delay):
        """Compute total end-to-end task latency T_{i,m}^{tot}(t) (Eq. 21)."""
        return float(uplink_delay + queueing_delay + execution_delay)

    def admit_task(self, uav_id, task, upload_delay_s, current_time_s):
        """
        Admit a task into UAV onboard queue upon upload completion.
        """
        task.serving_uav_id = uav_id
        task.t_upload_start = current_time_s
        task.t_upload_end = current_time_s + upload_delay_s
        task.upload_delay = upload_delay_s
        task.t_queue_entry = task.t_upload_end
        task.status = TaskStatus.QUEUED
        self.uav_queues[uav_id].append(task)

    def process_slot(self, uav_cpu_frequencies, current_time_s, delta_t=1.0):
        """
        Process computation queues across the fleet during time slot Delta_t.
        Queued tasks continue running even if no new task arrived in this slot.
        """
        slot_completed = []
        slot_expired = []

        for m in range(self.n_uavs):
            cpu_freq = max(float(uav_cpu_frequencies[m]), 0.0)
            available_cycles = cpu_freq * float(delta_t)
            time_offset = 0.0

            queue = self.uav_queues[m]
            while queue and available_cycles > 0.0:
                task = queue[0]

                # Check if deadline already expired while in queue
                now_est = current_time_s + time_offset
                if now_est > task.t_absolute_deadline:
                    task.status = TaskStatus.EXPIRED_IN_QUEUE
                    task.t_finish = now_est
                    task.total_latency = task.t_finish - task.t_gen
                    self.expired_tasks.append(task)
                    slot_expired.append(task)
                    queue.popleft()
                    continue

                if task.t_service_start is None:
                    task.t_service_start = now_est
                    task.queue_delay = max(0.0, task.t_service_start - task.t_queue_entry)
                task.status = TaskStatus.IN_SERVICE

                cycles_needed = task.remaining_cycles
                if available_cycles >= cycles_needed:
                    # Task finishes in this slot
                    exec_time = cycles_needed / max(cpu_freq, 1.0)
                    available_cycles -= cycles_needed
                    self.total_cycles_executed[m] += cycles_needed
                    time_offset += exec_time

                    task.remaining_cycles = 0.0
                    task.t_finish = current_time_s + time_offset
                    task.execution_delay = task.workload / max(cpu_freq, 1.0)
                    task.total_latency = task.t_finish - task.t_gen

                    if task.t_finish <= task.t_absolute_deadline:
                        task.status = TaskStatus.COMPLETED_ON_TIME
                        self.completed_tasks.append(task)
                        slot_completed.append(task)
                    else:
                        task.status = TaskStatus.EXPIRED_IN_SERVICE
                        self.expired_tasks.append(task)
                        slot_expired.append(task)

                    queue.popleft()
                else:
                    # Partial execution, remains in queue for next slot
                    task.remaining_cycles -= available_cycles
                    self.total_cycles_executed[m] += available_cycles
                    available_cycles = 0.0

        return slot_completed, slot_expired

    def update_queue_backlog(self, uav_id, aggregate_cpu_hz, newly_offloaded_workload, delta_t=1.0):
        """
        Macroscopic queue update for analytical checks (Eq. 18):
        Q_m(t+1) = [Q_m(t) - F_m(t) * Delta_t]^+ + sum y_{i,m}(t) * W_i(t)
        """
        processed = float(aggregate_cpu_hz) * float(delta_t)
        current = self.queues[uav_id]
        updated = max(0.0, current - processed) + float(newly_offloaded_workload)
        return updated
