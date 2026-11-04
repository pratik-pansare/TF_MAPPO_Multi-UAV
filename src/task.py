"""
IoT Device & Causal Event-Task Model
====================================
Paper Reference: Section III.A & III.C (Eq. 14–15)
Target Journal: IEEE Internet of Things Journal

Models:
  - Static ground IoT node position p_i = (x_i, y_i, 0)
  - Stochastic task arrival a_i(t) in {0, 1} with probability p_gen
  - Heterogeneous task tuple Omega_i(t) = {D_i(t), C_i(t), delta_i(t)}
      D_i(t)     : Input data size (bits)
      C_i(t)     : CPU cycles per bit (cycles/bit)
      delta_i(t) : Latency deadline (seconds)
      W_i(t)     : Total required CPU cycles = D_i * C_i
  - Causal timestamping: t_gen, t_upload_end, t_queue_entry, t_service_start, t_finish
  - Terminal states: COMPLETED_ON_TIME, EXPIRED_IN_QUEUE, EXPIRED_IN_SERVICE, REJECTED, UNFINISHED
"""

import enum
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import (
        IOT_CPU_FREQUENCY, IOT_TX_POWER,
        TASK_DATA_SIZE_RANGE, CYCLES_PER_BIT_RANGE,
        TASK_DEADLINE_RANGE, TASK_GENERATION_PROB
    )
except ImportError:
    IOT_CPU_FREQUENCY = 0.5e9
    IOT_TX_POWER = 0.1
    TASK_DATA_SIZE_RANGE = (0.5e6, 2.5e6)
    CYCLES_PER_BIT_RANGE = (300, 700)
    TASK_DEADLINE_RANGE = (1.0, 3.0)
    TASK_GENERATION_PROB = 0.7


class TaskStatus(enum.Enum):
    PENDING = "PENDING"
    UPLOADING = "UPLOADING"
    QUEUED = "QUEUED"
    IN_SERVICE = "IN_SERVICE"
    COMPLETED_ON_TIME = "COMPLETED_ON_TIME"
    EXPIRED_IN_QUEUE = "EXPIRED_IN_QUEUE"
    EXPIRED_IN_SERVICE = "EXPIRED_IN_SERVICE"
    REJECTED = "REJECTED"
    UNFINISHED = "UNFINISHED"


class IoTTask:
    """Represents a generated computation task Omega_i(t) with full event lifecycle."""

    _id_counter = 0

    def __init__(self, device_id, data_size_bits, cycles_per_bit, deadline_s,
                 slot_generated, cluster_id=-1, global_time_gen=0.0):
        IoTTask._id_counter += 1
        self.task_id = IoTTask._id_counter
        self.device_id = device_id
        self.cluster_id = cluster_id
        self.data_size = float(data_size_bits)
        self.cycles_per_bit = float(cycles_per_bit)
        self.deadline = float(deadline_s)
        self.workload = float(data_size_bits * cycles_per_bit)  # Total CPU cycles W_i
        self.remaining_cycles = float(self.workload)

        # Causal Timestamps
        self.slot_generated = slot_generated
        self.t_gen = float(global_time_gen)
        self.t_absolute_deadline = float(global_time_gen + deadline_s)

        self.t_upload_start = None
        self.t_upload_end = None
        self.t_queue_entry = None
        self.t_service_start = None
        self.t_finish = None

        self.upload_delay = 0.0
        self.queue_delay = 0.0
        self.execution_delay = 0.0
        self.total_latency = 0.0

        self.serving_uav_id = -1
        self.status = TaskStatus.PENDING

    @property
    def is_finished(self):
        return self.status in [TaskStatus.COMPLETED_ON_TIME, TaskStatus.EXPIRED_IN_QUEUE,
                               TaskStatus.EXPIRED_IN_SERVICE, TaskStatus.REJECTED]

    @property
    def is_completed_on_time(self):
        return self.status == TaskStatus.COMPLETED_ON_TIME


class StaticIoTDevice:
    """
    Represents a geographically static ground IoT device at p_i = (x_i, y_i, 0).
    """

    def __init__(self, device_id, position, cpu_freq=IOT_CPU_FREQUENCY, tx_power=IOT_TX_POWER,
                 nominal_rate=TASK_GENERATION_PROB):
        self.id = device_id
        self.position = np.array(position, dtype=float)  # (x_i, y_i, 0)
        self.cpu_freq = float(cpu_freq)
        self.tx_power = float(tx_power)
        self.nominal_rate = float(nominal_rate)

        # Nominal profiling parameters for pre-deployment registration
        self.nominal_data_size = (TASK_DATA_SIZE_RANGE[0] + TASK_DATA_SIZE_RANGE[1]) / 2.0
        self.nominal_cycles_per_bit = (CYCLES_PER_BIT_RANGE[0] + CYCLES_PER_BIT_RANGE[1]) / 2.0
        self.nominal_demand = self.nominal_rate * (self.nominal_data_size * self.nominal_cycles_per_bit)

        self.cluster_id = -1
        self.current_task = None
        self.served_this_slot = False
        self.all_tasks = []

    def reset(self):
        """Reset device state for new episode."""
        self.current_task = None
        self.served_this_slot = False
        self.all_tasks.clear()

    def generate_task(self, slot, rng, current_time=None):
        """
        Generate heterogeneous task Omega_i(t) using Bernoulli arrival process.
        """
        if current_time is None:
            current_time = float(slot)

        if rng.uniform() < self.nominal_rate:
            data_size = rng.uniform(TASK_DATA_SIZE_RANGE[0], TASK_DATA_SIZE_RANGE[1])
            cycles_per_bit = rng.uniform(CYCLES_PER_BIT_RANGE[0], CYCLES_PER_BIT_RANGE[1])
            deadline = rng.uniform(TASK_DEADLINE_RANGE[0], TASK_DEADLINE_RANGE[1])

            task = IoTTask(
                device_id=self.id,
                data_size_bits=data_size,
                cycles_per_bit=cycles_per_bit,
                deadline_s=deadline,
                slot_generated=slot,
                cluster_id=self.cluster_id,
                global_time_gen=current_time
            )
            self.current_task = task
            self.all_tasks.append(task)
        else:
            self.current_task = None

        return self.current_task
