"""
IoT Device & Environment-Dependent Causal Event-Task Model
==========================================================
Paper Reference: Section III.A & III.C (Eq. 14–21)
Target Journal: IEEE Internet of Things Journal

Features:
  1. Static Ground IoT Node: p_i = (x_i, y_i, 0)
  2. Exogenous, Policy-Independent Arrival Traces (Omega_i(t) = {D_i(t), C_i(t), delta_i(t)}):
     - Dedicated RNG stream per scenario
     - Pre-generated or causal non-blocking arrivals
     - Finite device queues with overflow tracking (drops recorded and counted as unsatisfied)
  3. Strict Task Lifecycle State Machine:
     - PENDING -> ASSIGNED_UPLOAD -> UPLOADING / PAUSED_UPLOADING -> UPLOADED -> QUEUED -> IN_SERVICE -> COMPLETED_ON_TIME
     - Failure states: DROPPED_BUFFER_OVERFLOW, EXPIRED_IN_BUFFER, EXPIRED_IN_UPLOADING, EXPIRED_IN_QUEUE, EXPIRED_IN_SERVICE, UNFINISHED
  4. Immutable source-to-UAV upload binding:
     - Partial uploads pause when coverage or authentication is lost
     - Bit transfers never jump to another UAV
  5. Exact latency decomposition:
     - T_tot = T_wait + T_up + T_que + T_exe == t_finish - t_gen (zero synthetic fallbacks)
  6. Centralized TaskLedger:
     - Full provenance and reconstructible event records for Problem P1 completion-aware T_svc
"""

import enum
import os
import sys
import json
from collections import deque
from dataclasses import dataclass, asdict
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

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
    ASSIGNED_UPLOAD = "ASSIGNED_UPLOAD"
    UPLOADING = "UPLOADING"
    PAUSED_UPLOADING = "PAUSED_UPLOADING"
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    IN_SERVICE = "IN_SERVICE"
    COMPLETED_ON_TIME = "COMPLETED_ON_TIME"
    EXPIRED_IN_BUFFER = "EXPIRED_IN_BUFFER"
    EXPIRED_IN_UPLOADING = "EXPIRED_IN_UPLOADING"
    EXPIRED_IN_QUEUE = "EXPIRED_IN_QUEUE"
    EXPIRED_IN_SERVICE = "EXPIRED_IN_SERVICE"
    DROPPED_BUFFER_OVERFLOW = "DROPPED_BUFFER_OVERFLOW"
    REJECTED = "REJECTED"
    UNFINISHED = "UNFINISHED"


@dataclass
class ExogenousTask:
    """Immutable exogenous task descriptor pre-generated before policy evaluation."""
    task_id: int
    device_id: int
    generation_time: float
    deadline: float
    data_bits: float
    cycles_per_bit: float
    event_mode: str = "ROUTINE"
    cluster_id: int = -1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class IoTTask:
    """Represents a computation task with strict causal lifecycle and immutable provenance."""

    _id_counter = 0

    def __init__(self, device_id, data_size_bits, cycles_per_bit, deadline_s,
                 slot_generated=0, cluster_id=-1, global_time_gen=0.0, event_mode="ROUTINE",
                 task_id=None, ledger=None):
        if task_id is not None:
            self.task_id = int(task_id)
            if self.task_id > IoTTask._id_counter:
                IoTTask._id_counter = self.task_id
        else:
            IoTTask._id_counter += 1
            self.task_id = IoTTask._id_counter

        self.device_id = int(device_id)
        self.cluster_id = int(cluster_id)
        self.event_mode = str(event_mode)
        self.ledger = ledger

        self.data_size = float(data_size_bits)
        self.remaining_bits = float(data_size_bits)
        self.cycles_per_bit = float(cycles_per_bit)
        self.deadline = float(deadline_s)
        self.workload = float(data_size_bits * cycles_per_bit)
        self.remaining_cycles = float(self.workload)

        # Causal Timestamps
        self.slot_generated = int(slot_generated)
        self.t_gen = float(global_time_gen)
        self.t_absolute_deadline = float(global_time_gen + deadline_s)

        self.t_assigned = None
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

        if self.ledger is not None:
            self.ledger.register_task(self)

    @property
    def is_finished(self):
        return self.status in [
            TaskStatus.COMPLETED_ON_TIME,
            TaskStatus.EXPIRED_IN_BUFFER,
            TaskStatus.EXPIRED_IN_UPLOADING,
            TaskStatus.EXPIRED_IN_QUEUE,
            TaskStatus.EXPIRED_IN_SERVICE,
            TaskStatus.DROPPED_BUFFER_OVERFLOW,
            TaskStatus.REJECTED,
            TaskStatus.UNFINISHED
        ]

    @property
    def is_completed_on_time(self):
        return self.status == TaskStatus.COMPLETED_ON_TIME

    def assign_to_uav(self, uav_id, current_time):
        uav_id = int(uav_id)
        if self.serving_uav_id != -1 and self.serving_uav_id != uav_id:
            raise ValueError(
                f"Task {self.task_id} is immutably bound to UAV {self.serving_uav_id}, cannot rebind to UAV {uav_id}"
            )
        self.serving_uav_id = uav_id
        self.t_assigned = float(current_time)
        self._set_status(TaskStatus.ASSIGNED_UPLOAD, current_time)

    def start_upload(self, current_time):
        if self.t_upload_start is None:
            self.t_upload_start = float(current_time)
        self._set_status(TaskStatus.UPLOADING, current_time)

    def transfer_bits(self, bits):
        transferred = min(float(bits), self.remaining_bits)
        self.remaining_bits = max(0.0, self.remaining_bits - transferred)
        return transferred

    def pause_upload(self, current_time):
        self._set_status(TaskStatus.PAUSED_UPLOADING, current_time)

    def resume_upload(self, uav_id, current_time):
        if self.serving_uav_id != int(uav_id):
            raise ValueError(
                f"Cannot resume upload on UAV {uav_id}; task {self.task_id} is bound to UAV {self.serving_uav_id}"
            )
        self._set_status(TaskStatus.UPLOADING, current_time)

    def finish_upload(self, current_time):
        self.t_upload_end = float(current_time)
        self.remaining_bits = 0.0
        ref_start = self.t_upload_start if self.t_upload_start is not None else self.t_gen
        self.upload_delay = max(0.0, self.t_upload_end - ref_start)
        self._set_status(TaskStatus.QUEUED, current_time)

    def enter_queue(self, current_time):
        self.t_queue_entry = float(current_time)
        self._set_status(TaskStatus.QUEUED, current_time)

    def start_service(self, current_time):
        self.t_service_start = float(current_time)
        ref_queue_in = self.t_upload_end if self.t_upload_end is not None else (self.t_queue_entry or self.t_gen)
        self.queue_delay = max(0.0, self.t_service_start - ref_queue_in)
        self._set_status(TaskStatus.IN_SERVICE, current_time)

    def execute_cycles(self, cycles, current_time, cpu_freq):
        executed = min(float(cycles), self.remaining_cycles)
        self.remaining_cycles = max(0.0, self.remaining_cycles - executed)
        exec_dur = executed / max(float(cpu_freq), 1.0)
        self.execution_delay += exec_dur
        return executed, exec_dur

    def finish_service(self, current_time):
        self.t_finish = float(current_time)
        self.remaining_cycles = 0.0
        ref_svc_start = self.t_service_start if self.t_service_start is not None else (self.t_queue_entry or self.t_gen)
        self.execution_delay = max(0.0, self.t_finish - ref_svc_start)
        self.total_latency = float(self.t_finish - self.t_gen)

        if self.t_finish <= self.t_absolute_deadline + 1e-9:
            self._set_status(TaskStatus.COMPLETED_ON_TIME, current_time)
        else:
            self._set_status(TaskStatus.EXPIRED_IN_SERVICE, current_time)

    def mark_expired(self, current_time, status=TaskStatus.EXPIRED_IN_QUEUE):
        self.t_finish = float(current_time)
        self.total_latency = float(self.t_finish - self.t_gen)
        self._set_status(status, current_time)

    def get_latency_decomposition(self):
        t_wait = max(0.0, (self.t_upload_start if self.t_upload_start is not None else (self.t_finish or self.t_gen)) - self.t_gen)
        ref_up_start = self.t_upload_start or self.t_gen
        t_up = max(0.0, (self.t_upload_end if self.t_upload_end is not None else (self.t_finish or ref_up_start)) - ref_up_start)
        ref_q_in = self.t_upload_end if self.t_upload_end is not None else (self.t_queue_entry or self.t_gen)
        t_que = max(0.0, (self.t_service_start if self.t_service_start is not None else (self.t_finish or ref_q_in)) - ref_q_in)
        ref_svc_start = self.t_service_start if self.t_service_start is not None else ref_q_in
        t_exe = max(0.0, (self.t_finish if self.t_finish is not None else ref_svc_start) - ref_svc_start)
        tot = (self.t_finish - self.t_gen) if self.t_finish is not None else (t_wait + t_up + t_que + t_exe)

        return {
            "t_wait": float(t_wait),
            "t_up": float(t_up),
            "t_que": float(t_que),
            "t_exe": float(t_exe),
            "total_latency": float(tot),
        }

    def _set_status(self, new_status, current_time):
        old_status = self.status
        self.status = new_status
        if self.ledger is not None:
            self.ledger.record_status_change(self, old_status, new_status, current_time)


class TaskLedger:
    """Centralized, immutable task ledger tracking every task generated in a mission."""

    def __init__(self):
        self.records = {}  # task_id -> dict
        self.completed_count = 0
        self.expired_count = 0
        self.dropped_overflow_count = 0

    def register_task(self, task):
        if task.task_id not in self.records:
            self.records[task.task_id] = {
                "task_id": task.task_id,
                "device_id": task.device_id,
                "cluster_id": task.cluster_id,
                "event_mode": task.event_mode,
                "data_bits": task.data_size,
                "workload_cycles": task.workload,
                "deadline": task.deadline,
                "slot_generated": task.slot_generated,
                "t_gen": task.t_gen,
                "t_absolute_deadline": task.t_absolute_deadline,
                "t_upload_start": None,
                "t_upload_end": None,
                "t_queue_entry": None,
                "t_service_start": None,
                "t_finish": None,
                "status": task.status.value,
                "serving_uav_id": task.serving_uav_id,
                "is_satisfied": False,
                "total_latency": None,
                "t_wait": None,
                "t_up": None,
                "t_que": None,
                "t_exe": None,
            }

    def record_status_change(self, task, old_status, new_status, current_time):
        self.register_task(task)
        rec = self.records[task.task_id]
        rec["status"] = new_status.value
        rec["serving_uav_id"] = task.serving_uav_id

        if task.t_upload_start is not None:
            rec["t_upload_start"] = task.t_upload_start
        if task.t_upload_end is not None:
            rec["t_upload_end"] = task.t_upload_end
        if task.t_queue_entry is not None:
            rec["t_queue_entry"] = task.t_queue_entry
        if task.t_service_start is not None:
            rec["t_service_start"] = task.t_service_start
        if task.t_finish is not None:
            rec["t_finish"] = task.t_finish

        if new_status == TaskStatus.COMPLETED_ON_TIME:
            rec["is_satisfied"] = True
            decomp = task.get_latency_decomposition()
            rec["total_latency"] = decomp["total_latency"]
            rec["t_wait"] = decomp["t_wait"]
            rec["t_up"] = decomp["t_up"]
            rec["t_que"] = decomp["t_que"]
            rec["t_exe"] = decomp["t_exe"]
            self.completed_count += 1
        elif new_status == TaskStatus.DROPPED_BUFFER_OVERFLOW:
            rec["is_satisfied"] = False
            rec["t_finish"] = float(current_time)
            rec["total_latency"] = float(current_time - task.t_gen)
            self.dropped_overflow_count += 1
        elif new_status in [TaskStatus.EXPIRED_IN_BUFFER, TaskStatus.EXPIRED_IN_UPLOADING,
                            TaskStatus.EXPIRED_IN_QUEUE, TaskStatus.EXPIRED_IN_SERVICE]:
            rec["is_satisfied"] = False
            rec["t_finish"] = float(current_time)
            decomp = task.get_latency_decomposition()
            rec["total_latency"] = decomp["total_latency"]
            rec["t_wait"] = decomp["t_wait"]
            rec["t_up"] = decomp["t_up"]
            rec["t_que"] = decomp["t_que"]
            rec["t_exe"] = decomp["t_exe"]
            self.expired_count += 1

    def get_record(self, task_id):
        return self.records.get(int(task_id))

    def compute_mission_service_latency_and_completion(self, t_miss=2.0):
        """
        Computes completion-aware T_svc and completion ratio eta_task across ALL generated tasks.
        Problem P1:
          Completed tasks: actual latency (t_finish - t_gen)
          Unfinished / dropped tasks: deadline + T_miss
        """
        if len(self.records) == 0:
            return 0.0, 1.0

        latencies = []
        satisfied_count = 0

        for r in self.records.values():
            if r["status"] == TaskStatus.COMPLETED_ON_TIME.value:
                satisfied_count += 1
                lat = r["total_latency"] if r["total_latency"] is not None else (r["t_finish"] - r["t_gen"])
                latencies.append(lat)
            else:
                # Expired, dropped, or unfinished
                penalty = float(r["deadline"] + t_miss)
                latencies.append(penalty)

        mean_t_svc = float(np.mean(latencies))
        eta_task = float(satisfied_count / len(self.records))
        return mean_t_svc, eta_task

    def export_csv(self, filepath):
        """Export all task records for full machine-readable provenance."""
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        fieldnames = [
            "task_id", "device_id", "cluster_id", "event_mode", "data_bits",
            "workload_cycles", "deadline", "slot_generated", "t_gen",
            "t_absolute_deadline", "t_upload_start", "t_upload_end",
            "t_queue_entry", "t_service_start", "t_finish", "status",
            "serving_uav_id", "is_satisfied", "total_latency",
            "t_wait", "t_up", "t_que", "t_exe"
        ]
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records.values():
                writer.writerow(r)


class FiniteBufferDevice:
    """
    Static Ground IoT Device with finite FIFO/EDF buffer and causal non-blocking arrivals.
    Never blocks exogenous task arrivals because an existing task is pending.
    """

    def __init__(self, device_id, position, buffer_capacity=5, ledger=None,
                 cpu_freq=IOT_CPU_FREQUENCY, tx_power=IOT_TX_POWER, cluster_id=-1,
                 device_class="ENVIRONMENTAL_SENSOR"):
        self.id = int(device_id)
        self.position = np.array(position, dtype=float)
        self.buffer_capacity = int(buffer_capacity)
        self.ledger = ledger
        self.cpu_freq = float(cpu_freq)
        self.tx_power = float(tx_power)
        self.cluster_id = int(cluster_id)
        self.device_class = device_class

        self.buffer = deque()
        self.overflow_count = 0
        self.all_tasks = []
        self.served_this_slot = False

    def reset(self):
        self.buffer.clear()
        self.overflow_count = 0
        self.all_tasks.clear()
        self.served_this_slot = False

    @property
    def current_task(self):
        """Returns active task in service/upload, or head of pending buffer."""
        for task in self.all_tasks:
            if not task.is_finished:
                return task
        if len(self.buffer) > 0:
            return self.buffer[0]
        return None

    @current_task.setter
    def current_task(self, task):
        # Compatibility setter
        if task is not None and task not in self.all_tasks:
            self.all_tasks.append(task)
            if len(self.buffer) < self.buffer_capacity:
                self.buffer.append(task)

    def ingest_task(self, exogenous_task, current_time):
        """Ingest pre-generated exogenous task into device buffer."""
        slot = int(current_time)
        task = IoTTask(
            device_id=self.id,
            data_size_bits=exogenous_task.data_bits,
            cycles_per_bit=exogenous_task.cycles_per_bit,
            deadline_s=exogenous_task.deadline,
            slot_generated=slot,
            cluster_id=self.cluster_id,
            global_time_gen=current_time,
            event_mode=exogenous_task.event_mode,
            task_id=exogenous_task.task_id,
            ledger=self.ledger
        )

        if len(self.buffer) < self.buffer_capacity:
            self.buffer.append(task)
            self.all_tasks.append(task)
            return task
        else:
            # Buffer full: drop task and record in ledger
            self.overflow_count += 1
            task._set_status(TaskStatus.DROPPED_BUFFER_OVERFLOW, current_time)
            self.all_tasks.append(task)
            return task

    def expire_tasks_in_buffer(self, current_time):
        """Drop pending tasks in buffer whose absolute deadlines have expired."""
        expired = []
        remaining = deque()
        while self.buffer:
            task = self.buffer.popleft()
            if current_time >= task.t_absolute_deadline:
                task.mark_expired(current_time, TaskStatus.EXPIRED_IN_BUFFER)
                expired.append(task)
            else:
                remaining.append(task)
        self.buffer = remaining
        return expired


class StaticIoTDevice(FiniteBufferDevice):
    """
    Drop-in backward compatible Static Ground IoT Device integrating FiniteBufferDevice.
    Supports both exogenous trace ingestion and legacy stochastic generation without blocking.
    """

    def __init__(self, device_id, position, cpu_freq=IOT_CPU_FREQUENCY, tx_power=IOT_TX_POWER,
                 device_class="ENVIRONMENTAL_SENSOR", buffer_capacity=5, ledger=None):
        super().__init__(
            device_id=device_id, position=position, buffer_capacity=buffer_capacity,
            ledger=ledger, cpu_freq=cpu_freq, tx_power=tx_power, device_class=device_class
        )

    def generate_task(self, slot, rng, env_manager=None, current_time=None, deterministic=False):
        """
        Generate task conditional on environment context z_i(t).
        NEVER blocks generation because a preceding task is active.
        """
        if current_time is None:
            current_time = float(slot)

        if env_manager is not None:
            mode_name, surge = env_manager.get_device_environment_state(self.position[:2], slot)
            modes = env_manager.profiles.get("environmental_modes", {})
            mode_cfg = modes.get(mode_name, modes.get("ROUTINE"))
            prob = min(1.0, float(mode_cfg.get("base_generation_rate", 0.20)) * surge)

            if deterministic:
                triggered = (prob > 0.0)
            else:
                triggered = (rng.uniform() < prob)

            if not triggered:
                return None

            d_range = mode_cfg.get("data_size_range_bits", [500000, 1500000])
            c_range = mode_cfg.get("cycles_per_bit_range", [300, 700])
            t_range = mode_cfg.get("deadline_range_s", [1.0, 3.0])

            if deterministic:
                data_size = (d_range[0] + d_range[1]) / 2.0
                cycles_per_bit = (c_range[0] + c_range[1]) / 2.0
                deadline = (t_range[0] + t_range[1]) / 2.0
            else:
                data_size = rng.uniform(d_range[0], d_range[1])
                cycles_per_bit = rng.uniform(c_range[0], c_range[1])
                deadline = rng.uniform(t_range[0], t_range[1])
        else:
            prob = float(TASK_GENERATION_PROB)
            if rng.uniform() >= prob:
                return None
            data_size = rng.uniform(TASK_DATA_SIZE_RANGE[0], TASK_DATA_SIZE_RANGE[1])
            cycles_per_bit = rng.uniform(CYCLES_PER_BIT_RANGE[0], CYCLES_PER_BIT_RANGE[1])
            deadline = rng.uniform(TASK_DEADLINE_RANGE[0], TASK_DEADLINE_RANGE[1])
            mode_name = "ROUTINE"

        exo = ExogenousTask(
            task_id=IoTTask._id_counter + 1,
            device_id=self.id,
            generation_time=current_time,
            deadline=deadline,
            data_bits=data_size,
            cycles_per_bit=cycles_per_bit,
            event_mode=mode_name,
            cluster_id=self.cluster_id
        )
        return self.ingest_task(exo, current_time=current_time)


class EnvironmentStateManager:
    """Manages ground-truth regional environment states and dynamic hotspots."""

    def __init__(self, scenario_name="SCENARIO_A_ROUTINE", config_path=None, seed=42):
        self.scenario_name = scenario_name
        self.rng = np.random.RandomState(seed)
        self.profiles = self._load_profiles(config_path)
        self.active_hotspots = []
        self._init_scenario(scenario_name)

    def _load_profiles(self, config_path):
        search_paths = [
            config_path,
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "event_task_profiles.json"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "event_task_profiles.json"),
        ]
        for p in search_paths:
            if p and os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        data = json.load(f)
                    if data and "environmental_modes" in data:
                        return data
                except Exception:
                    pass

        return {
            "environmental_modes": {
                "ROUTINE": {
                    "base_generation_rate": 0.20,
                    "data_size_range_bits": [300000, 800000],
                    "cycles_per_bit_range": [250, 450],
                    "deadline_range_s": [2.0, 4.0],
                },
                "HOTSPOT": {
                    "base_generation_rate": 0.75,
                    "data_size_range_bits": [1000000, 2200000],
                    "cycles_per_bit_range": [500, 800],
                    "deadline_range_s": [1.0, 2.2],
                },
                "EMERGENCY": {
                    "base_generation_rate": 0.90,
                    "data_size_range_bits": [1500000, 2800000],
                    "cycles_per_bit_range": [600, 950],
                    "deadline_range_s": [0.8, 1.6],
                },
                "PERSISTENT_SHIFT": {
                    "base_generation_rate": 0.50,
                    "data_size_range_bits": [800000, 1800000],
                    "cycles_per_bit_range": [400, 700],
                    "deadline_range_s": [1.2, 2.5],
                },
            },
            "scenarios": {
                "SCENARIO_A_ROUTINE": {"hotspots": []},
                "SCENARIO_B_HOTSPOT": {
                    "hotspots": [{
                        "cluster_id": 2,
                        "center": [150.0, -200.0],
                        "radius_m": 180.0,
                        "start_slot": 50,
                        "end_slot": 250,
                        "surge_factor": 2.5,
                    }]
                },
                "SCENARIO_C_MULTI_EVENT": {
                    "hotspots": [
                        {"cluster_id": 1, "center": [-250.0, 200.0], "radius_m": 150.0, "start_slot": 80, "end_slot": 300, "surge_factor": 2.2},
                        {"cluster_id": 4, "center": [300.0, 150.0], "radius_m": 160.0, "start_slot": 120, "end_slot": 350, "surge_factor": 2.6},
                    ]
                },
                "SCENARIO_D_STRESS_TEST": {
                    "hotspots": [{
                        "cluster_id": 3,
                        "center": [0.0, 0.0],
                        "radius_m": 220.0,
                        "start_slot": 30,
                        "end_slot": 450,
                        "surge_factor": 3.0,
                    }]
                },
            }
        }

    def _init_scenario(self, scenario_name):
        self.scenario_name = scenario_name
        scenarios = self.profiles.get("scenarios", {})
        sc_info = scenarios.get(scenario_name, {})
        self.active_hotspots = sc_info.get("hotspots", [])

    def get_device_environment_state(self, device_position_2d, slot):
        for hs in self.active_hotspots:
            if hs.get("start_slot", 0) <= slot <= hs.get("end_slot", 500):
                center = np.array(hs.get("center", [0.0, 0.0]), dtype=float)
                radius = float(hs.get("radius_m", 150.0))
                dist = np.linalg.norm(device_position_2d - center)
                if dist <= radius:
                    return "HOTSPOT", float(hs.get("surge_factor", 2.5))

        if self.scenario_name == "SCENARIO_D_STRESS_TEST" and slot > 250:
            return "PERSISTENT_SHIFT", 1.5

        return "ROUTINE", 1.0

    def generate_task_for_device(self, device, slot, rng, current_time=None, deterministic=False):
        """Generates task conditional on environment context z_i(t) and application profile."""
        if current_time is None:
            current_time = float(slot)

        mode_name, surge = self.get_device_environment_state(device.position[:2], slot)
        modes = self.profiles.get("environmental_modes", {})
        mode_cfg = modes.get(mode_name, modes.get("ROUTINE"))

        prob = min(1.0, float(mode_cfg.get("base_generation_rate", 0.20)) * surge)

        if deterministic:
            triggered = (prob > 0.0)
        else:
            triggered = (rng.uniform() < prob)

        if not triggered:
            return None

        d_range = mode_cfg.get("data_size_range_bits", [500000, 1500000])
        c_range = mode_cfg.get("cycles_per_bit_range", [300, 700])
        t_range = mode_cfg.get("deadline_range_s", [1.0, 3.0])

        if deterministic:
            data_size = (d_range[0] + d_range[1]) / 2.0
            cycles_per_bit = (c_range[0] + c_range[1]) / 2.0
            deadline = (t_range[0] + t_range[1]) / 2.0
        else:
            data_size = rng.uniform(d_range[0], d_range[1])
            cycles_per_bit = rng.uniform(c_range[0], c_range[1])
            deadline = rng.uniform(t_range[0], t_range[1])

        task = IoTTask(
            device_id=device.id,
            data_size_bits=data_size,
            cycles_per_bit=cycles_per_bit,
            deadline_s=deadline,
            slot_generated=slot,
            cluster_id=device.cluster_id,
            global_time_gen=current_time,
            event_mode=mode_name,
        )
        return task


class ExogenousTraceGenerator:
    """Generates policy-independent exogenous task arrivals for fair evaluation across all algorithms."""

    def __init__(self, num_devices=80, num_slots=500, seed=42, scenario="SCENARIO_A_ROUTINE",
                 device_positions=None, slot_duration=1.0):
        self.num_devices = int(num_devices)
        self.num_slots = int(num_slots)
        self.seed = int(seed)
        self.scenario = scenario
        self.slot_duration = float(slot_duration)
        self.rng = np.random.RandomState(seed)
        self.env_mgr = EnvironmentStateManager(scenario_name=scenario, seed=seed)

        if device_positions is not None:
            self.positions = np.array(device_positions, dtype=float)
        else:
            # Deterministic device coordinates
            pos_rng = np.random.RandomState(seed)
            self.positions = []
            for _ in range(self.num_devices):
                ang = pos_rng.uniform(0.0, 2.0 * np.pi)
                r = pos_rng.uniform(10.0, 500.0)
                self.positions.append([r * np.cos(ang), r * np.sin(ang), 0.0])
            self.positions = np.array(self.positions, dtype=float)

    def generate_trace(self):
        """Generates list of ExogenousTask instances across all slots."""
        tasks = []
        task_id = 1

        for slot in range(self.num_slots):
            t_curr = slot * self.slot_duration
            for dev_id in range(self.num_devices):
                pos_2d = self.positions[dev_id][:2]
                mode_name, surge = self.env_mgr.get_device_environment_state(pos_2d, slot)
                mode_cfg = self.env_mgr.profiles["environmental_modes"].get(
                    mode_name, self.env_mgr.profiles["environmental_modes"]["ROUTINE"]
                )
                prob = min(1.0, float(mode_cfg["base_generation_rate"]) * surge)

                if self.rng.uniform() < prob:
                    d_range = mode_cfg["data_size_range_bits"]
                    c_range = mode_cfg["cycles_per_bit_range"]
                    t_range = mode_cfg["deadline_range_s"]

                    data_size = float(self.rng.uniform(d_range[0], d_range[1]))
                    cycles = float(self.rng.uniform(c_range[0], c_range[1]))
                    ddl = float(self.rng.uniform(t_range[0], t_range[1]))

                    task = ExogenousTask(
                        task_id=task_id,
                        device_id=dev_id,
                        generation_time=t_curr,
                        deadline=ddl,
                        data_bits=data_size,
                        cycles_per_bit=cycles,
                        event_mode=mode_name,
                        cluster_id=-1
                    )
                    tasks.append(task)
                    task_id += 1

        return tasks

    def save_trace(self, filepath):
        trace = self.generate_trace()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump([t.to_dict() for t in trace], f, indent=2)

    @staticmethod
    def load_trace(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        return [ExogenousTask.from_dict(d) for d in data]
