"""
IoT Device & Environment-Dependent Causal Event-Task Model
==========================================================
Paper Reference: Section III.A & III.C (Eq. 14–15)
Target Journal: IEEE Internet of Things Journal

Features:
  1. Static Ground IoT Node: p_i = (x_i, y_i, 0)
  2. Environment State z_i(t) & Dynamic Scenario Modes:
     - ROUTINE, HOTSPOT, EMERGENCY, PERSISTENT_SHIFT
     - Causal event trigger: a_i(t) = 1[g_i(z_i(t), t) triggers job]
  3. Context-Conditioned Task Profile Tuple Omega_i(t) = {D_i(t), C_i(t), delta_i(t)}:
     - D_i(t): Input data size (bits)
     - C_i(t): CPU cycles per bit (cycles/bit)
     - delta_i(t): Strict latency deadline (seconds)
     - W_i(t) = D_i * C_i (Total CPU cycles required)
  4. Multi-Slot Uplink Transmission & Queue State Lifecycle:
     - PENDING -> UPLOADING -> QUEUED -> IN_SERVICE -> COMPLETED_ON_TIME
     - Failure states: EXPIRED_IN_UPLOADING, EXPIRED_IN_QUEUE, EXPIRED_IN_SERVICE, REJECTED
  5. Zero Oracle Workload Assumption:
     - Agents observe only active, reachable task descriptors; future event schedules remain hidden.
"""

import enum
import os
import sys
import json
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
    UPLOADING = "UPLOADING"
    QUEUED = "QUEUED"
    IN_SERVICE = "IN_SERVICE"
    COMPLETED_ON_TIME = "COMPLETED_ON_TIME"
    EXPIRED_IN_UPLOADING = "EXPIRED_IN_UPLOADING"
    EXPIRED_IN_QUEUE = "EXPIRED_IN_QUEUE"
    EXPIRED_IN_SERVICE = "EXPIRED_IN_SERVICE"
    REJECTED = "REJECTED"
    UNFINISHED = "UNFINISHED"


class IoTTask:
    """Represents an environment-generated computation task Omega_i(t) with full causal lifecycle."""

    _id_counter = 0

    def __init__(self, device_id, data_size_bits, cycles_per_bit, deadline_s,
                 slot_generated, cluster_id=-1, global_time_gen=0.0, event_mode="ROUTINE"):
        IoTTask._id_counter += 1
        self.task_id = IoTTask._id_counter
        self.device_id = device_id
        self.cluster_id = cluster_id
        self.event_mode = event_mode

        self.data_size = float(data_size_bits)
        self.remaining_bits = float(data_size_bits)
        self.cycles_per_bit = float(cycles_per_bit)
        self.deadline = float(deadline_s)
        self.workload = float(data_size_bits * cycles_per_bit)  # Total CPU cycles W_i
        self.remaining_cycles = float(self.workload)

        # Causal Timestamps
        self.slot_generated = int(slot_generated)
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
        return self.status in [
            TaskStatus.COMPLETED_ON_TIME,
            TaskStatus.EXPIRED_IN_UPLOADING,
            TaskStatus.EXPIRED_IN_QUEUE,
            TaskStatus.EXPIRED_IN_SERVICE,
            TaskStatus.REJECTED
        ]

    @property
    def is_completed_on_time(self):
        return self.status == TaskStatus.COMPLETED_ON_TIME


class EnvironmentStateManager:
    """
    Manages regional ground-truth environmental state, time-varying hotspots,
    and profile-driven task generation without oracle leakage to agents.
    """

    def __init__(self, scenario_name="SCENARIO_A_ROUTINE", config_path=None, seed=42):
        self.scenario_name = scenario_name
        self.rng = np.random.RandomState(seed)
        self.profiles = self._load_profiles(config_path)
        self.active_hotspots = []
        self._init_scenario(scenario_name)

    def _load_profiles(self, config_path):
        """Loads human-readable event task profiles from JSON/YAML or returns defaults."""
        search_paths = [
            config_path,
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "event_task_profiles.json"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "event_task_profiles.json"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "event_task_profiles.yaml"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "event_task_profiles.yaml"),
        ]
        for p in search_paths:
            if p and os.path.exists(p):
                try:
                    if p.endswith(".json"):
                        with open(p, "r") as f:
                            data = json.load(f)
                    elif yaml is not None:
                        with open(p, "r") as f:
                            data = yaml.safe_load(f)
                    else:
                        continue
                    if data and "environmental_modes" in data:
                        return data
                except Exception:
                    pass
                    pass

        # Fallback explicit default specification
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
        """
        Evaluates dynamic environment context z_i(t) for a device at position p_i and slot t.
        Returns: event_mode (str), arrival_multiplier (float).
        """
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
        """
        Generates task conditional on environment context z_i(t) and application profile.
        """
        if current_time is None:
            current_time = float(slot)

        mode_name, surge = self.get_device_environment_state(device.position[:2], slot)
        modes = self.profiles.get("environmental_modes", {})
        mode_cfg = modes.get(mode_name, modes.get("ROUTINE"))

        prob = min(1.0, float(mode_cfg.get("base_generation_rate", 0.20)) * surge)

        # Event trigger check
        if deterministic:
            # Deterministic trigger mode for test reproducibility
            triggered = (prob > 0.0)
        else:
            triggered = (rng.uniform() < prob)

        if not triggered:
            return None

        # Sample task parameters conditional on mode profile
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


class StaticIoTDevice:
    """
    Represents a geographically static ground IoT device at p_i = (x_i, y_i, 0).
    Task generation is causally triggered by local environmental events.
    """

    def __init__(self, device_id, position, cpu_freq=IOT_CPU_FREQUENCY, tx_power=IOT_TX_POWER,
                 device_class="ENVIRONMENTAL_SENSOR"):
        self.id = device_id
        self.position = np.array(position, dtype=float)  # (x_i, y_i, 0)
        self.cpu_freq = float(cpu_freq)
        self.tx_power = float(tx_power)
        self.device_class = device_class

        self.cluster_id = -1
        self.current_task = None
        self.served_this_slot = False
        self.all_tasks = []

    def reset(self):
        """Reset device state for new episode."""
        self.current_task = None
        self.served_this_slot = False
        self.all_tasks.clear()

    def generate_task(self, slot, rng, env_manager=None, current_time=None, deterministic=False):
        """
        Generate heterogeneous task Omega_i(t) using environment state manager.
        If current_task is already pending or uploading, remains in progress unless finished.
        """
        if current_time is None:
            current_time = float(slot)

        # Check if existing task is active
        if self.current_task is not None and not self.current_task.is_finished:
            return self.current_task

        if env_manager is not None:
            new_task = env_manager.generate_task_for_device(
                self, slot, rng, current_time=current_time, deterministic=deterministic
            )
        else:
            # Fallback nominal Bernoulli generator
            prob = float(TASK_GENERATION_PROB)
            if rng.uniform() < prob:
                d_size = rng.uniform(TASK_DATA_SIZE_RANGE[0], TASK_DATA_SIZE_RANGE[1])
                c_per_bit = rng.uniform(CYCLES_PER_BIT_RANGE[0], CYCLES_PER_BIT_RANGE[1])
                ddl = rng.uniform(TASK_DEADLINE_RANGE[0], TASK_DEADLINE_RANGE[1])
                new_task = IoTTask(
                    device_id=self.id,
                    data_size_bits=d_size,
                    cycles_per_bit=c_per_bit,
                    deadline_s=ddl,
                    slot_generated=slot,
                    cluster_id=self.cluster_id,
                    global_time_gen=current_time,
                    event_mode="ROUTINE"
                )
            else:
                new_task = None

        self.current_task = new_task
        if new_task is not None:
            self.all_tasks.append(new_task)

        return self.current_task


class LegacyBernoulliDevice(StaticIoTDevice):
    """
    Legacy Ablation: Device using fixed stationary Bernoulli arrivals.
    Retained solely for ablation benchmarking against the environment-dependent generator.
    """

    def __init__(self, device_id, position, nominal_rate=0.7, **kwargs):
        super().__init__(device_id, position, **kwargs)
        self.nominal_rate = float(nominal_rate)

    def generate_task(self, slot, rng, env_manager=None, current_time=None, **kwargs):
        if self.current_task is not None and not self.current_task.is_finished:
            return self.current_task
        if rng.uniform() < self.nominal_rate:
            d_size = rng.uniform(TASK_DATA_SIZE_RANGE[0], TASK_DATA_SIZE_RANGE[1])
            c_per_bit = rng.uniform(CYCLES_PER_BIT_RANGE[0], CYCLES_PER_BIT_RANGE[1])
            ddl = rng.uniform(TASK_DEADLINE_RANGE[0], TASK_DEADLINE_RANGE[1])
            task = IoTTask(
                device_id=self.id,
                data_size_bits=d_size,
                cycles_per_bit=c_per_bit,
                deadline_s=ddl,
                slot_generated=slot,
                cluster_id=self.cluster_id,
                global_time_gen=float(slot) if current_time is None else float(current_time)
            )
            self.current_task = task
            self.all_tasks.append(task)
        else:
            self.current_task = None
        return self.current_task
