"""
Multi-Agent Environment for Trust- and Fault-Aware Multi-UAV Edge IoT Networks
=================================================================================
Target Journal: IEEE Internet of Things Journal
Paper Title: Trust- and Fault-Aware Action-Masked MAPPO for Dynamic Multi-UAV Enabled
             IoT Mobile Edge Computing with Adaptive Task Offloading and Trajectory Optimization

Features:
  - Static ground IoT devices with variable task generation
  - Pre-deployment workload-aware IoT clustering (weighted K-Means)
  - Dynamic UAV-cluster association x_{m,c}(t) in {0, 1}
  - A2G LoS/NLoS elevation-dependent communication channel
  - Unfinished computation workload queue model Q_m(t+1)
  - Rotary-wing UAV propulsion, CPU computation, and U2U coordination energy model
  - Authentication trust tau_m(t) and operational availability phi_m(t)
  - Feasible state-dependent action masking M_m(t)
  - Multi-objective normalized reward function (Problem P1)
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.channel_model import ChannelModel
from src.clustering import WorkloadAwareClusterManager
from src.trust_fault_manager import TrustFaultManager


class IoTDevice:
    """Represents a static ground IoT device generating computation tasks."""
    def __init__(self, device_id, position, cpu_freq=IOT_CPU_FREQUENCY, tx_power=IOT_TX_POWER):
        self.id = device_id
        self.position = np.array(position, dtype=float)
        self.cpu_freq = cpu_freq
        self.tx_power = tx_power
        self.cluster_id = -1
        self.latency = 0.0
        self.served_this_slot = False
        self.current_task = None

    def reset(self):
        self.latency = 0.0
        self.served_this_slot = False
        self.current_task = None

    def generate_task(self, rng):
        """Generate computation task Omega_i(t) = {D_i(t), C_i(t), delta_i(t)} (Eq. 14)."""
        if rng.uniform() < TASK_GENERATION_PROB:
            data_size = rng.uniform(TASK_DATA_SIZE_RANGE[0], TASK_DATA_SIZE_RANGE[1])
            cpu_cycles = rng.uniform(TASK_CPU_CYCLES_RANGE[0], TASK_CPU_CYCLES_RANGE[1])
            deadline = rng.uniform(TASK_DEADLINE_RANGE[0], TASK_DEADLINE_RANGE[1])
            self.current_task = {
                'data_size': data_size,
                'cpu_cycles': cpu_cycles,
                'deadline': deadline,
                'workload': data_size * cpu_cycles,
                'generated_slot': 0
            }
        else:
            self.current_task = None
        return self.current_task


class UAV:
    """Represents a rotary-wing UAV-MEC server."""
    def __init__(self, uav_id, position, profile=None):
        self.id = uav_id
        self.position = np.array(position, dtype=float)
        self.initial_position = np.array(position, dtype=float)
        profile = profile or UAV_PROFILES[uav_id % len(UAV_PROFILES)]
        self.profile = profile

        self.cpu_freq = profile['cpu_freq']
        self.bandwidth = profile['bandwidth']
        self.altitude = profile['altitude']
        self.max_displacement = profile['max_displacement']
        self.battery = 10000.0  # Joules
        self.queue_backlog = 0.0  # CPU cycles
        self.total_energy = 0.0
        self.assigned_cluster = -1

    def reset(self, initial_pos=None):
        if initial_pos is not None:
            self.position = np.array(initial_pos, dtype=float)
        else:
            self.position = self.initial_position.copy()
        self.battery = 10000.0
        self.queue_backlog = 0.0
        self.total_energy = 0.0
        self.assigned_cluster = -1


class MultiUAVMECEnv:
    """
    Multi-UAV Mobile Edge Computing Environment for IEEE IoT-J Paper.
    """

    def __init__(self, num_users=NUM_USERS, num_uavs=NUM_UAVS, num_jammers=0, seed=42):
        self.num_users = num_users
        self.num_uavs = num_uavs
        self.n_agents = num_uavs
        self.num_active = num_uavs
        self.num_clusters = NUM_CLUSTERS
        self.rng = np.random.RandomState(seed)
        self.current_slot = 0

        # Initialize Channel Model and Trust/Fault Manager
        self.channel_model = ChannelModel()
        self.trust_manager = TrustFaultManager(n_uavs=num_uavs, random_state=seed)
        self.cluster_manager = WorkloadAwareClusterManager(n_clusters=NUM_CLUSTERS, lambda_w=LAMBDA_W, random_state=seed)

        # Initialize IoT devices randomly in service area
        self.devices = []
        for i in range(num_users):
            angle = self.rng.uniform(0, 2 * np.pi)
            r = self.rng.uniform(10, AREA_RADIUS)
            pos = [r * np.cos(angle), r * np.sin(angle), 0.0]
            dev = IoTDevice(device_id=i, position=pos)
            self.devices.append(dev)

        # Workload-Aware Pre-deployment IoT Clustering (Eqs. 1–3)
        uav_altitudes = [UAV_PROFILES[m % len(UAV_PROFILES)]['altitude'] for m in range(num_uavs)]
        init_positions = self.cluster_manager.get_initial_uav_positions(self.devices, uav_altitudes)

        # Initialize Heterogeneous UAVs at cluster centroids
        self.uavs = [
            UAV(uav_id=m, position=init_positions[m], profile=UAV_PROFILES[m % len(UAV_PROFILES)])
            for m in range(num_uavs)
        ]

        # Dimension specifications for RL agent
        self.agent_action_dim = 5  # dx, dy, allocated_cpu, cluster_assoc, offloading_ratio
        self.local_obs_dim = 15
        self.global_state_dim = self.n_agents * self.local_obs_dim + 10

    def reset(self, seed=None):
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.current_slot = 0
        self.trust_manager.reset()

        for dev in self.devices:
            dev.reset()
            dev.generate_task(self.rng)

        # Re-cluster IoT devices
        uav_altitudes = [u.altitude for u in self.uavs]
        init_positions = self.cluster_manager.get_initial_uav_positions(self.devices, uav_altitudes)

        for m, uav in enumerate(self.uavs):
            uav.reset(initial_pos=init_positions[m])


        return self._get_obs_and_state()

    def get_action_mask(self):
        """Generate feasible state-dependent action mask matrix M_m(t)."""
        uav_positions = np.array([u.position for u in self.uavs])
        uav_batteries = np.array([u.battery for u in self.uavs])
        uav_capacities = np.array([u.cpu_freq for u in self.uavs])
        return self.trust_manager.compute_action_mask(
            uav_positions, uav_batteries, uav_capacities, action_dim=self.agent_action_dim
        )

    def _get_obs_and_state(self):
        """Construct local observations o_m(t) and global state s_t."""
        eligibility = self.trust_manager.step_states(
            self.current_slot, [u.battery for u in self.uavs]
        )

        obs_all = []
        for m, uav in enumerate(self.uavs):
            obs = np.array([
                uav.position[0] / AREA_RADIUS,
                uav.position[1] / AREA_RADIUS,
                uav.position[2] / UAV_ALT_MAX,
                uav.battery / 10000.0,
                uav.queue_backlog / 1e10,
                uav.cpu_freq / 3.5e9,
                float(uav.assigned_cluster) / max(self.num_clusters, 1),
                float(self.trust_manager.tau[m]),
                float(self.trust_manager.phi[m]),
                float(eligibility[m]),
                0.0, 0.0, 0.0, 0.0, 0.0
            ], dtype=np.float32)
            obs_all.append(obs)

        obs_all = np.array(obs_all, dtype=np.float32)
        global_state = np.concatenate([obs_all.flatten(), np.zeros(10, dtype=np.float32)])
        return obs_all, global_state

    def step(self, actions):
        """
        Execute environment step under action-masked decisions.
        """
        self.current_slot += 1
        actions = np.array(actions, dtype=float)

        # 1. Update UAV Trajectories
        for m, uav in enumerate(self.uavs):
            if self.trust_manager.eligibility[m] == 1:
                dx = np.clip(actions[m, 0], -1.0, 1.0) * uav.max_displacement
                dy = np.clip(actions[m, 1], -1.0, 1.0) * uav.max_displacement
                uav.position[0] = np.clip(uav.position[0] + dx, -AREA_RADIUS, AREA_RADIUS)
                uav.position[1] = np.clip(uav.position[1] + dy, -AREA_RADIUS, AREA_RADIUS)
                uav.assigned_cluster = int(np.clip(actions[m, 3], 0, self.num_clusters - 1))

        # 2. Task Offloading & Queue Processing
        completed_tasks = 0
        total_latency = 0.0
        deadline_violations = 0

        for dev in self.devices:
            dev.served_this_slot = False
            task = dev.current_task
            if task is None:
                continue

            # Identify serving UAV based on assigned cluster
            c_id = dev.cluster_id
            serving_uavs = [m for m, u in enumerate(self.uavs) if u.assigned_cluster == c_id and self.trust_manager.eligibility[m] == 1]

            if len(serving_uavs) > 0:
                target_uav_idx = serving_uavs[0]
                uav = self.uavs[target_uav_idx]

                # A2G Uplink Delay T_{i,m}^{up}(t)
                t_up = self.channel_model.compute_uploading_delay(
                    task['data_size'], dev.position, uav.position, dev.tx_power, uav.bandwidth / 10.0
                )

                # CPU Execution Delay T_{i,m}^{exe}(t)
                allocated_cpu = uav.cpu_freq * float(np.clip(actions[target_uav_idx, 2], 0.1, 1.0))
                t_exe = task['workload'] / max(allocated_cpu, 1e6)

                # Queueing Delay T_m^{que}(t)
                t_que = uav.queue_backlog / max(uav.cpu_freq, 1e6)
                uav.queue_backlog = max(0.0, uav.queue_backlog - uav.cpu_freq * SLOT_DURATION) + task['workload']

                # Total Latency T_{i,m}^{tot}(t)
                t_tot = t_up + t_que + t_exe
                dev.latency = t_tot
                total_latency += t_tot

                if t_tot <= task['deadline']:
                    completed_tasks += 1
                    dev.served_this_slot = True
                else:
                    deadline_violations += 1
            else:
                # Fallback to local computation
                t_local = task['workload'] / dev.cpu_freq
                dev.latency = t_local
                total_latency += t_local
                if t_local > task['deadline']:
                    deadline_violations += 1

            # Generate task for next slot
            dev.generate_task(self.rng)

        # 3. Compute UAV Energy Expenditure (Propulsion + Computation + U2U)
        total_fleet_energy = 0.0
        for m, uav in enumerate(self.uavs):
            # Rotary-wing propulsion power (Eq. 24)
            v_m = uav.max_displacement / SLOT_DURATION
            p_fly = P1_BLADE + P2_INDUCED * (1.0 + 3.0 * (v_m**2) / (ROTOR_TIP_SPEED**2))
            e_fly = p_fly * SLOT_DURATION
            e_comp = EFFECTIVE_CAPACITANCE * (uav.cpu_freq**3) * SLOT_DURATION
            e_u2u = 5.0 * SLOT_DURATION  # Coordination energy

            e_step = e_fly + e_comp + e_u2u
            uav.battery = max(0.0, uav.battery - e_step)
            uav.total_energy += e_step
            total_fleet_energy += e_step

        # 4. Multi-Objective Reward Calculation (Problem P1)
        r_latency = -OMEGA_T * (total_latency / max(self.num_users, 1))
        r_energy = -OMEGA_E * (total_fleet_energy / (self.num_uavs * 500.0))
        r_task = 0.1 * completed_tasks - 0.05 * deadline_violations

        reward = r_latency + r_energy + r_task
        rewards = np.full(self.n_agents, reward, dtype=np.float32)

        done = (self.current_slot >= NUM_TIME_SLOTS)
        next_obs, next_state = self._get_obs_and_state()

        info = {
            'completed_tasks': completed_tasks,
            'total_latency': total_latency,
            'total_energy': total_fleet_energy,
            'deadline_violations': deadline_violations,
        }

        return next_obs, next_state, rewards, done, info