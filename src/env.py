"""
Multi-Agent Environment for Trust- and Fault-Aware Multi-UAV Edge IoT Networks
=================================================================================
Target Journal: IEEE Internet of Things Journal
Paper Title: Trust- and Fault-Aware Action-Masked MAPPO for Dynamic Multi-UAV Enabled
             IoT Mobile Edge Computing with Adaptive Task Offloading and Trajectory Optimization

Integrated Modular Architecture:
  1. Static ground IoT devices with causal stochastic task generation (src.task)
  2. Pre-deployment workload-aware IoT clustering & BS commissioning (src.clustering)
  3. Dynamic UAV-cluster association x_{m,c}(t) in {0, 1} with exclusivity enforcement
  4. Elevation-dependent A2G channel model with positive NLoS attenuation (src.channel)
  5. Multi-slot discrete-event CPU computation queues on each UAV (src.computation)
  6. Rotary-wing propulsion (with hover power), computation, & U2U energy model (src.energy)
  7. Certificate authentication trust tau_m(t) & operational availability phi_m(t) (src.trust, src.faults)
  8. State-dependent discrete action feasibility masking & joint safety shield (src.action_mask)
  9. Multi-objective Problem P1 reward function with exact component logging
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.task import StaticIoTDevice, TaskStatus
from src.clustering import WorkloadAwareClusterManager, BSCommissioningManager, IoTDeviceRegistrationReport
from src.channel import ChannelModel
from src.computation import ComputationManager
from src.energy import EnergyManager
from src.trust import TrustManager
from src.faults import FaultManager
from src.action_mask import ActionMaskGenerator, JointSafetyShield


class UAV:
    """Represents a rotary-wing UAV-MEC server."""

    def __init__(self, uav_id, position, profile=None):
        self.id = uav_id
        self.position = np.array(position, dtype=float)
        self.initial_position = np.array(position, dtype=float)
        profile = profile or UAV_PROFILES[uav_id % len(UAV_PROFILES)]
        self.profile = profile

        self.cpu_freq = float(profile['cpu_freq'])
        self.bandwidth = float(profile['bandwidth'])
        self.altitude = float(profile['altitude'])
        self.max_displacement = float(profile['max_displacement'])
        self.battery = 150000.0  # Joules (150 kJ battery capacity for 500s mission)
        self.total_energy = 0.0
        self.assigned_cluster = 0  # 0 = Idle, 1..C = Assigned cluster
        self.velocity = np.zeros(3, dtype=float)

    def reset(self, initial_pos=None):
        if initial_pos is not None:
            self.position = np.array(initial_pos, dtype=float)
        else:
            self.position = self.initial_position.copy()
        self.battery = 150000.0
        self.total_energy = 0.0
        self.assigned_cluster = 0
        self.velocity = np.zeros(3, dtype=float)


class MultiUAVMECEnv:
    """
    Multi-UAV Mobile Edge Computing Gym-like Environment.
    """

    def __init__(self, num_users=NUM_USERS, num_uavs=NUM_UAVS, num_clusters=NUM_CLUSTERS,
                 seed=42, area_radius=AREA_RADIUS):
        self.num_users = num_users
        self.num_uavs = num_uavs
        self.n_agents = num_uavs
        self.num_active = num_uavs
        self.num_clusters = num_clusters
        self.area_radius = float(area_radius)
        self.rng = np.random.RandomState(seed)
        self.current_slot = 0
        self.global_time = 0.0

        # Domain Managers
        self.commissioning_mgr = BSCommissioningManager()
        self.channel_model = ChannelModel()
        self.trust_manager = TrustManager(n_uavs=num_uavs, cert_duration=CERTIFICATE_DURATION)
        self.fault_manager = FaultManager(n_uavs=num_uavs, fault_prob=FAULT_PROBABILITY, random_state=seed)
        self.cluster_manager = WorkloadAwareClusterManager(n_clusters=num_clusters, lambda_w=LAMBDA_W, random_state=seed)
        self.computation_manager = ComputationManager(n_uavs=num_uavs)
        self.energy_manager = EnergyManager(kappa_m=EFFECTIVE_CAPACITANCE)
        self.action_mask_generator = ActionMaskGenerator(n_uavs=num_uavs, n_clusters=num_clusters)
        self.safety_shield = JointSafetyShield(n_uavs=num_uavs, area_radius=self.area_radius)

        # Dimension Specifications
        self.agent_action_dim = 5  # [cluster_id (0..C), dx, dy, dz, cpu_fraction]
        self.action_mask_dim = num_clusters + 1  # Discrete cluster mask dimension (C + 1)
        self.local_obs_dim = 16
        self.global_state_dim = self.n_agents * self.local_obs_dim + 12

        # Initialize Static Ground IoT Devices & Commissioning
        self.devices = []
        for i in range(num_users):
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            r = self.rng.uniform(10.0, self.area_radius)
            pos = [r * np.cos(angle), r * np.sin(angle), 0.0]
            dev = StaticIoTDevice(device_id=i, position=pos)
            self.devices.append(dev)

            # Pre-deployment BS Registration
            report = IoTDeviceRegistrationReport(
                device_id=i,
                position=pos,
                nominal_task_rate=dev.nominal_rate,
                nominal_data_size=dev.nominal_data_size,
                nominal_cycles_per_bit=dev.nominal_cycles_per_bit,
                timestamp=0.0
            )
            self.commissioning_mgr.register_device(report)

        # Pre-deployment Spatial Workload-Aware Clustering (Lloyd's weighted iteration)
        uav_altitudes = [UAV_PROFILES[m % len(UAV_PROFILES)]['altitude'] for m in range(num_uavs)]
        init_positions = self.cluster_manager.get_initial_uav_positions(self.devices, uav_altitudes)

        # Initialize Heterogeneous UAV Fleet
        self.uavs = [
            UAV(uav_id=m, position=init_positions[m], profile=UAV_PROFILES[m % len(UAV_PROFILES)])
            for m in range(num_uavs)
        ]

        # Reward reconciliation log
        self.last_reward_components = {}

    def reset(self, seed=None):
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.current_slot = 0
        self.global_time = 0.0

        self.trust_manager.reset()
        self.fault_manager.reset(seed=seed)
        self.computation_manager.reset()

        for dev in self.devices:
            dev.reset()
            dev.generate_task(slot=0, rng=self.rng, current_time=0.0)

        # Re-cluster with seed reproducibility
        uav_altitudes = [u.altitude for u in self.uavs]
        init_positions = self.cluster_manager.get_initial_uav_positions(self.devices, uav_altitudes)

        for m, uav in enumerate(self.uavs):
            uav.reset(initial_pos=init_positions[m])

        return self._get_obs_and_state()

    def get_action_mask(self):
        """Generate discrete cluster selection feasibility mask of shape (M, C + 1)."""
        uav_positions = np.array([u.position for u in self.uavs])
        uav_batteries = np.array([u.battery for u in self.uavs])
        cluster_centroids = self.cluster_manager.centroids
        tau_states = self.trust_manager.tau
        phi_states = self.fault_manager.phi
        eligibility = self.fault_manager.eligibility

        return self.action_mask_generator.compute_cluster_mask(
            uav_positions=uav_positions,
            uav_batteries=uav_batteries,
            cluster_centroids=cluster_centroids,
            eligibility_states=eligibility,
            tau_states=tau_states,
            phi_states=phi_states
        )

    def _get_obs_and_state(self):
        """Construct local observation vectors o_m(t) and global state vector s_t."""
        tau_states = self.trust_manager.update_trust_states(self.current_slot)
        eligibility = self.fault_manager.update_fault_and_eligibility(
            self.current_slot, tau_states, [u.battery for u in self.uavs]
        )

        obs_all = []
        cluster_centroids = self.cluster_manager.centroids

        for m, uav in enumerate(self.uavs):
            c_idx = uav.assigned_cluster
            if c_idx > 0 and cluster_centroids is not None and c_idx - 1 < len(cluster_centroids):
                c_pos = cluster_centroids[c_idx - 1]
                dx_centroid = (c_pos[0] - uav.position[0]) / self.area_radius
                dy_centroid = (c_pos[1] - uav.position[1]) / self.area_radius
            else:
                dx_centroid = 0.0
                dy_centroid = 0.0

            # Count pending tasks in assigned cluster
            tasks_in_cluster = 0
            if c_idx > 0:
                for dev in self.devices:
                    if dev.cluster_id == c_idx - 1 and dev.current_task is not None:
                        tasks_in_cluster += 1

            obs = np.array([
                uav.position[0] / self.area_radius,
                uav.position[1] / self.area_radius,
                uav.position[2] / UAV_ALT_MAX,
                uav.battery / 10000.0,
                self.computation_manager.queues[m] / 1e10,
                uav.cpu_freq / 3.5e9,
                float(uav.assigned_cluster) / max(self.num_clusters, 1),
                float(tau_states[m]),
                float(self.fault_manager.phi[m]),
                float(eligibility[m]),
                dx_centroid,
                dy_centroid,
                float(tasks_in_cluster) / max(self.num_users, 1),
                float(self.current_slot) / max(NUM_TIME_SLOTS, 1),
                float(uav.velocity[0]) / 50.0,
                float(uav.velocity[1]) / 50.0,
            ], dtype=np.float32)
            obs_all.append(obs)

        obs_all = np.array(obs_all, dtype=np.float32)

        # Global features: overall fleet stats and total pending work
        total_backlog = float(np.sum(self.computation_manager.queues)) / 1e10
        total_battery = float(np.sum([u.battery for u in self.uavs])) / (self.num_uavs * 10000.0)
        active_uavs = float(np.sum(eligibility)) / self.num_uavs
        global_features = np.array([
            total_backlog, total_battery, active_uavs,
            float(self.current_slot) / max(NUM_TIME_SLOTS, 1),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)

        global_state = np.concatenate([obs_all.flatten(), global_features])
        return obs_all, global_state

    def step(self, actions):
        """
        Execute joint environment transition for interval [t*Delta_t, (t+1)*Delta_t].
        """
        self.current_slot += 1
        self.global_time += SLOT_DURATION
        actions = np.array(actions, dtype=float)

        proposed_clusters = np.round(actions[:, 0]).astype(int)
        proposed_clusters = np.clip(proposed_clusters, 0, self.num_clusters)
        proposed_mov = actions[:, 1:4]  # (M, 3) in [-1, 1]
        proposed_cpu_frac = np.clip(actions[:, 4], 0.0, 1.0)

        # -------------------------------------------------------------
        # 1. Joint Cluster Exclusivity Resolution (sum_m x_{m,c} <= 1)
        # -------------------------------------------------------------
        uav_positions = np.array([u.position for u in self.uavs])
        eligibility = self.fault_manager.eligibility
        cluster_centroids = self.cluster_manager.centroids

        executed_clusters = self.safety_shield.resolve_cluster_conflicts(
            proposed_clusters=proposed_clusters,
            uav_positions=uav_positions,
            cluster_centroids=cluster_centroids,
            eligibility=eligibility
        )

        for m, uav in enumerate(self.uavs):
            uav.assigned_cluster = executed_clusters[m]

        # -------------------------------------------------------------
        # 2. Continuous-Time Trajectory Safety Shielding
        # -------------------------------------------------------------
        proposed_displacements = np.zeros((self.num_uavs, 3), dtype=float)
        for m, uav in enumerate(self.uavs):
            if eligibility[m] == 1:
                # Scale normalized movement by max displacement
                norm_u = np.linalg.norm(proposed_mov[m])
                scale = 1.0 / max(1.0, norm_u)
                disp = proposed_mov[m] * scale * uav.max_displacement
                proposed_displacements[m] = disp

        safe_displacements, shield_intervened = self.safety_shield.shield_trajectories(
            uav_positions=uav_positions,
            proposed_displacements=proposed_displacements
        )

        for m, uav in enumerate(self.uavs):
            if eligibility[m] == 1:
                uav.position += safe_displacements[m]
                uav.velocity = safe_displacements[m] / SLOT_DURATION
            else:
                uav.velocity = np.zeros(3)

        # -------------------------------------------------------------
        # 3. Dynamic Task Admission & Uplink Transmission
        # -------------------------------------------------------------
        # Map cluster to serving UAV
        cluster_to_uav = {}
        for m, uav in enumerate(self.uavs):
            if uav.assigned_cluster > 0 and eligibility[m] == 1:
                cluster_to_uav[uav.assigned_cluster - 1] = m

        # Identify generated tasks per cluster
        tasks_to_admit = {m: [] for m in range(self.num_uavs)}
        devices_with_task = [dev for dev in self.devices if dev.current_task is not None]

        for dev in devices_with_task:
            task = dev.current_task
            c_id = dev.cluster_id
            if c_id in cluster_to_uav:
                serving_uav_idx = cluster_to_uav[c_id]
                tasks_to_admit[serving_uav_idx].append((dev, task))
            else:
                # No eligible UAV serving this cluster: task rejected
                task.status = TaskStatus.REJECTED

        # Upload admitted tasks over orthogonal bandwidth
        for m, dev_task_list in tasks_to_admit.items():
            if len(dev_task_list) == 0:
                continue
            uav = self.uavs[m]
            k_tasks = len(dev_task_list)
            ortho_bw = uav.bandwidth / max(1, k_tasks)

            for dev, task in dev_task_list:
                t_up = self.channel_model.compute_uploading_delay(
                    task.data_size, dev.position, uav.position, dev.tx_power, ortho_bw
                )
                self.computation_manager.admit_task(
                    uav_id=m,
                    task=task,
                    upload_delay_s=t_up,
                    current_time_s=self.global_time
                )
                dev.served_this_slot = True

        # -------------------------------------------------------------
        # 4. Onboard Multi-Slot Queue Processing (src.computation)
        # -------------------------------------------------------------
        allocated_cpus = [
            self.uavs[m].cpu_freq * float(np.clip(proposed_cpu_frac[m], 0.2, 1.0))
            if (eligibility[m] == 1 and self.uavs[m].assigned_cluster > 0) else 0.0
            for m in range(self.num_uavs)
        ]

        slot_completed, slot_expired = self.computation_manager.process_slot(
            uav_cpu_frequencies=allocated_cpus,
            current_time_s=self.global_time,
            delta_t=SLOT_DURATION
        )

        # -------------------------------------------------------------
        # 5. UAV Energy Expenditure (Propulsion + CPU + U2U)
        # -------------------------------------------------------------
        total_fleet_energy = 0.0
        propulsion_energy = 0.0
        computation_energy = 0.0
        u2u_energy = 0.0

        for m, uav in enumerate(self.uavs):
            speed = float(np.linalg.norm(uav.velocity))
            e_fly = self.energy_manager.compute_propulsion_power(speed) * SLOT_DURATION
            exec_time = min(SLOT_DURATION, 1.0)
            e_comp = self.energy_manager.compute_computation_energy(allocated_cpus[m], exec_time)
            # U2U signaling with neighboring active UAVs
            e_u2u = self.energy_manager.compute_u2u_energy(0.01)

            e_step = e_fly + e_comp + e_u2u
            uav.battery = max(0.0, uav.battery - e_step)
            uav.total_energy += e_step
            total_fleet_energy += e_step
            propulsion_energy += e_fly
            computation_energy += e_comp
            u2u_energy += e_u2u

        # -------------------------------------------------------------
        # 6. Generate Stochastic Tasks for Next Slot
        # -------------------------------------------------------------
        for dev in self.devices:
            dev.generate_task(slot=self.current_slot, rng=self.rng, current_time=self.global_time)

        # -------------------------------------------------------------
        # 7. Problem P1 Multi-Objective Reward Calculation
        # -------------------------------------------------------------
        # E_ref = M * 200 J/slot, T_ref = 2.0 s
        E_ref = max(self.num_uavs * 200.0, 1.0)
        T_ref = 2.0

        completed_latencies = [t.total_latency for t in slot_completed]
        mean_slot_latency = float(np.mean(completed_latencies)) if len(completed_latencies) > 0 else 0.0

        r_energy = -OMEGA_E * (total_fleet_energy / E_ref)
        r_latency = -OMEGA_T * (mean_slot_latency / T_ref)
        r_completion = 0.5 * len(slot_completed) - 0.2 * len(slot_expired)
        r_shield_pen = -0.1 if shield_intervened else 0.0

        step_reward = r_energy + r_latency + r_completion + r_shield_pen
        rewards = np.full(self.n_agents, step_reward, dtype=np.float32)

        self.last_reward_components = {
            'step': self.current_slot,
            'r_energy': float(r_energy),
            'r_latency': float(r_latency),
            'r_completion': float(r_completion),
            'r_shield_penalty': float(r_shield_pen),
            'total_step_reward': float(step_reward),
        }

        done = (self.current_slot >= NUM_TIME_SLOTS)
        next_obs, next_state = self._get_obs_and_state()

        info = {
            'completed_tasks_step': len(slot_completed),
            'expired_tasks_step': len(slot_expired),
            'mean_slot_latency': mean_slot_latency,
            'total_energy': total_fleet_energy,
            'propulsion_energy': propulsion_energy,
            'computation_energy': computation_energy,
            'u2u_energy': u2u_energy,
            'shield_intervened': shield_intervened,
            'reward_components': self.last_reward_components,
        }

        return next_obs, next_state, rewards, done, info
