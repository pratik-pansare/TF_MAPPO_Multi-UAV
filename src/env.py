"""
Multi-Agent Environment for Trust- and Fault-Aware Multi-UAV Edge IoT Networks
=================================================================================
Target Journal: IEEE Internet of Things Journal
Paper Title: Trust- and Fault-Aware Action-Masked MAPPO for Dynamic Multi-UAV Enabled
             IoT Mobile Edge Computing with Adaptive Task Offloading and Trajectory Optimization

Integrated Modular Architecture (Aligned with Supervisor Directives):
  1. Static ground IoT devices with environment-dependent causal task generation (src.task)
  2. Pure spatial pre-deployment clustering & BS commissioning (src.clustering)
  3. Dynamic physical coverage-driven link set I_{m,cov}(t) based on 3D distance and channel quality
  4. Hard radio concurrency limit sum_i u_{im}(t) <= K_m^comm and exclusivity sum_m u_{im}(t) <= 1
  5. Multi-slot discrete-event uplink transmission & onboard CPU computation queues (src.computation)
  6. Rotary-wing propulsion, computation, & U2U energy model (src.energy)
  7. Certificate authentication trust tau_m(t) & operational availability phi_m(t) (src.trust, src.faults)
  8. State-dependent action feasibility masking & joint safety shield (src.action_mask)
  9. Multi-objective Problem P1 reward function with exact component logging
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.task import StaticIoTDevice, TaskStatus, EnvironmentStateManager
from src.clustering import SpatialClusterManager, BSCommissioningManager, IoTDeviceRegistrationReport
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
        self.assigned_cluster = 0  # 0 = Idle, 1..C = Target spatial cluster region
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
    Multi-UAV Mobile Edge Computing Gym-like Environment with coverage-driven communication.
    """

    def __init__(self, num_users=NUM_USERS, num_uavs=NUM_UAVS, num_clusters=NUM_CLUSTERS,
                 seed=42, area_radius=AREA_RADIUS, scenario="SCENARIO_A_ROUTINE",
                 k_comm=K_COMM_DEFAULT, max_candidate_pool=MAX_CANDIDATE_POOL):
        self.num_users = num_users
        self.num_uavs = num_uavs
        self.n_agents = num_uavs
        self.num_active = num_uavs
        self.num_clusters = num_clusters
        self.area_radius = float(area_radius)
        self.scenario_name = scenario
        self.k_comm = int(k_comm)
        self.max_candidate_pool = int(max_candidate_pool)
        self.rng = np.random.RandomState(seed)
        self.current_slot = 0
        self.global_time = 0.0

        # Domain Managers
        self.commissioning_mgr = BSCommissioningManager()
        self.channel_model = ChannelModel()
        self.trust_manager = TrustManager(n_uavs=num_uavs, cert_duration=CERTIFICATE_DURATION)
        self.fault_manager = FaultManager(n_uavs=num_uavs, fault_prob=FAULT_PROBABILITY, random_state=seed)
        self.cluster_manager = SpatialClusterManager(n_clusters=num_clusters, max_coverage_radius=MAX_SERVICE_RADIUS, random_state=seed)
        self.computation_manager = ComputationManager(n_uavs=num_uavs)
        self.energy_manager = EnergyManager(kappa_m=EFFECTIVE_CAPACITANCE)
        self.env_state_mgr = EnvironmentStateManager(scenario_name=scenario, seed=seed)
        self.action_mask_generator = ActionMaskGenerator(
            n_uavs=num_uavs, n_clusters=num_clusters,
            max_service_radius=MAX_SERVICE_RADIUS, k_comm=self.k_comm,
            max_candidate_pool=self.max_candidate_pool
        )
        self.safety_shield = JointSafetyShield(
            n_uavs=num_uavs, area_radius=self.area_radius, k_comm=self.k_comm
        )

        # Dimension Specifications
        # Action: [cluster_target (1), offload_priorities (max_candidate_pool), dx, dy, dz (3), cpu_frac (1)]
        self.agent_action_dim = 1 + self.max_candidate_pool + 3 + 1
        # Action Mask: [cluster_mask (C + 1), candidate_mask (max_candidate_pool)]
        self.action_mask_dim = (num_clusters + 1) + self.max_candidate_pool
        self.local_obs_dim = 16
        self.global_state_dim = self.n_agents * self.local_obs_dim + 12

        # Initialize Static Ground IoT Devices & Pre-deployment Commissioning
        self.devices = []
        for i in range(num_users):
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            r = self.rng.uniform(10.0, self.area_radius)
            pos = [r * np.cos(angle), r * np.sin(angle), 0.0]
            dev = StaticIoTDevice(device_id=i, position=pos)
            self.devices.append(dev)

            # Pre-deployment BS Registration (identity + location only)
            report = IoTDeviceRegistrationReport(device_id=i, position=pos, timestamp=0.0)
            self.commissioning_mgr.register_device(report)

        # Pre-deployment Spatial Clustering (Unweighted K-Means on coordinates)
        uav_altitudes = [UAV_PROFILES[m % len(UAV_PROFILES)]['altitude'] for m in range(num_uavs)]
        init_positions = self.cluster_manager.get_initial_uav_positions(self.devices, uav_altitudes)

        # Initialize Heterogeneous UAV Fleet
        self.uavs = [
            UAV(uav_id=m, position=init_positions[m], profile=UAV_PROFILES[m % len(UAV_PROFILES)])
            for m in range(num_uavs)
        ]

        # Active candidate caching
        self.cached_candidates = {m: [] for m in range(num_uavs)}
        self.last_reward_components = {}
        self.last_scheduled_links = {m: [] for m in range(num_uavs)}

    def reset(self, seed=None):
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.current_slot = 0
        self.global_time = 0.0

        self.trust_manager.reset()
        self.fault_manager.reset(seed=seed)
        self.computation_manager.reset()
        self.env_state_mgr = EnvironmentStateManager(scenario_name=self.scenario_name, seed=seed or 42)

        for dev in self.devices:
            dev.reset()
            dev.generate_task(slot=0, rng=self.rng, env_manager=self.env_state_mgr, current_time=0.0)

        # Pre-deployment Spatial Clustering
        uav_altitudes = [u.altitude for u in self.uavs]
        init_positions = self.cluster_manager.get_initial_uav_positions(self.devices, uav_altitudes)

        for m, uav in enumerate(self.uavs):
            uav.reset(initial_pos=init_positions[m])

        self._update_candidate_cache()
        return self._get_obs_and_state()

    def _update_candidate_cache(self):
        """
        Evaluates physical coverage I_{m,cov}(t) for all UAVs based on 3D distance,
        minimum rate threshold R_min, and UAV service eligibility e_m(t) = 1.
        Stores up to max_candidate_pool candidates per UAV.
        """
        eligibility = self.fault_manager.eligibility
        self.cached_candidates = {m: [] for m in range(self.num_uavs)}

        for m, uav in enumerate(self.uavs):
            if eligibility[m] == 0 or uav.battery <= MIN_BATTERY_RESERVE:
                continue

            candidates = []
            for dev in self.devices:
                if dev.current_task is None or dev.current_task.is_finished:
                    continue

                d_3d = self.channel_model.compute_distance_3d(dev.position, uav.position)
                if d_3d <= MAX_SERVICE_RADIUS:
                    # Orthogonal bandwidth under nominal concurrency cap
                    rate = self.channel_model.compute_uplink_rate(
                        dev.position, uav.position, dev.tx_power, uav.bandwidth / max(1, self.k_comm)
                    )
                    if rate >= R_MIN_UPLINK:
                        candidates.append((dev, rate, d_3d, dev.current_task.deadline))

            # Prioritize earliest deadline tasks (EDF order), then higher rate
            candidates.sort(key=lambda item: (item[3], -item[1]))
            self.cached_candidates[m] = candidates[:self.max_candidate_pool]

    @property
    def cluster_centroids(self):
        return self.cluster_manager.centroids

    def get_action_mask(self):
        """
        Generate action mask of shape (M, (C + 1) + max_candidate_pool):
          - Cluster target mask (C + 1)
          - Candidate scheduling mask (max_candidate_pool)
        """
        self._update_candidate_cache()

        uav_positions = np.array([u.position for u in self.uavs])
        uav_batteries = np.array([u.battery for u in self.uavs])
        cluster_centroids = self.cluster_manager.centroids
        tau_states = self.trust_manager.tau
        phi_states = self.fault_manager.phi
        eligibility = self.fault_manager.eligibility

        clu_mask = self.action_mask_generator.compute_cluster_mask(
            uav_positions=uav_positions,
            uav_batteries=uav_batteries,
            cluster_centroids=cluster_centroids,
            eligibility_states=eligibility,
            tau_states=tau_states,
            phi_states=phi_states
        )

        cand_counts = [len(self.cached_candidates[m]) for m in range(self.num_uavs)]
        cand_mask = self.action_mask_generator.compute_candidate_mask(cand_counts)

        # Concatenate cluster mask and candidate mask
        full_mask = np.concatenate([clu_mask, cand_mask], axis=1)
        return full_mask

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

            # Number of covered candidate tasks currently in range of this UAV
            n_covered = len(self.cached_candidates[m])

            obs = np.array([
                uav.position[0] / self.area_radius,
                uav.position[1] / self.area_radius,
                uav.position[2] / UAV_ALT_MAX,
                uav.battery / 150000.0,
                self.computation_manager.queues[m] / 1e10,
                uav.cpu_freq / 3.5e9,
                float(uav.assigned_cluster) / max(self.num_clusters, 1),
                float(tau_states[m]),
                float(self.fault_manager.phi[m]),
                float(eligibility[m]),
                dx_centroid,
                dy_centroid,
                float(n_covered) / float(self.max_candidate_pool),
                float(self.current_slot) / max(NUM_TIME_SLOTS, 1),
                float(uav.velocity[0]) / 50.0,
                float(uav.velocity[1]) / 50.0,
            ], dtype=np.float32)
            obs_all.append(obs)

        obs_all = np.array(obs_all, dtype=np.float32)

        total_backlog = float(np.sum(self.computation_manager.queues)) / 1e10
        total_battery = float(np.sum([u.battery for u in self.uavs])) / (self.num_uavs * 150000.0)
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

        if actions.shape[1] == 5:
            # Backward-compatible 5-dim action format [cluster, dx, dy, dz, cpu]
            proposed_clusters = np.round(actions[:, 0]).astype(int)
            proposed_offload_scores = np.ones((self.num_uavs, self.max_candidate_pool), dtype=float)
            proposed_mov = actions[:, 1:4]
            proposed_cpu_frac = np.clip(actions[:, 4], 0.0, 1.0)
        else:
            # Full 4-head factorized action format [cluster, offload_scores (K_pool), dx, dy, dz, cpu]
            proposed_clusters = np.round(actions[:, 0]).astype(int)
            idx_off_end = 1 + self.max_candidate_pool
            proposed_offload_scores = actions[:, 1:idx_off_end]
            proposed_mov = actions[:, idx_off_end:idx_off_end + 3]
            proposed_cpu_frac = np.clip(actions[:, idx_off_end + 3], 0.0, 1.0)

        proposed_clusters = np.clip(proposed_clusters, 0, self.num_clusters)

        # -------------------------------------------------------------
        # 1. Macro Cluster Target Assignment
        # -------------------------------------------------------------
        uav_positions = np.array([u.position for u in self.uavs])
        eligibility = self.fault_manager.eligibility

        executed_clusters = self.safety_shield.resolve_cluster_conflicts(
            proposed_clusters=proposed_clusters,
            uav_positions=uav_positions,
            cluster_centroids=self.cluster_manager.centroids,
            eligibility=eligibility
        )
        cluster_intervened = bool(np.any(executed_clusters != proposed_clusters))

        for m, uav in enumerate(self.uavs):
            uav.assigned_cluster = executed_clusters[m]

        # -------------------------------------------------------------
        # 2. Continuous-Time Trajectory Safety Shielding
        # -------------------------------------------------------------
        proposed_displacements = np.zeros((self.num_uavs, 3), dtype=float)
        for m, uav in enumerate(self.uavs):
            if eligibility[m] == 1:
                norm_u = np.linalg.norm(proposed_mov[m])
                scale = 1.0 / max(1.0, norm_u)
                disp = proposed_mov[m] * scale * uav.max_displacement
                proposed_displacements[m] = disp

        safe_displacements, traj_intervened = self.safety_shield.shield_trajectories(
            uav_positions=uav_positions,
            proposed_displacements=proposed_displacements
        )
        shield_intervened = bool(traj_intervened or cluster_intervened)

        for m, uav in enumerate(self.uavs):
            if eligibility[m] == 1:
                uav.position += safe_displacements[m]
                uav.velocity = safe_displacements[m] / SLOT_DURATION
            else:
                uav.velocity = np.zeros(3)

        # -------------------------------------------------------------
        # 3. Dynamic Coverage-Driven Link Scheduling under Concurrency Cap
        # -------------------------------------------------------------
        self._update_candidate_cache()

        proposed_links_by_uav = {}
        candidate_rates_by_uav = {}

        for m in range(self.num_uavs):
            cands = self.cached_candidates[m]
            candidate_rates_by_uav[m] = {cand[0].id: cand[1] for cand in cands}

            if len(cands) == 0:
                proposed_links_by_uav[m] = []
                continue

            # Rank candidates by policy priority score across all max_candidate_pool slots
            scores = proposed_offload_scores[m]
            ranked_indices = np.argsort(-scores)  # Highest score first

            # Select up to k_comm candidates corresponding to valid candidate indices
            selected_dev_ids = [cands[idx][0].id for idx in ranked_indices[:self.k_comm] if idx < len(cands)]
            proposed_links_by_uav[m] = selected_dev_ids

        # Enforce link exclusivity (sum_m u_im <= 1) and per-UAV concurrency (sum_i u_im <= K_m^comm)
        scheduled_links = self.safety_shield.resolve_link_conflicts(
            proposed_links_by_uav=proposed_links_by_uav,
            candidate_rates_by_uav=candidate_rates_by_uav,
            k_comm=self.k_comm
        )
        self.last_scheduled_links = scheduled_links

        # -------------------------------------------------------------
        # 4. Multi-Slot Uplink Transmission & Queue Execution
        # -------------------------------------------------------------
        active_uplink_pairs = []
        active_rates = {}

        for m in range(self.num_uavs):
            uav = self.uavs[m]
            link_dev_ids = scheduled_links.get(m, [])
            n_active_links = len(link_dev_ids)
            if n_active_links == 0:
                continue

            ortho_bw = uav.bandwidth / n_active_links
            for dev_id in link_dev_ids:
                dev = self.devices[dev_id]
                task = dev.current_task
                if task is not None and not task.is_finished:
                    rate = self.channel_model.compute_uplink_rate(
                        dev.position, uav.position, dev.tx_power, ortho_bw
                    )
                    active_uplink_pairs.append((m, task))
                    active_rates[task.task_id] = rate
                    dev.served_this_slot = True

        # Process multi-slot upload progress
        newly_queued, upload_expired = self.computation_manager.process_uplinks(
            active_links=active_uplink_pairs,
            rates=active_rates,
            current_time_s=self.global_time,
            delta_t=SLOT_DURATION
        )

        # Onboard CPU computation
        allocated_cpus = [
            self.uavs[m].cpu_freq * float(np.clip(proposed_cpu_frac[m], 0.2, 1.0))
            if eligibility[m] == 1 else 0.0
            for m in range(self.num_uavs)
        ]

        slot_completed, slot_expired = self.computation_manager.process_slot(
            uav_cpu_frequencies=allocated_cpus,
            current_time_s=self.global_time,
            delta_t=SLOT_DURATION
        )

        total_slot_expired = slot_expired + upload_expired

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
            e_u2u = self.energy_manager.compute_u2u_energy(0.01)

            e_step = e_fly + e_comp + e_u2u
            uav.battery = max(0.0, uav.battery - e_step)
            uav.total_energy += e_step
            total_fleet_energy += e_step
            propulsion_energy += e_fly
            computation_energy += e_comp
            u2u_energy += e_u2u

        # -------------------------------------------------------------
        # 6. Generate Stochastic Tasks for Next Slot via Environment State
        # -------------------------------------------------------------
        new_tasks_generated = 0
        for dev in self.devices:
            new_task = dev.generate_task(
                slot=self.current_slot,
                rng=self.rng,
                env_manager=self.env_state_mgr,
                current_time=self.global_time
            )
            if new_task is not None and getattr(new_task, 'slot_generated', None) == self.current_slot:
                new_tasks_generated += 1

        # -------------------------------------------------------------
        # 7. Multi-Objective Reward Calculation (Problem P1)
        # -------------------------------------------------------------
        E_ref = max(self.num_uavs * 200.0, 1.0)
        T_ref = 2.0

        completed_latencies = [t.total_latency for t in slot_completed]
        if len(completed_latencies) > 0:
            mean_slot_latency = float(np.mean(completed_latencies))
        else:
            mean_slot_latency = T_ref

        unserved_count = sum(
            1 for dev in self.devices
            if dev.current_task is not None and not dev.current_task.is_finished and not dev.served_this_slot
        )

        r_energy = -OMEGA_E * (total_fleet_energy / E_ref)
        r_latency = -OMEGA_T * (mean_slot_latency / T_ref)
        r_completion = 0.8 * len(slot_completed) - 0.2 * len(total_slot_expired) - 0.05 * unserved_count
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

        # Count active communication links across the fleet
        active_links_count = sum(len(links) for links in scheduled_links.values())

        info = {
            'completed_tasks_step': len(slot_completed),
            'expired_tasks_step': len(total_slot_expired),
            'generated_tasks_step': new_tasks_generated,
            'active_links_count': active_links_count,
            'mean_slot_latency': mean_slot_latency,
            'total_energy': total_fleet_energy,
            'propulsion_energy': propulsion_energy,
            'computation_energy': computation_energy,
            'u2u_energy': u2u_energy,
            'shield_intervened': shield_intervened,
            'slot_completed_tasks': slot_completed,
            'reward_components': self.last_reward_components,
            'scheduled_links': scheduled_links,
        }

        return next_obs, next_state, rewards, done, info
