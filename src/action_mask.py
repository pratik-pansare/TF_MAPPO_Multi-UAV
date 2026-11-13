"""
State-Dependent Action Feasibility Mask & Joint Safety Shield
============================================================
Paper Reference: Section III.F & Constraints (C1–C15)
Target Journal: IEEE Internet of Things Journal

Features:
  1. Discrete Action Feasibility Masking:
     - a_m^clu in {0, 1, ..., C}: Macro spatial target for navigation.
       Multiple UAVs may approach the same hotspot region when demand surges.
       Choice 0 (idle) is always safe and valid.
     - a_m^off: Masked candidate scheduling head respecting K_m^comm.
       Only IoT devices physically within actual 3D coverage radius,
       with rate R_im >= R_min and eligible UAV (e_m=1), can have active mask = 1.
  2. Multi-Device Radio Concurrency & Exclusivity Shield:
     - sum_i u_im(t) <= K_m^comm per UAV (Constraint C3a).
     - sum_m u_im(t) <= 1 per IoT device (Constraint C3b).
     - Deterministic tie-breaking by channel quality/rate.
  3. Continuous-Time Trajectory Collision Shield:
     - Continuous segment evaluation s in [0, 1]:
       min_{s in [0, 1]} ||(q_m - q_n) + s*(d_m - d_n)|| >= d_safe
     - Decelerates or halts hazardous movement proposals.
  4. Physical Boundary & Battery Reserve Projections.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import (
        MIN_BATTERY_RESERVE, MAX_SERVICE_RADIUS, SAFE_INTER_UAV_DISTANCE,
        AREA_RADIUS, UAV_ALT_MIN, UAV_ALT_MAX, K_COMM_DEFAULT, MAX_CANDIDATE_POOL,
        R_MIN_UPLINK
    )
except ImportError:
    MIN_BATTERY_RESERVE = 50.0
    MAX_SERVICE_RADIUS = 250.0
    SAFE_INTER_UAV_DISTANCE = 10.0
    AREA_RADIUS = 500.0
    UAV_ALT_MIN = 50.0
    UAV_ALT_MAX = 150.0
    K_COMM_DEFAULT = 3
    MAX_CANDIDATE_POOL = 8
    R_MIN_UPLINK = 1e5


class ActionMaskGenerator:
    """
    Generates state-dependent feasibility masks for:
      1. Macro target cluster choice c_m in {0, 1, ..., C}.
      2. Offload scheduling candidate selection over padded candidate list (size MAX_CANDIDATE_POOL).
    """

    def __init__(self, n_uavs, n_clusters=5, action_dim=5, min_battery_res=MIN_BATTERY_RESERVE,
                 max_service_radius=MAX_SERVICE_RADIUS, k_comm=K_COMM_DEFAULT,
                 max_candidate_pool=MAX_CANDIDATE_POOL, **kwargs):
        self.n_uavs = n_uavs
        self.n_clusters = n_clusters
        self.action_dim = action_dim
        self.min_battery_res = min_battery_res
        self.max_service_radius = max_service_radius
        self.k_comm = k_comm
        self.max_candidate_pool = max_candidate_pool

    def compute_cluster_mask(self, uav_positions, uav_batteries, cluster_centroids,
                             eligibility_states, tau_states, phi_states):
        """
        Compute binary cluster target mask of shape (n_uavs, n_clusters + 1).
        Index 0 is ALWAYS 1.0 (idle is always safe).
        Multiple UAVs are allowed to approach the same hotspot if demand warrants it.
        """
        mask = np.ones((self.n_uavs, self.n_clusters + 1), dtype=np.float32)

        for m in range(self.n_uavs):
            # Ineligible UAV: unauthenticated, hardware fault, or low battery
            if eligibility_states[m] == 0 or tau_states[m] == 0 or phi_states[m] == 0 or uav_batteries[m] <= self.min_battery_res:
                mask[m, 1:] = 0.0
                mask[m, 0] = 1.0
                continue

            # Ensure at least idle is valid
            mask[m, 0] = 1.0

        return mask

    def compute_candidate_mask(self, candidate_counts):
        """
        Compute binary candidate mask of shape (n_uavs, max_candidate_pool).
        mask[m, k] = 1.0 if candidate k is a real covered node, 0.0 if padded/empty.
        """
        mask = np.zeros((self.n_uavs, self.max_candidate_pool), dtype=np.float32)
        for m in range(self.n_uavs):
            c_count = min(int(candidate_counts[m]), self.max_candidate_pool)
            if c_count > 0:
                mask[m, :c_count] = 1.0
        return mask

    def compute_action_mask(self, uav_positions, uav_batteries, uav_capacities=None,
                            eligibility_states=None, tau_states=None, phi_states=None,
                            cluster_centroids=None, candidate_counts=None):
        """
        Backward-compatible unified action mask getter.
        Returns cluster target mask of shape (n_uavs, n_clusters + 1).
        """
        if eligibility_states is None:
            eligibility_states = np.ones(self.n_uavs, dtype=int)
        if tau_states is None:
            tau_states = np.ones(self.n_uavs, dtype=int)
        if phi_states is None:
            phi_states = np.ones(self.n_uavs, dtype=int)

        return self.compute_cluster_mask(
            uav_positions=uav_positions,
            uav_batteries=uav_batteries,
            cluster_centroids=cluster_centroids,
            eligibility_states=eligibility_states,
            tau_states=tau_states,
            phi_states=phi_states
        )


class JointSafetyShield:
    """
    Deterministic Safety & Concurrency Shield enforcing:
      1. Single connection per device: sum_m u_{i,m}(t) <= 1
      2. Per-UAV concurrency cap: sum_i u_{i,m}(t) <= K_m^comm
      3. Continuous-time trajectory collision avoidance: min_{s in [0,1]} ||Delta p(s)|| >= d_safe
      4. Physical workspace boundaries and battery depletion limits
    """

    def __init__(self, n_uavs, safe_dist=SAFE_INTER_UAV_DISTANCE,
                 area_radius=AREA_RADIUS, alt_min=UAV_ALT_MIN, alt_max=UAV_ALT_MAX,
                 k_comm=K_COMM_DEFAULT):
        self.n_uavs = n_uavs
        self.safe_dist = safe_dist
        self.area_radius = area_radius
        self.alt_min = alt_min
        self.alt_max = alt_max
        self.k_comm = k_comm
        self.total_shield_interventions = 0
        self.link_conflicts_resolved = 0

    def resolve_cluster_conflicts(self, proposed_clusters, uav_positions=None, cluster_centroids=None, eligibility=None, strict_exclusivity=False):
        """
        Enforce cluster target exclusivity deterministically when strict_exclusivity=True,
        or allow multi-UAV regional concentration when strict_exclusivity=False (default).
        """
        if eligibility is None:
            eligibility = np.ones(self.n_uavs, dtype=int)

        executed_clusters = np.zeros(self.n_uavs, dtype=int)

        if not strict_exclusivity:
            for m in range(self.n_uavs):
                if eligibility[m] == 1:
                    executed_clusters[m] = int(proposed_clusters[m])
                else:
                    executed_clusters[m] = 0
            return executed_clusters

        # Strict cluster exclusivity (sum_m x_{m,c} <= 1)
        claimed_clusters = set()
        scores = []
        for m in range(self.n_uavs):
            c = int(proposed_clusters[m])
            if c == 0 or eligibility[m] == 0 or cluster_centroids is None or c - 1 >= len(cluster_centroids) or uav_positions is None:
                dist = 1e9
            else:
                dist = np.linalg.norm(uav_positions[m][:2] - cluster_centroids[c - 1])
            scores.append((eligibility[m], -dist, m, c))

        scores.sort(reverse=True)

        for elig, neg_dist, m, c in scores:
            if elig == 0 or c == 0:
                executed_clusters[m] = 0
            elif c not in claimed_clusters:
                executed_clusters[m] = c
                claimed_clusters.add(c)
            else:
                self.link_conflicts_resolved += 1
                best_alt = 0
                best_alt_dist = 1e9
                if cluster_centroids is not None:
                    for alt_c in range(1, len(cluster_centroids) + 1):
                        if alt_c not in claimed_clusters:
                            alt_dist = np.linalg.norm(uav_positions[m][:2] - cluster_centroids[alt_c - 1]) if uav_positions is not None else 0.0
                            if alt_dist < best_alt_dist:
                                best_alt_dist = alt_dist
                                best_alt = alt_c
                if best_alt > 0:
                    executed_clusters[m] = best_alt
                    claimed_clusters.add(best_alt)
                else:
                    executed_clusters[m] = 0

        return executed_clusters

    def resolve_link_conflicts(self, proposed_links_by_uav, candidate_rates_by_uav, k_comm=None):
        """
        Enforces:
          1. sum_i u_im(t) <= K_m^comm
          2. sum_m u_im(t) <= 1
        proposed_links_by_uav: dict {m: [list of device_ids proposed]}
        candidate_rates_by_uav: dict {m: {device_id: rate}}
        Returns: scheduled_links {m: [list of device_ids]}
        """
        cap = k_comm if k_comm is not None else self.k_comm
        scheduled = {m: [] for m in range(self.n_uavs)}
        claimed_devices = set()

        # Collect all link proposals as (rate, m, dev_id) tuples sorted descending by rate
        all_proposals = []
        for m in range(self.n_uavs):
            for dev_id in proposed_links_by_uav.get(m, []):
                rate = candidate_rates_by_uav.get(m, {}).get(dev_id, 0.0)
                all_proposals.append((rate, m, dev_id))

        all_proposals.sort(key=lambda item: item[0], reverse=True)

        for rate, m, dev_id in all_proposals:
            if len(scheduled[m]) >= cap:
                continue
            if dev_id in claimed_devices:
                self.link_conflicts_resolved += 1
                continue
            scheduled[m].append(dev_id)
            claimed_devices.add(dev_id)

        return scheduled

    def check_piecewise_fly_hover_separation(self, p1, p2, d1, d2, v_cr=15.0, delta_t=1.0):
        """
        Check minimum Euclidean separation over actual piecewise fly-then-hover trajectories:
        Each UAV flies at cruise speed v_cr during [0, t_fly], then hovers at destination during [t_fly, delta_t].
        """
        norm1 = np.linalg.norm(d1)
        norm2 = np.linalg.norm(d2)
        t_f1 = min(float(delta_t), float(norm1 / max(v_cr, 1e-3)))
        t_f2 = min(float(delta_t), float(norm2 / max(v_cr, 1e-3)))

        # Velocity vectors during flight phase
        v1 = d1 / max(t_f1, 1e-6) if t_f1 > 1e-6 else np.zeros(3)
        v2 = d2 / max(t_f2, 1e-6) if t_f2 > 1e-6 else np.zeros(3)

        # Critical transition times partitioning [0, delta_t]
        tau_cuts = sorted(list(set([0.0, min(t_f1, t_f2), max(t_f1, t_f2), float(delta_t)])))

        min_dist = float('inf')
        t_crit = 0.0

        for i in range(len(tau_cuts) - 1):
            t_a = tau_cuts[i]
            t_b = tau_cuts[i + 1]
            if t_b - t_a < 1e-7:
                continue

            # Position functions on [t_a, t_b]
            # q1(t) = A1 + B1 * t
            if t_a >= t_f1:
                A1 = p1 + d1
                B1 = np.zeros(3)
            else:
                A1 = p1
                B1 = v1

            if t_a >= t_f2:
                A2 = p2 + d2
                B2 = np.zeros(3)
            else:
                A2 = p2
                B2 = v2

            delta_A = A1 - A2
            delta_B = B1 - B2

            denom = np.dot(delta_B, delta_B)
            if denom < 1e-9:
                cand_t = [t_a, t_b]
            else:
                t_star = -np.dot(delta_A, delta_B) / denom
                cand_t = [t_a, t_b, float(np.clip(t_star, t_a, t_b))]

            for t_eval in cand_t:
                sep = np.linalg.norm(delta_A + delta_B * t_eval)
                if sep < min_dist:
                    min_dist = sep
                    t_crit = t_eval

        return float(min_dist), float(t_crit)

    def check_continuous_separation(self, p1, p2, d1, d2, v_cr=15.0, delta_t=1.0):
        """Dispatches to piecewise fly-then-hover separation check."""
        return self.check_piecewise_fly_hover_separation(p1, p2, d1, d2, v_cr=v_cr, delta_t=delta_t)

    def shield_trajectories(self, uav_positions, proposed_displacements, v_cr=15.0, delta_t=1.0):
        """
        Enforce continuous pairwise separation and physical boundary constraints.
        Returns: safe_displacements, intervention_flag
        """
        safe_disp = np.copy(proposed_displacements)
        intervened = False

        # 1. Pairwise Piecewise Fly-Hover Continuous Collision Avoidance
        for m in range(self.n_uavs):
            for n in range(m + 1, self.n_uavs):
                d_min, s_crit = self.check_piecewise_fly_hover_separation(
                    uav_positions[m], uav_positions[n],
                    safe_disp[m], safe_disp[n],
                    v_cr=v_cr, delta_t=delta_t
                )

                if d_min < self.safe_dist:
                    intervened = True
                    self.total_shield_interventions += 1
                    scale = max(0.0, (d_min / max(self.safe_dist, 1.0)) - 0.1)
                    safe_disp[m] *= scale
                    safe_disp[n] *= scale

        # 2. Boundary Constraints
        for m in range(self.n_uavs):
            target_pos = uav_positions[m] + safe_disp[m]
            target_pos[0] = np.clip(target_pos[0], -self.area_radius, self.area_radius)
            target_pos[1] = np.clip(target_pos[1], -self.area_radius, self.area_radius)
            target_pos[2] = np.clip(target_pos[2], self.alt_min, self.alt_max)
            safe_disp[m] = target_pos - uav_positions[m]

        return safe_disp, intervened
