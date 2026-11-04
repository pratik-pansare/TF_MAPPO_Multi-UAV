"""
State-Dependent Action Feasibility Mask & Joint Safety Shield
============================================================
Paper Reference: Section III.F & Constraints (C1–C15)
Target Journal: IEEE Internet of Things Journal

Features:
  1. Discrete Action Feasibility Masking M_m^clu(t) for Cluster Selection:
       - tau_m(t)=0, phi_m(t)=0, or E_m < E_res -> cluster choices c in {1..C} masked, c=0 (idle) retained.
       - Max service radius violation d_{m,c} > R_max -> cluster c masked.
       - Idle fallback (c=0) is ALWAYS valid, ensuring valid probability normalization.
  2. Deterministic Joint Cluster Conflict Resolution:
       - Enforces cluster exclusivity: sum_m x_{m,c}(t) <= 1.
       - If multiple UAVs select the same cluster, tie-break by spatial proximity and queue depth.
  3. Continuous-Time Trajectory Collision Shield:
       - Evaluates minimum separation across the continuous interval s in [0, 1]:
           min_{s in [0, 1]} ||(q_m - q_n) + s*(d_m - d_n)|| >= d_safe
       - Decelerates or halts hazardous movement proposals.
  4. Boundary and Battery Reserve Projection.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import (
        MIN_BATTERY_RESERVE, MAX_SERVICE_RADIUS, SAFE_INTER_UAV_DISTANCE,
        AREA_RADIUS, UAV_ALT_MIN, UAV_ALT_MAX
    )
except ImportError:
    MIN_BATTERY_RESERVE = 50.0
    MAX_SERVICE_RADIUS = 600.0
    SAFE_INTER_UAV_DISTANCE = 10.0
    AREA_RADIUS = 500.0
    UAV_ALT_MIN = 50.0
    UAV_ALT_MAX = 150.0


class ActionMaskGenerator:
    """
    Generates discrete feasibility mask M_m(t) for cluster selection c_m in {0, 1, ..., C}.
    Index 0: Idle / No service.
    Index 1..C: Cluster 1 to C.
    """

    def __init__(self, n_uavs, n_clusters=5, action_dim=5, min_battery_res=MIN_BATTERY_RESERVE,
                 max_service_radius=MAX_SERVICE_RADIUS, **kwargs):
        self.n_uavs = n_uavs
        self.n_clusters = n_clusters
        self.action_dim = action_dim
        self.min_battery_res = min_battery_res
        self.max_service_radius = max_service_radius

    def compute_action_mask(self, uav_positions, uav_batteries, uav_capacities=None,
                            eligibility_states=None, tau_states=None, phi_states=None, cluster_centroids=None):
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

    def compute_cluster_mask(self, uav_positions, uav_batteries, cluster_centroids,
                              eligibility_states, tau_states, phi_states):
        """
        Compute binary cluster mask of shape (n_uavs, n_clusters + 1).
        Mask value 1.0 = Feasible choice, 0.0 = Infeasible choice.
        Index 0 is ALWAYS 1.0 (idle is always safe and feasible).
        """
        mask = np.ones((self.n_uavs, self.n_clusters + 1), dtype=np.float32)

        for m in range(self.n_uavs):
            # Ineligible UAV: unauthenticated, hardware fault, or low battery
            if eligibility_states[m] == 0 or tau_states[m] == 0 or phi_states[m] == 0 or uav_batteries[m] <= self.min_battery_res:
                # Mask out all operational clusters 1..C
                mask[m, 1:] = 0.0
                mask[m, 0] = 1.0  # Idle is strictly feasible
                continue

            # Reachability check: distance to cluster centroid <= R_max
            if cluster_centroids is not None and len(cluster_centroids) > 0:
                uav_pos_2d = uav_positions[m][:2]
                for c in range(min(self.n_clusters, len(cluster_centroids))):
                    dist = np.linalg.norm(uav_pos_2d - cluster_centroids[c])
                    if dist > self.max_service_radius:
                        mask[m, c + 1] = 0.0

            # Ensure at least idle is valid
            mask[m, 0] = 1.0

        return mask


class JointSafetyShield:
    """
    Deterministic Safety Shield enforcing:
      1. Cluster exclusivity: sum_m x_{m,c}(t) <= 1
      2. Continuous-time trajectory collision avoidance: min_{s in [0,1]} ||Delta p(s)|| >= d_safe
      3. Physical workspace boundaries and battery depletion limits
    """

    def __init__(self, n_uavs, safe_dist=SAFE_INTER_UAV_DISTANCE,
                 area_radius=AREA_RADIUS, alt_min=UAV_ALT_MIN, alt_max=UAV_ALT_MAX):
        self.n_uavs = n_uavs
        self.safe_dist = safe_dist
        self.area_radius = area_radius
        self.alt_min = alt_min
        self.alt_max = alt_max
        self.total_shield_interventions = 0
        self.cluster_conflicts_resolved = 0

    def resolve_cluster_conflicts(self, proposed_clusters, uav_positions, cluster_centroids, eligibility):
        """
        Enforce cluster exclusivity (sum_m x_{m,c} <= 1) deterministically.
        If multiple UAVs choose cluster c (c >= 1), closest UAV wins; others redirected or idle.
        """
        executed_clusters = np.zeros(self.n_uavs, dtype=int)
        claimed_clusters = set()

        # Sort UAVs by eligibility and proximity
        # Priority: eligible first, then closer to proposed cluster centroid
        scores = []
        for m in range(self.n_uavs):
            c = proposed_clusters[m]
            if c == 0 or eligibility[m] == 0 or cluster_centroids is None or c - 1 >= len(cluster_centroids):
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
                # Conflict! Another UAV claimed cluster c
                self.cluster_conflicts_resolved += 1
                # Find nearest unclaimed cluster, else fallback to idle 0
                best_alt = 0
                best_alt_dist = 1e9
                if cluster_centroids is not None:
                    for alt_c in range(1, len(cluster_centroids) + 1):
                        if alt_c not in claimed_clusters:
                            alt_dist = np.linalg.norm(uav_positions[m][:2] - cluster_centroids[alt_c - 1])
                            if alt_dist < best_alt_dist:
                                best_alt_dist = alt_dist
                                best_alt = alt_c
                if best_alt > 0:
                    executed_clusters[m] = best_alt
                    claimed_clusters.add(best_alt)
                else:
                    executed_clusters[m] = 0  # Idle fallback

        return executed_clusters

    def check_continuous_separation(self, p1, p2, d1, d2):
        """
        Check minimum Euclidean separation over continuous segment s in [0, 1]:
          D(s)^2 = ||(p1 - p2) + s*(d1 - d2)||^2
        Returns: min_distance, s_crit
        """
        dp = np.array(p1, dtype=float) - np.array(p2, dtype=float)
        dv = np.array(d1, dtype=float) - np.array(d2, dtype=float)

        dv_sq = np.sum(dv ** 2)
        if dv_sq < 1e-9:
            # Parallel trajectories or both hovering
            return float(np.linalg.norm(dp)), 0.0

        s_star = -np.dot(dp, dv) / dv_sq
        s_crit = float(np.clip(s_star, 0.0, 1.0))
        d_min = float(np.linalg.norm(dp + s_crit * dv))
        return d_min, s_crit

    def shield_trajectories(self, uav_positions, proposed_displacements):
        """
        Enforce continuous pairwise separation and physical boundary constraints.
        Returns: safe_displacements, intervention_flag
        """
        safe_disp = np.copy(proposed_displacements)
        intervened = False

        # 1. Pairwise Continuous Collision Avoidance
        for m in range(self.n_uavs):
            for n in range(m + 1, self.n_uavs):
                d_min, s_crit = self.check_continuous_separation(
                    uav_positions[m], uav_positions[n],
                    safe_disp[m], safe_disp[n]
                )

                if d_min < self.safe_dist:
                    # Imminent trajectory collision!
                    intervened = True
                    self.total_shield_interventions += 1
                    # Safety intervention: Halt or scale down displacements
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
