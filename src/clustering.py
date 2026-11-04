"""
Pre-deployment Workload-Aware IoT Clustering & BS Commissioning
==============================================================
Paper Reference: Section III.A & III.B (Eq. 1–3)
Target Journal: IEEE Internet of Things Journal

Partitions N geographically distributed static ground IoT devices into C compact
service clusters before UAV deployment.

1. Device Registration & Commissioning (at Ground Base Station):
   Each device sends an authenticated metadata report:
     Report_i = {id, (x, y, 0), lambda_i^0, D_i^0, C_i^0, t_reg}
   Nominal demand:
     W_i^0 = lambda_i^0 * (D_i^0 * C_i^0)  [CPU cycles / second]

2. Workload Weight Equation (Eq. 2):
     w_i = 1 + lambda_w * (W_i^0 / W_max)
   where W_max = max_j W_j^0 (with zero-demand fallback).

3. Weighted Lloyd Clustering (Eq. 1 & 3):
   Objective:
     J_clu = sum_{c=1}^C sum_{i in I_c} w_i * ||p_i - z_c||^2
   Assignment:
     Nearest spatial centroid: argmin_c ||p_i - z_c||
   Centroid update:
     z_c = (sum_{i in I_c} w_i * p_i) / (sum_{i in I_c} w_i)

4. Initial UAV Deployment Vector:
     q_m(0) = (z_{c_m}, H_m)
"""

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import LAMBDA_W, NUM_CLUSTERS
except ImportError:
    LAMBDA_W = 0.5
    NUM_CLUSTERS = 5


class IoTDeviceRegistrationReport:
    """Represents pre-deployment commissioning report sent by IoT node to BS."""

    REPORT_SIZE_BYTES = 32  # 32 bytes: DeviceID (4B) + 3D pos (12B) + rate (4B) + bits (4B) + cycles (4B) + timestamp (4B)

    def __init__(self, device_id, position, nominal_task_rate=0.7,
                 nominal_data_size=1.5e6, nominal_cycles_per_bit=500.0, timestamp=0.0):
        self.device_id = device_id
        self.position = np.array(position, dtype=float)
        self.nominal_task_rate = float(nominal_task_rate)          # lambda_i^0 (tasks/sec)
        self.nominal_data_size = float(nominal_data_size)          # bits/task
        self.nominal_cycles_per_bit = float(nominal_cycles_per_bit)# cycles/bit
        self.timestamp = float(timestamp)

    @property
    def nominal_demand(self):
        """W_i^0 = lambda_i^0 * (D_i^0 * C_i^0) in CPU cycles/sec."""
        return self.nominal_task_rate * (self.nominal_data_size * self.nominal_cycles_per_bit)


class BSCommissioningManager:
    """
    Manages one-time pre-deployment registration, metadata collection,
    signaling overhead accounting, and offline clustering execution.
    """

    def __init__(self, tx_power_w=0.1, bs_bandwidth_hz=1e6):
        self.tx_power = tx_power_w
        self.bs_bandwidth = bs_bandwidth_hz
        self.reports = []
        self.total_signaling_bytes = 0
        self.total_signaling_energy = 0.0
        self.total_signaling_delay = 0.0

    def register_device(self, report):
        """Collects authenticated registration report from an IoT device."""
        self.reports.append(report)
        self.total_signaling_bytes += IoTDeviceRegistrationReport.REPORT_SIZE_BYTES
        # Approximate terrestrial uplink delay and transmit energy to BS/gateway
        approx_rate = self.bs_bandwidth * 2.0  # ~2 Mbps nominal
        t_tx = (IoTDeviceRegistrationReport.REPORT_SIZE_BYTES * 8.0) / approx_rate
        e_tx = self.tx_power * t_tx
        self.total_signaling_delay += t_tx
        self.total_signaling_energy += e_tx

    def get_commissioning_stats(self):
        return {
            'total_registered': len(self.reports),
            'total_bytes': self.total_signaling_bytes,
            'total_energy_joules': self.total_signaling_energy,
            'total_signaling_delay_s': self.total_signaling_delay,
        }


class WorkloadAwareClusterManager:
    """
    Spatial Workload-Weighted K-Means (Weighted Lloyd) Clustering Manager.
    """

    def __init__(self, n_clusters=NUM_CLUSTERS, lambda_w=LAMBDA_W, random_state=42):
        self.n_clusters = n_clusters
        self.lambda_w = lambda_w
        self.rng = np.random.RandomState(random_state)
        self.centroids = None          # Array of shape (n_clusters, 2)
        self.cluster_assignments = None # Array of shape (n_devices,)
        self.device_weights = None      # Array of shape (n_devices,)
        self.nominal_demands = None     # Array of shape (n_devices,)
        self.last_fit_time = 0.0

    def compute_device_weights(self, iot_devices):
        """
        Compute importance weights w_i = 1 + lambda_w * (W_i^0 / W_max).
        Robust against all-zero demands.
        """
        demands = []
        for dev in iot_devices:
            if hasattr(dev, 'nominal_demand'):
                w = dev.nominal_demand
            elif hasattr(dev, 'nominal_workload'):
                w = dev.nominal_workload
            elif hasattr(dev, 'current_task') and dev.current_task is not None:
                task = dev.current_task
                w = getattr(task, 'workload', 1.5e6 * 500.0)
            else:
                w = 1.5e6 * 500.0 * 0.7  # default nominal cycles/sec
            demands.append(float(w))

        demands = np.array(demands, dtype=float)
        self.nominal_demands = demands
        w_max = np.max(demands) if len(demands) > 0 and np.max(demands) > 0 else 1.0

        if w_max > 0:
            weights = 1.0 + self.lambda_w * (demands / w_max)
        else:
            weights = np.ones_like(demands)

        self.device_weights = weights
        return weights

    def fit_clusters(self, iot_devices, max_iter=100, tol=1e-4):
        """
        Perform spatial workload-weighted Lloyd iterations:
          1. Nearest centroid assignment
          2. Weighted centroid center-of-mass update
          3. Empty cluster repair
          4. Deterministic tie-breaking
        """
        t_start = time.perf_counter()
        n_devices = len(iot_devices)
        if n_devices == 0:
            self.centroids = np.zeros((self.n_clusters, 2))
            self.cluster_assignments = np.zeros(0, dtype=int)
            return self.centroids, self.cluster_assignments

        n_clusters = min(self.n_clusters, n_devices)
        positions = np.array([dev.position[:2] for dev in iot_devices], dtype=float)  # (N, 2)
        weights = self.compute_device_weights(iot_devices)                             # (N,)

        # Weighted K-Means++ Initialization
        init_indices = [int(self.rng.choice(n_devices))]
        for _ in range(1, n_clusters):
            # Compute squared distances to nearest existing centroid
            dist_sq = np.min([np.sum((positions - positions[i]) ** 2, axis=1) for i in init_indices], axis=0)
            # Probability proportional to weight * distance^2
            weighted_dist_sq = dist_sq * weights
            total_prob = np.sum(weighted_dist_sq)
            if total_prob > 0:
                probs = weighted_dist_sq / total_prob
            else:
                probs = np.ones(n_devices) / n_devices
            next_idx = int(self.rng.choice(n_devices, p=probs))
            init_indices.append(next_idx)

        centroids = positions[init_indices].copy()
        assignments = np.zeros(n_devices, dtype=int)

        for iteration in range(max_iter):
            # Step 1: Spatial Assignment to nearest centroid
            # Distances: (N, C)
            distances = np.linalg.norm(positions[:, None, :] - centroids[None, :, :], axis=2)
            new_assignments = np.argmin(distances, axis=1)

            # Step 2: Empty Cluster Repair
            assigned_counts = np.bincount(new_assignments, minlength=n_clusters)
            for c in range(n_clusters):
                if assigned_counts[c] == 0:
                    # Find device farthest from its currently assigned centroid
                    current_dists = distances[np.arange(n_devices), new_assignments]
                    farthest_idx = int(np.argmax(current_dists))
                    new_assignments[farthest_idx] = c
                    centroids[c] = positions[farthest_idx].copy()
                    assigned_counts = np.bincount(new_assignments, minlength=n_clusters)

            # Check convergence
            if np.array_equal(new_assignments, assignments):
                break
            assignments = new_assignments

            # Step 3: Weighted Centroid Update
            new_centroids = np.zeros_like(centroids)
            for c in range(n_clusters):
                mask = (assignments == c)
                if np.any(mask):
                    w_sub = weights[mask]
                    p_sub = positions[mask]
                    new_centroids[c] = np.sum(p_sub * w_sub[:, None], axis=0) / np.sum(w_sub)
                else:
                    new_centroids[c] = centroids[c]

            shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids
            if shift < tol:
                break

        # Assign cluster IDs back to IoT devices
        for idx, dev in enumerate(iot_devices):
            dev.cluster_id = int(assignments[idx])

        self.centroids = centroids
        self.cluster_assignments = assignments
        self.last_fit_time = time.perf_counter() - t_start
        return centroids, assignments

    def compute_clustering_objective(self, iot_devices):
        """Compute J_clu = sum_c sum_{i in I_c} w_i ||p_i - z_c||^2 (Eq. 1)."""
        if self.centroids is None or self.cluster_assignments is None:
            self.fit_clusters(iot_devices)

        positions = np.array([dev.position[:2] for dev in iot_devices], dtype=float)
        weights = self.device_weights if self.device_weights is not None else self.compute_device_weights(iot_devices)

        j_clu = 0.0
        for i, dev in enumerate(iot_devices):
            c = self.cluster_assignments[i]
            z_c = self.centroids[c]
            dist_sq = np.sum((positions[i] - z_c) ** 2)
            j_clu += weights[i] * dist_sq
        return float(j_clu)

    def get_cluster_stats(self, iot_devices):
        """
        Computes detailed cluster metrics: device counts, workload totals,
        workload Coefficient of Variation (CV = std/mean), max cluster size,
        and maximum cluster coverage radius.
        """
        if self.centroids is None:
            self.fit_clusters(iot_devices)

        positions = np.array([dev.position[:2] for dev in iot_devices], dtype=float)
        weights = self.device_weights if self.device_weights is not None else self.compute_device_weights(iot_devices)

        cluster_workloads = []
        cluster_radii = []
        cluster_device_counts = []
        stats = []

        for c in range(len(self.centroids)):
            dev_indices = np.where(self.cluster_assignments == c)[0]
            count = len(dev_indices)
            cluster_device_counts.append(count)

            if count > 0:
                c_workload = float(np.sum(weights[dev_indices]))
                dists_to_centroid = np.linalg.norm(positions[dev_indices] - self.centroids[c], axis=1)
                radius = float(np.max(dists_to_centroid))
            else:
                c_workload = 0.0
                radius = 0.0

            cluster_workloads.append(c_workload)
            cluster_radii.append(radius)

            stats.append({
                'cluster_id': c,
                'centroid': self.centroids[c].tolist(),
                'n_devices': count,
                'total_weight': c_workload,
                'coverage_radius_m': radius,
            })

        mean_workload = float(np.mean(cluster_workloads)) if len(cluster_workloads) > 0 else 0.0
        std_workload = float(np.std(cluster_workloads)) if len(cluster_workloads) > 0 else 0.0
        workload_cv = (std_workload / mean_workload) if mean_workload > 0 else 0.0

        summary = {
            'clusters': stats,
            'workload_cv': float(workload_cv),
            'max_cluster_size': int(np.max(cluster_device_counts)) if len(cluster_device_counts) > 0 else 0,
            'max_coverage_radius_m': float(np.max(cluster_radii)) if len(cluster_radii) > 0 else 0.0,
            'clustering_objective_J': self.compute_clustering_objective(iot_devices),
            'offline_runtime_sec': float(self.last_fit_time),
        }
        return summary

    def get_initial_uav_positions(self, iot_devices, uav_altitudes):
        """Initial UAV positions q_m(0) = (z_{c_m}, H_m) (Eq. 3)."""
        M = len(uav_altitudes)
        if self.centroids is None:
            self.fit_clusters(iot_devices)

        initial_positions = np.zeros((M, 3), dtype=float)
        for m in range(M):
            cluster_idx = m % len(self.centroids)
            centroid_2d = self.centroids[cluster_idx]
            initial_positions[m] = [centroid_2d[0], centroid_2d[1], uav_altitudes[m]]

        return initial_positions
