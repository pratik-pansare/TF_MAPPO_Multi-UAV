"""
Pre-deployment Workload-Aware IoT Clustering
=============================================
Reference: Section III.B (Eq. 8–12) of Paper

Partitions $N$ geographically distributed ground IoT devices into $C$ compact
service clusters before UAV deployment.

Workload Weight Equation (Eq. 10):
    w_i = 1 + lambda_w * (W_i / W_max)
where W_i = D_i * C_i is nominal computational workload of device i.

Weighted Cluster Centroid Equation (Eq. 11):
    z_c = sum_{i in I_c} (w_i * p_i) / sum_{i in I_c} w_i

Initial UAV Deployment Vector (Eq. 12):
    q_m(0) = (z_c, H_m)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import LAMBDA_W
except ImportError:
    LAMBDA_W = 0.5



class WorkloadAwareClusterManager:
    """
    Workload-aware K-Means IoT Clustering Manager.
    Organizes ground IoT devices into compact service regions based on
    spatial proximity and computational workload requirements.
    """

    def __init__(self, n_clusters=5, lambda_w=0.5, random_state=42):
        self.n_clusters = n_clusters
        self.lambda_w = lambda_w
        self.rng = np.random.RandomState(random_state)
        self.centroids = None          # Array of shape (n_clusters, 2)
        self.cluster_assignments = None # Array of shape (n_devices,)
        self.device_weights = None      # Array of shape (n_devices,)

    def compute_device_weights(self, iot_devices):
        """
        Compute workload importance weights w_i for each IoT device (Eq. 10).
        w_i = 1 + lambda_w * (W_i / W_max)
        """
        workloads = []
        for dev in iot_devices:
            if hasattr(dev, 'current_task') and dev.current_task is not None:
                w = dev.current_task.get('data_size', 2.75e6) * dev.current_task.get('cpu_cycles', 0.5e9)
            else:
                w = 2.75e6 * 0.5e9  # nominal workload
            workloads.append(w)
        
        workloads = np.array(workloads, dtype=float)
        w_max = np.max(workloads) if len(workloads) > 0 and np.max(workloads) > 0 else 1.0
        weights = 1.0 + self.lambda_w * (workloads / w_max)
        self.device_weights = weights
        return weights

    def fit_clusters(self, iot_devices, max_iter=100):
        """
        Perform weighted K-Means clustering on IoT device positions (Eq. 8–11).
        
        Parameters
        ----------
        iot_devices : list of IoTDevice objects
        max_iter    : int, maximum iterations for K-Means

        Returns
        -------
        centroids           : np.ndarray of shape (n_clusters, 2)
        cluster_assignments : np.ndarray of shape (n_devices,)
        """
        n_devices = len(iot_devices)
        if n_devices == 0:
            return np.zeros((self.n_clusters, 2)), np.zeros(0, dtype=int)

        n_clusters = min(self.n_clusters, n_devices)
        positions = np.array([dev.position[:2] for dev in iot_devices], dtype=float) # (N, 2)
        weights = self.compute_device_weights(iot_devices)

        # K-Means++ initialisation
        init_indices = [self.rng.choice(n_devices)]
        for _ in range(1, n_clusters):
            dist_sq = np.min([np.sum((positions - positions[i])**2, axis=1) for i in init_indices], axis=0)
            probs = dist_sq / np.sum(dist_sq) if np.sum(dist_sq) > 0 else np.ones(n_devices)/n_devices
            next_idx = self.rng.choice(n_devices, p=probs)
            init_indices.append(next_idx)

        centroids = positions[init_indices].copy()
        assignments = np.zeros(n_devices, dtype=int)

        for _ in range(max_iter):
            # Assignment step: find nearest centroid for each device
            distances = np.linalg.norm(positions[:, None, :] - centroids[None, :, :], axis=2) # (N, n_clusters)
            new_assignments = np.argmin(distances, axis=1)

            if np.array_equal(new_assignments, assignments):
                break
            assignments = new_assignments

            # Update step: weighted centroid calculation (Eq. 11)
            for c in range(n_clusters):
                mask = (assignments == c)
                if np.any(mask):
                    weighted_pos = positions[mask] * weights[mask, None]
                    centroids[c] = np.sum(weighted_pos, axis=0) / np.sum(weights[mask])
                else:
                    # Re-initialise empty cluster to a random device
                    centroids[c] = positions[self.rng.choice(n_devices)]

        # Assign cluster IDs back to IoT devices
        for idx, dev in enumerate(iot_devices):
            dev.cluster_id = int(assignments[idx])

        self.centroids = centroids
        self.cluster_assignments = assignments
        return centroids, assignments

    def get_initial_uav_positions(self, iot_devices, uav_altitudes):
        """
        Determine initial UAV positions q_m(0) = (z_c, H_m) (Eq. 12).
        
        Parameters
        ----------
        iot_devices   : list of IoTDevice objects
        uav_altitudes : list or array of UAV deployment altitudes H_m

        Returns
        -------
        initial_positions : np.ndarray of shape (M, 3)
        """
        M = len(uav_altitudes)
        self.fit_clusters(iot_devices)

        initial_positions = np.zeros((M, 3), dtype=float)
        for m in range(M):
            cluster_idx = m % len(self.centroids)
            centroid_2d = self.centroids[cluster_idx]
            initial_positions[m] = [centroid_2d[0], centroid_2d[1], uav_altitudes[m]]

        return initial_positions

    def get_cluster_stats(self, iot_devices):
        """Returns metadata for each cluster (device count, total workload)."""
        if self.centroids is None:
            self.fit_clusters(iot_devices)

        stats = []
        for c in range(len(self.centroids)):
            device_indices = np.where(self.cluster_assignments == c)[0]
            workload_sum = np.sum(self.device_weights[device_indices]) if len(device_indices) > 0 else 0.0
            stats.append({
                'cluster_id': c,
                'centroid': self.centroids[c],
                'n_devices': len(device_indices),
                'total_weight': workload_sum,
            })
        return stats
