"""
Spatial IoT Clustering & Pre-Deployment BS Commissioning
========================================================
Paper Reference: Section III.A (Eq. 1)
Target Journal: IEEE Internet of Things Journal

Partitions N geographically distributed static ground IoT devices into C compact
spatial regions before UAV deployment without assuming future workload oracle knowledge.

1. Device Registration & Commissioning (at Ground Base Station):
   Each device sends an authenticated registration record:
     Record_i = {id, (x_i, y_i, 0), t_reg}
   No future arrival rates or workload predictions are reported.

2. Pure Spatial K-Means Clustering (Eq. 1):
   Objective:
     J_clu = sum_{c=1}^C sum_{i in I_c} ||p_i - z_c||^2
   Assignment:
     Nearest spatial centroid: argmin_c ||p_i - z_c||
   Centroid update:
     z_c = (1 / |I_c|) * sum_{i in I_c} p_i  (unweighted arithmetic mean)

3. Initial UAV Deployment Vector:
     q_m(0) = (z_{c_m}, H_m)

4. Legacy Ablation:
   Workload-weighted Lloyd clustering is retained strictly as an isolated
   ablation class (LegacyWorkloadClusterManager) for historical comparison.
"""

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import NUM_CLUSTERS, MAX_SERVICE_RADIUS, LAMBDA_W
except ImportError:
    NUM_CLUSTERS = 5
    MAX_SERVICE_RADIUS = 250.0
    LAMBDA_W = 0.5


class IoTDeviceRegistrationReport:
    """
    Represents pre-deployment commissioning report sent by IoT node to BS.
    Contains strictly geometric and identity metadata; no future workload oracle.
    """

    REPORT_SIZE_BYTES = 32  # 32 bytes: DeviceID (4B) + 3D position (12B) + status (4B) + timestamp (8B) + signature (4B)

    def __init__(self, device_id, position, timestamp=0.0, **kwargs):
        self.device_id = device_id
        self.position = np.array(position, dtype=float)
        self.timestamp = float(timestamp)


class BSCommissioningManager:
    """
    Manages one-time pre-deployment registration, metadata collection,
    signaling overhead accounting, and offline spatial clustering execution.
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
        approx_rate = self.bs_bandwidth * 2.0  # ~2 Mbps nominal terrestrial gateway link
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


class SpatialClusterManager:
    """
    Standard Geometric Spatial K-Means Cluster Manager.
    Partitions IoT nodes based purely on physical coordinates p_i = (x_i, y_i).
    Clusters remain fixed during a mission and serve as macro-level spatial
    indices and initial UAV dispatch locations. Real-time communication eligibility
    is determined dynamically by instantaneous UAV coverage, not cluster membership.
    """

    def __init__(self, n_clusters=NUM_CLUSTERS, max_coverage_radius=MAX_SERVICE_RADIUS, random_state=42, **kwargs):
        self.n_clusters = n_clusters
        self.max_coverage_radius = max_coverage_radius
        self.rng = np.random.RandomState(random_state)
        self.centroids = None           # Array of shape (n_clusters, 2)
        self.cluster_assignments = None  # Array of shape (n_devices,)
        self.iterations_run = 0
        self.last_fit_time = 0.0
        self.uncovered_nodes = []

    def fit_clusters(self, iot_devices, max_iter=100, tol=1e-4):
        """
        Perform standard spatial K-means with K-means++ initialization:
          1. Nearest spatial centroid assignment: argmin_c ||p_i - z_c||
          2. Arithmetic mean centroid update: z_c = (1 / |I_c|) * sum p_i
          3. Empty cluster repair
        """
        t_start = time.perf_counter()
        n_devices = len(iot_devices)
        if n_devices == 0:
            self.centroids = np.zeros((self.n_clusters, 2))
            self.cluster_assignments = np.zeros(0, dtype=int)
            return self.centroids, self.cluster_assignments

        n_clusters = min(self.n_clusters, n_devices)
        positions = np.array([dev.position[:2] for dev in iot_devices], dtype=float)  # (N, 2)

        # 1. Seeded K-Means++ Initialization
        init_indices = [int(self.rng.choice(n_devices))]
        for _ in range(1, n_clusters):
            dist_sq = np.min([np.sum((positions - positions[i]) ** 2, axis=1) for i in init_indices], axis=0)
            total_prob = np.sum(dist_sq)
            if total_prob > 0:
                probs = dist_sq / total_prob
            else:
                probs = np.ones(n_devices) / n_devices
            next_idx = int(self.rng.choice(n_devices, p=probs))
            init_indices.append(next_idx)

        centroids = positions[init_indices].copy()
        assignments = np.zeros(n_devices, dtype=int)

        self.iterations_run = 0
        for iteration in range(max_iter):
            self.iterations_run += 1
            # Step 1: Assign each device to the nearest spatial centroid
            distances = np.linalg.norm(positions[:, None, :] - centroids[None, :, :], axis=2)
            new_assignments = np.argmin(distances, axis=1)

            # Step 2: Empty Cluster Repair
            assigned_counts = np.bincount(new_assignments, minlength=n_clusters)
            for c in range(n_clusters):
                if assigned_counts[c] == 0:
                    current_dists = distances[np.arange(n_devices), new_assignments]
                    farthest_idx = int(np.argmax(current_dists))
                    new_assignments[farthest_idx] = c
                    centroids[c] = positions[farthest_idx].copy()
                    assigned_counts = np.bincount(new_assignments, minlength=n_clusters)

            # Check convergence
            if np.array_equal(new_assignments, assignments):
                break
            assignments = new_assignments

            # Step 3: Unweighted Centroid Update (Arithmetic Center of Mass)
            new_centroids = np.zeros_like(centroids)
            for c in range(n_clusters):
                mask = (assignments == c)
                if np.any(mask):
                    new_centroids[c] = np.mean(positions[mask], axis=0)
                else:
                    new_centroids[c] = centroids[c]

            shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids
            if shift < tol:
                break

        # Validate maximum coverage radius & record any uncovered nodes
        self.uncovered_nodes.clear()
        for i, dev in enumerate(iot_devices):
            dev.cluster_id = int(assignments[i])
            c = assignments[i]
            dist_to_c = np.linalg.norm(positions[i] - centroids[c])
            if dist_to_c > self.max_coverage_radius:
                self.uncovered_nodes.append((dev.id, float(dist_to_c)))

        self.centroids = centroids
        self.cluster_assignments = assignments
        self.last_fit_time = time.perf_counter() - t_start
        return centroids, assignments

    def compute_device_weights(self, iot_devices):
        """Unweighted geometric clustering: all devices have equal unit weight."""
        return np.ones(len(iot_devices), dtype=float)

    def compute_clustering_objective(self, iot_devices):
        """Compute standard spatial J_clu = sum_c sum_{i in I_c} ||p_i - z_c||^2 (Eq. 1)."""
        if self.centroids is None or self.cluster_assignments is None:
            self.fit_clusters(iot_devices)

        positions = np.array([dev.position[:2] for dev in iot_devices], dtype=float)
        j_clu = 0.0
        for i, dev in enumerate(iot_devices):
            c = self.cluster_assignments[i]
            z_c = self.centroids[c]
            j_clu += np.sum((positions[i] - z_c) ** 2)
        return float(j_clu)

    def get_cluster_stats(self, iot_devices):
        """
        Computes spatial cluster metrics: device counts, maximum radius,
        clustering objective J_clu, iterations, and uncovered node diagnostics.
        """
        if self.centroids is None:
            self.fit_clusters(iot_devices)

        positions = np.array([dev.position[:2] for dev in iot_devices], dtype=float)
        cluster_radii = []
        cluster_device_counts = []
        stats = []

        for c in range(len(self.centroids)):
            dev_indices = np.where(self.cluster_assignments == c)[0]
            count = len(dev_indices)
            cluster_device_counts.append(count)

            if count > 0:
                dists_to_centroid = np.linalg.norm(positions[dev_indices] - self.centroids[c], axis=1)
                radius = float(np.max(dists_to_centroid))
            else:
                radius = 0.0

            cluster_radii.append(radius)
            stats.append({
                'cluster_id': c,
                'centroid': self.centroids[c].tolist(),
                'n_devices': count,
                'coverage_radius_m': radius,
            })

        summary = {
            'clusters': stats,
            'max_cluster_size': int(np.max(cluster_device_counts)) if len(cluster_device_counts) > 0 else 0,
            'max_coverage_radius_m': float(np.max(cluster_radii)) if len(cluster_radii) > 0 else 0.0,
            'clustering_objective_J': self.compute_clustering_objective(iot_devices),
            'offline_runtime_sec': float(self.last_fit_time),
            'iterations_run': int(self.iterations_run),
            'uncovered_node_count': len(self.uncovered_nodes),
            'uncovered_nodes': self.uncovered_nodes,
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


# ------------------------------------------------------------------------------
# Backward Compatibility & Legacy Ablation
# ------------------------------------------------------------------------------
WorkloadAwareClusterManager = SpatialClusterManager


class LegacyWorkloadClusterManager:
    """
    Legacy Ablation: Workload-Weighted K-Means assuming prior oracle arrival rates.
    Retained solely for ablation benchmarking against the proposed SpatialClusterManager.
    """

    def __init__(self, n_clusters=NUM_CLUSTERS, lambda_w=LAMBDA_W, random_state=42):
        self.n_clusters = n_clusters
        self.lambda_w = lambda_w
        self.rng = np.random.RandomState(random_state)
        self.centroids = None
        self.cluster_assignments = None
        self.device_weights = None

    def fit_clusters(self, iot_devices, max_iter=100, tol=1e-4):
        positions = np.array([dev.position[:2] for dev in iot_devices], dtype=float)
        n_devices = len(iot_devices)
        demands = np.array([getattr(d, 'nominal_demand', 1.5e6 * 500 * 0.7) for d in iot_devices])
        w_max = max(np.max(demands), 1.0)
        weights = 1.0 + self.lambda_w * (demands / w_max)
        self.device_weights = weights

        n_clusters = min(self.n_clusters, n_devices)
        centroids = positions[:n_clusters].copy()
        assignments = np.zeros(n_devices, dtype=int)

        for _ in range(max_iter):
            distances = np.linalg.norm(positions[:, None, :] - centroids[None, :, :], axis=2)
            new_assignments = np.argmin(distances, axis=1)
            if np.array_equal(new_assignments, assignments):
                break
            assignments = new_assignments
            for c in range(n_clusters):
                mask = (assignments == c)
                if np.any(mask):
                    centroids[c] = np.sum(positions[mask] * weights[mask, None], axis=0) / np.sum(weights[mask])

        for idx, dev in enumerate(iot_devices):
            dev.cluster_id = int(assignments[idx])
        self.centroids = centroids
        self.cluster_assignments = assignments
        return centroids, assignments


# ==============================================================================
# Exact K-Means++ & Geometry Selection (Experiment B0)
# ==============================================================================
class SpatialKMeans:
    """
    Deterministic Pure-NumPy K-Means++ implementation matching scikit-learn
    KMeans(init='k-means++', algorithm='lloyd', n_init=20) specifications.
    """

    def __init__(self, n_clusters=4, n_init=20, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.all_inertias_ = []

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=float)
        N = len(X)
        K = min(self.n_clusters, N)

        master_rng = np.random.RandomState(self.random_state)
        restart_seeds = master_rng.randint(0, 1000000, size=self.n_init)

        best_inertia = np.inf
        best_centers = None
        best_labels = None
        self.all_inertias_ = []

        for seed in restart_seeds:
            rng = np.random.RandomState(seed)
            # 1. K-Means++ Initialization
            centers = np.zeros((K, X.shape[1]), dtype=float)
            idx0 = rng.randint(0, N)
            centers[0] = X[idx0]

            for k in range(1, K):
                dists_sq = np.min(np.sum((X[:, None, :] - centers[:k, :][None, :, :]) ** 2, axis=2), axis=1)
                total_d = np.sum(dists_sq)
                if total_d > 0:
                    probs = dists_sq / total_d
                else:
                    probs = np.ones(N) / N
                next_idx = rng.choice(N, p=probs)
                centers[k] = X[next_idx]

            # 2. Lloyd Iterations
            labels = np.zeros(N, dtype=int)
            for _ in range(self.max_iter):
                # Step 1: Assign to nearest centroid
                sq_dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                new_labels = np.argmin(sq_dists, axis=1)

                # Step 2: Empty cluster repair
                counts = np.bincount(new_labels, minlength=K)
                for c in range(K):
                    if counts[c] == 0:
                        cur_dists = sq_dists[np.arange(N), new_labels]
                        farthest = np.argmax(cur_dists)
                        new_labels[farthest] = c
                        centers[c] = X[farthest].copy()
                        counts = np.bincount(new_labels, minlength=K)

                # Step 3: Unweighted Centroid Update (Arithmetic Mean)
                new_centers = np.zeros_like(centers)
                for c in range(K):
                    mask = (new_labels == c)
                    if np.any(mask):
                        new_centers[c] = np.mean(X[mask], axis=0)
                    else:
                        new_centers[c] = centers[c]

                shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
                centers = new_centers
                labels = new_labels
                if shift < self.tol:
                    break

            # Calculate inertia: sum_c sum_{i in I_c} ||p_i - z_c||^2
            min_sq = np.min(np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
            inertia = float(np.sum(min_sq))
            self.all_inertias_.append(inertia)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


def compute_silhouette_score(X, labels):
    """
    Computes mean Euclidean silhouette coefficient S(C) across all samples:
      s(i) = (b(i) - a(i)) / max(a(i), b(i))
    """
    X = np.ascontiguousarray(X, dtype=float)
    N = len(X)
    unique_labels = np.unique(labels)
    K = len(unique_labels)
    if K <= 1 or K >= N:
        return 0.0

    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    s_vals = np.zeros(N)

    for i in range(N):
        c_i = labels[i]
        same_cluster = (labels == c_i)
        same_count = np.sum(same_cluster)

        if same_count <= 1:
            s_vals[i] = 0.0
            continue

        a_i = np.sum(dists[i, same_cluster]) / (same_count - 1)

        b_i = np.inf
        for c in unique_labels:
            if c == c_i:
                continue
            other_cluster = (labels == c)
            other_count = np.sum(other_cluster)
            if other_count > 0:
                mean_dist = np.sum(dists[i, other_cluster]) / other_count
                if mean_dist < b_i:
                    b_i = mean_dist

        max_ab = max(a_i, b_i)
        if max_ab > 1e-12:
            s_vals[i] = (b_i - a_i) / max_ab
        else:
            s_vals[i] = 0.0

    return float(np.mean(s_vals))


def compute_r_plan_numerical(h_nom=100.0, p_tx=0.1, b_user=None, r_min=1e5):
    """
    Derives navigation planning radius R_plan numerically from ChannelModel.
    """
    try:
        from src.channel import ChannelModel
        from config import UAV_PROFILES, K_COMM_DEFAULT
        if b_user is None:
            bw_nom = float(UAV_PROFILES[0]['bandwidth'])
            b_user = bw_nom / max(1, K_COMM_DEFAULT)
        ch = ChannelModel()
    except Exception:
        return 280.0

    r_low = 0.0
    r_high = 1200.0
    for _ in range(60):
        r_mid = (r_low + r_high) / 2.0
        rate = ch.compute_uplink_rate([r_mid, 0.0, 0.0], [0.0, 0.0, h_nom], p_tx, b_user)
        if rate >= r_min:
            r_low = r_mid
        else:
            r_high = r_mid
    return float(r_low)


class GeometricClusterSelector:
    """
    Performs pre-deployment spatial K-means++ clustering screening and selects
    the geometry-selected cluster count C_geo (the manuscript's C^*).
    """

    def __init__(self, c_candidates=None, n_init=20, max_iter=300, tol=1e-4,
                 random_state=42, h_nom=100.0, p_tx=0.1, r_min=1e5):
        self.c_candidates = c_candidates or list(range(2, 13))
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.h_nom = h_nom
        self.p_tx = p_tx
        self.r_min = r_min
        self.r_plan = compute_r_plan_numerical(h_nom=h_nom, p_tx=p_tx, r_min=r_min)

    def screen_geometry(self, positions_xy):
        """
        Screen candidate cluster counts on 2D ground coordinates positions_xy (N, 2).
        Returns:
          results_table: list of dicts for each tested C
          c_geo: the selected cluster count
          models: dict {C: model_info}
        """
        N = len(positions_xy)
        positions_xy = np.ascontiguousarray(positions_xy, dtype=float)

        results = []
        models = {}

        for C in self.c_candidates:
            if C >= N:
                continue

            t0 = time.perf_counter()
            km = SpatialKMeans(
                n_clusters=C,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state
            ).fit(positions_xy)
            fit_time = time.perf_counter() - t0

            labels = km.labels_
            centroids = km.cluster_centers_
            inertia = km.inertia_
            inertia_per_node = inertia / max(1, N)

            # Compute silhouette score
            sil = compute_silhouette_score(positions_xy, labels)

            # Compute per-cluster radii and r_max
            cluster_radii = []
            cluster_sizes = []
            for c_idx in range(C):
                pts = positions_xy[labels == c_idx]
                cluster_sizes.append(len(pts))
                if len(pts) > 0:
                    r_c = float(np.max(np.linalg.norm(pts - centroids[c_idx], axis=1)))
                else:
                    r_c = 0.0
                cluster_radii.append(r_c)

            r_max = float(np.max(cluster_radii)) if cluster_radii else 0.0
            r_mean = float(np.mean(cluster_radii)) if cluster_radii else 0.0
            is_feasible = (r_max <= self.r_plan)

            row = {
                'C': C,
                'inertia': inertia,
                'inertia_per_node': inertia_per_node,
                'silhouette': sil,
                'r_max': r_max,
                'r_mean': r_mean,
                'r_plan': self.r_plan,
                'radius_feasible': is_feasible,
                'min_cluster_size': int(np.min(cluster_sizes)) if cluster_sizes else 0,
                'max_cluster_size': int(np.max(cluster_sizes)) if cluster_sizes else 0,
                'fit_time_ms': fit_time * 1000.0,
                'selected_c_geo': False,
                'selection_reason': "",
            }
            results.append(row)
            models[C] = {
                'centroids': centroids,
                'labels': labels,
                'inertia': inertia,
                'all_inertias': km.all_inertias_,
                'silhouette': sil,
                'r_max': r_max,
                'cluster_radii': cluster_radii,
                'cluster_sizes': cluster_sizes,
                'fit_time_ms': fit_time * 1000.0,
            }

        # ----------------------------------------------------------------------
        # Apply Exact Selection Rule
        # ----------------------------------------------------------------------
        k_plan = [r for r in results if r['radius_feasible']]

        if len(k_plan) > 0:
            # Maximizes silhouette with smaller-C tie break
            sorted_k = sorted(k_plan, key=lambda r: (-r['silhouette'], r['C']))
            selected_row = sorted_k[0]
            selection_reason = "max_silhouette_in_K_plan"
        else:
            # Fallback: candidate with smallest r_max, breaking ties by higher silhouette
            sorted_k = sorted(results, key=lambda r: (r['r_max'], -r['silhouette'], r['C']))
            selected_row = sorted_k[0]
            selection_reason = "min_rmax_fallback_K_plan_empty"

        c_geo = selected_row['C']
        for r in results:
            if r['C'] == c_geo:
                r['selected_c_geo'] = True
                r['selection_reason'] = selection_reason

        return results, c_geo, models

    def export_csv(self, results, out_path):
        import csv
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        keys = list(results[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

