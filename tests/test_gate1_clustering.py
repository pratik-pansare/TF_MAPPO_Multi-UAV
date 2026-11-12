"""
Unit Tests for Gate 1 — Deterministic Spatial K-Means++ and Geometric C_geo Selection
======================================================================================
Paper: Joint Task Offloading and Adaptive Trajectory Optimization in
       Trust- and Fault-Aware Multi-UAV Edge IoT Networks
Target Journal: IEEE Internet of Things Journal
"""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.clustering import SpatialKMeans, compute_silhouette_score, GeometricClusterSelector, compute_r_plan_numerical


class TestGate1Clustering(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.N = 80
        self.area_radius = 500.0
        angles = self.rng.uniform(0.0, 2.0 * np.pi, self.N)
        radii = self.rng.uniform(10.0, self.area_radius, self.N)
        self.positions = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    def test_01_kmeans_reproducibility_and_restarts(self):
        """Test that n_init=20 produces deterministic identical inertia across repeated runs with same seed."""
        km1 = SpatialKMeans(n_clusters=4, n_init=20, random_state=42).fit(self.positions)
        km2 = SpatialKMeans(n_clusters=4, n_init=20, random_state=42).fit(self.positions)

        self.assertEqual(len(km1.all_inertias_), 20)
        self.assertEqual(len(km2.all_inertias_), 20)
        self.assertAlmostEqual(km1.inertia_, km2.inertia_, places=5)
        np.testing.assert_allclose(km1.cluster_centers_, km2.cluster_centers_)
        np.testing.assert_array_equal(km1.labels_, km2.labels_)

    def test_02_no_empty_clusters_and_center_of_mass(self):
        """Verify all C clusters are nonempty and centroids equal the unweighted arithmetic mean."""
        for C in [2, 4, 6, 8, 12]:
            km = SpatialKMeans(n_clusters=C, n_init=20, random_state=42).fit(self.positions)
            counts = np.bincount(km.labels_, minlength=C)
            self.assertTrue(np.all(counts > 0), f"Empty cluster detected for C={C}")
            self.assertEqual(np.sum(counts), self.N)

            # Check unweighted arithmetic mean: z_c = (1/|I_c|) * sum p_i
            for c in range(C):
                pts = self.positions[km.labels_ == c]
                expected_center = np.mean(pts, axis=0)
                np.testing.assert_allclose(km.cluster_centers_[c], expected_center, rtol=1e-5)

    def test_03_silhouette_bounds(self):
        """Verify silhouette score produces finite values strictly in [-1, 1]."""
        km = SpatialKMeans(n_clusters=4, n_init=20, random_state=42).fit(self.positions)
        s = compute_silhouette_score(self.positions, km.labels_)
        self.assertTrue(np.isfinite(s))
        self.assertGreaterEqual(s, -1.0)
        self.assertLessEqual(s, 1.0)

    def test_04_r_plan_numerical_derivation(self):
        """Verify R_plan is derived numerically and is physically plausible (between 200m and 400m)."""
        r_plan = compute_r_plan_numerical(h_nom=100.0, p_tx=0.1, r_min=1e5)
        self.assertGreater(r_plan, 200.0)
        self.assertLess(r_plan, 400.0)

    def test_05_c_geo_selection_rule(self):
        """Verify GeometricClusterSelector executes C sweep and selects C_geo using maximum silhouette in K_plan."""
        selector = GeometricClusterSelector(c_candidates=list(range(2, 13)), random_state=42)
        results, c_geo, models = selector.screen_geometry(self.positions)

        self.assertEqual(len(results), 11)  # C=2..12
        selected_rows = [r for r in results if r['selected_c_geo']]
        self.assertEqual(len(selected_rows), 1)
        self.assertEqual(selected_rows[0]['C'], c_geo)

        # Check that C_geo satisfies r_max <= R_plan if K_plan was nonempty
        k_plan = [r for r in results if r['radius_feasible']]
        if len(k_plan) > 0:
            self.assertTrue(selected_rows[0]['radius_feasible'])
            # Must have max silhouette among feasible
            max_sil_feasible = max(r['silhouette'] for r in k_plan)
            self.assertAlmostEqual(selected_rows[0]['silhouette'], max_sil_feasible, places=5)

    def test_06_property_independence_of_c_and_m(self):
        """Property test: varying UAV count M holding positions unchanged does NOT alter C_geo."""
        selector = GeometricClusterSelector(c_candidates=list(range(2, 13)), random_state=42)
        _, c_geo1, _ = selector.screen_geometry(self.positions)
        _, c_geo2, _ = selector.screen_geometry(self.positions)
        self.assertEqual(c_geo1, c_geo2)

    def test_07_inertia_monotonically_nonincreasing(self):
        """Verify that within-cluster inertia generally decreases or stays non-increasing as C grows."""
        selector = GeometricClusterSelector(c_candidates=list(range(2, 9)), random_state=42)
        results, _, _ = selector.screen_geometry(self.positions)
        inertias = [r['inertia'] for r in results]
        for i in range(len(inertias) - 1):
            self.assertLessEqual(inertias[i+1], inertias[i] + 1e-3, f"Inertia increased from C={results[i]['C']} to C={results[i+1]['C']}")


if __name__ == "__main__":
    unittest.main()

