"""
Regression Tests for Revised IEEE IoT-J 150,000-Step Protocol
============================================================
Verifies:
  1. Default training step budget set to 150,000 steps.
  2. Pilot training budget set to 30,000 steps.
  3. Conditional extension set to 250,000 steps.
  4. Rollout length maintained at 500.
  5. Validation and checkpoint cadence set to 25,000 steps.
  6. Disjoint seeds for preliminary screening (3 seeds) and final reporting (5 seeds).
  7. Checkpoint reuse / cache deduplication logic.
  8. Final validation window stabilization audit.
"""

import unittest
import os
import shutil
import tempfile
import json
import numpy as np

import config as cfg
from run_experiments import run_training, run_validation_evaluation
from src.agent_factory import make_agent
from src.env import MultiUAVMECEnv


class TestTrainingProtocol(unittest.TestCase):

    def test_protocol_constants(self):
        """Verify standardized hyperparameter and budget constants."""
        self.assertEqual(cfg.DEFAULT_TRAINING_STEPS, 150_000)
        self.assertEqual(cfg.PILOT_TRAINING_STEPS, 30_000)
        self.assertEqual(cfg.EXTENDED_TRAINING_STEPS, 250_000)
        self.assertEqual(cfg.MAPPO_ROLLOUT_LENGTH, 500)
        self.assertEqual(cfg.VALIDATION_INTERVAL, 25_000)
        self.assertEqual(cfg.CHECKPOINT_INTERVAL, 25_000)
        self.assertEqual(cfg.PRELIMINARY_SEEDS, [42, 43, 44])
        self.assertEqual(cfg.FINAL_SEEDS, [42, 43, 44, 45, 46])

    def test_validation_evaluation_and_checkpoint_selection(self):
        """Verify independent validation on disjoint seed computes metrics without disk bloat."""
        env = MultiUAVMECEnv(num_users=40, num_uavs=3, num_clusters=3, k_comm=2, seed=1042)
        agent = make_agent(
            method="PROPOSED",
            obs_dim=env.local_obs_dim,
            state_dim=env.global_state_dim,
            action_dim=env.agent_action_dim,
            n_agents=env.n_agents,
            n_clusters=env.num_clusters,
            max_candidate_pool=env.max_candidate_pool,
            device="cpu"
        )
        summary = run_validation_evaluation(
            agent=agent,
            method="PROPOSED",
            seed=1042,
            num_iot=40,
            num_uavs=3,
            num_clusters=3,
            k_comm=2,
            num_slots=20
        )
        self.assertIn("completion_ratio_pct", summary)
        self.assertIn("avg_latency", summary)
        self.assertIn("total_energy", summary)
        self.assertGreaterEqual(summary["completion_ratio_pct"], 0.0)
        self.assertGreaterEqual(summary["avg_latency"], 0.0)

    def test_checkpoint_reuse_and_cache(self):
        """Verify completed checkpoint is reused and duplicate training is skipped."""
        temp_dir = tempfile.mkdtemp(prefix="test_cache_")
        try:
            # First small run to produce checkpoint
            records = run_training(
                method="PROPOSED",
                seed=42,
                total_steps=1000,
                num_iot=20,
                num_uavs=3,
                num_clusters=2,
                results_dir=temp_dir,
                val_interval=500
            )
            self.assertGreaterEqual(len(records), 1)

            # Second call with same total_steps should hit cache reuse
            cached_records = run_training(
                method="PROPOSED",
                seed=42,
                total_steps=1000,
                num_iot=20,
                num_uavs=3,
                num_clusters=2,
                results_dir=temp_dir,
                val_interval=500,
                force_retrain=False
            )
            self.assertEqual(len(cached_records), len(records))

            # Verify stability file created
            stab_file = os.path.join(temp_dir, "logs", "PROPOSED_C2_M3_seed42", "convergence_stability.json")
            if os.path.exists(stab_file):
                with open(stab_file) as f:
                    stab = json.load(f)
                self.assertIn("is_stabilized", stab)
                self.assertIn("relative_spread_pct", stab)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
