"""
Extended Deep Convergence Training Runner
=========================================
Executes extended full-scale training for the primary models:
  - PROPOSED (C=4, M=4, N=80): 900,000 steps (as specified in config/reference.yaml)
  - PROPOSED (C=2, M=4, N=80): 300,000 steps (validation-best model deep convergence)
  - MAPPO (C=4, M=4, N=80): 150,000 steps (unmasked baseline convergence)
  - IPPO (C=4, M=4, N=80): 150,000 steps (decentralized critic baseline convergence)

Followed by evaluation and re-rendering of all 18 publication-grade IEEE figures.
"""

import os
import sys
import time

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_experiments import run_training, evaluate_policy, generate_all_publication_figures

def main():
    results_dir = "results/new_experiment"
    figures_dir = "outputs/paper_figures"
    eval_slots = 300
    seed = 42

    print("\n" + "=" * 80)
    print("   EXTENDED PRODUCTION CONVERGENCE TRAINING PIPELINE (900,000 STEPS)   ")
    print("=" * 80)
    start_total = time.time()

    # 1. Hero Model: PROPOSED at C=4 (900,000 joint steps as specified in reference.yaml)
    print("\n[Phase 1/4] Training PROPOSED at Reference C=4 for 900,000 steps...")
    run_training(
        method="PROPOSED",
        seed=seed,
        total_steps=900000,
        num_iot=80,
        num_uavs=4,
        num_clusters=4,
        results_dir=results_dir
    )
    tag_p4 = f"PROPOSED_C4_M4_seed{seed}"
    ckpt_p4 = os.path.join(results_dir, "checkpoints", tag_p4, "checkpoint_best.pt")
    evaluate_policy(
        method="PROPOSED",
        ckpt_path=ckpt_p4,
        seed=seed,
        num_iot=80,
        num_uavs=4,
        num_clusters=4,
        num_slots=eval_slots,
        results_dir=results_dir,
        tag="PROPOSED_C4"
    )

    # 2. Validation-Best Model: PROPOSED at C=2 (300,000 steps)
    print("\n[Phase 2/4] Training PROPOSED at Validation-Best C=2 for 300,000 steps...")
    run_training(
        method="PROPOSED",
        seed=seed,
        total_steps=300000,
        num_iot=80,
        num_uavs=4,
        num_clusters=2,
        results_dir=results_dir
    )
    tag_p2 = f"PROPOSED_C2_M4_seed{seed}"
    ckpt_p2 = os.path.join(results_dir, "checkpoints", tag_p2, "checkpoint_best.pt")
    evaluate_policy(
        method="PROPOSED",
        ckpt_path=ckpt_p2,
        seed=seed,
        num_iot=80,
        num_uavs=4,
        num_clusters=2,
        num_slots=eval_slots,
        results_dir=results_dir,
        tag="PROPOSED_C2"
    )

    # 3. Unmasked Baseline: MAPPO at C=4 (150,000 steps)
    print("\n[Phase 3/4] Training MAPPO (No Mask) at C=4 for 150,000 steps...")
    run_training(
        method="MAPPO",
        seed=seed,
        total_steps=150000,
        num_iot=80,
        num_uavs=4,
        num_clusters=4,
        results_dir=results_dir
    )
    tag_m4 = f"MAPPO_C4_M4_seed{seed}"
    ckpt_m4 = os.path.join(results_dir, "checkpoints", tag_m4, "checkpoint_best.pt")
    evaluate_policy(
        method="MAPPO",
        ckpt_path=ckpt_m4,
        seed=seed,
        num_iot=80,
        num_uavs=4,
        num_clusters=4,
        num_slots=eval_slots,
        results_dir=results_dir,
        tag="MAPPO_C4"
    )

    # 4. CTDE Baseline: IPPO at C=4 (150,000 steps)
    print("\n[Phase 4/4] Training IPPO (Decentralized Critic) at C=4 for 150,000 steps...")
    run_training(
        method="IPPO",
        seed=seed,
        total_steps=150000,
        num_iot=80,
        num_uavs=4,
        num_clusters=4,
        results_dir=results_dir
    )
    tag_i4 = f"IPPO_C4_M4_seed{seed}"
    ckpt_i4 = os.path.join(results_dir, "checkpoints", tag_i4, "checkpoint_best.pt")
    evaluate_policy(
        method="IPPO",
        ckpt_path=ckpt_i4,
        seed=seed,
        num_iot=80,
        num_uavs=4,
        num_clusters=4,
        num_slots=eval_slots,
        results_dir=results_dir,
        tag="IPPO_C4"
    )

    # Re-render all publication figures with converged curves
    print("\n[✓] All extended training complete! Updating IEEE publication figures...")
    generate_all_publication_figures(results_dir, figures_dir, c_geo=12, c_val=2)

    total_min = (time.time() - start_total) / 60.0
    print(f"\n[★] Extended Training Pipeline Finished in {total_min:.1f} minutes!")

if __name__ == "__main__":
    main()

