# RESULTS SUMMARY
**Paper Title:** Joint Task Offloading and Adaptive Trajectory Optimization in Trust- and Fault-Aware Multi-UAV Edge IoT Networks  
**Target Journal:** IEEE Internet of Things Journal  
**Execution Environment:** macOS Darwin, Python 3.14, PyTorch 2.13.0, NumPy 2.5.2  
**Evaluation Date:** November 2026  

---

## 1. Quantitative Benchmark Results

The benchmark comparison was executed using `scripts/run_benchmarks.py` across five baseline methods under identical channel realizations and traffic demands. The summary metrics are recorded in `outputs/Milestone_4_baselines/benchmark_results.csv` and `results/logs/benchmark_comparison.json`:

| Method | Average Latency (s) | 95% CI (s) | Total Energy (J) | 95% CI (J) | Deadline Completion (%) | 95% CI (%) |
|---|---|---|---|---|---|---|
| **TF-MAPPO (Proposed)** | 2.141 | $\pm$ 0.140 | 53,084.7 | $\pm$ 4,583.6 | 3.22% | $\pm$ 1.40% |
| **MAPPO-NoMask** | 2.171 | $\pm$ 0.008 | 53,012.0 | $\pm$ 3,873.9 | 2.01% | $\pm$ 0.75% |
| **MAPPO-NoTrust** | 2.132 | $\pm$ 0.228 | 49,338.1 | $\pm$ 4,088.8 | 1.80% | $\pm$ 0.23% |
| **Dynamic Greedy** | 2.199 | $\pm$ 0.100 | 8,482.8 | $\pm$ 0.0 | 11.59% | $\pm$ 0.17% |
| **Local Only** | 0.000* | $\pm$ 0.000 | 8,424.6 | $\pm$ 0.0 | 0.00%* | $\pm$ 0.00% |

*\*Note: Under Local-Only baseline, no UAV MEC offloading occurs; tasks that exceed the local device deadline of [1.0, 3.0] s on 0.5 GHz local microcontrollers expire unserved.*

---

## 2. Validation Gate Results

All 12 required test gates from Section 8 of the Master Prompt were verified using `tests/test_audit_gates.py` and `tests/test_sanity.py`:
- **Gate 1 (Clustering Partition & Centroid Invariance):** PASSED (100% device coverage, no empty clusters, weighted centroid center-of-mass verified, `lambda_w=0` and zero-demand edge cases validated).
- **Gate 2 (BS Commissioning Overhead):** PASSED (32-byte authenticated metadata report per node accurately accounted).
- **Gate 3 (Channel Monotonicity & Valid Power):** PASSED (Monotonic path loss with distance, positive rates, non-negative gains, corrected positive NLoS attenuation $\eta_N = 20.0$ dB).
- **Gate 4 (Causal Task Timestamps & Lifecycle):** PASSED ($t_{\text{gen}} \le t_{\text{up}} \le t_{\text{serv\_start}} \le t_{\text{finish}}$, latency equals finish minus arrival).
- **Gate 5 (Task Completion Ratio Normalization):** PASSED (Normalized strictly by $N_{\text{gen}}$, not $N \cdot T$).
- **Gate 6 (Multi-Slot Queue Continuity):** PASSED (Onboard computation queues continue executing pending tasks across slots without requiring new arrivals).
- **Gate 7 (Trust & Fault Eligibility):** PASSED (UAVs with $\tau_m=0$ or $\phi_m=0$ masked from new cluster service; cached certificates valid until expiration).
- **Gate 8 (Battery & Aerodynamic Hover Power):** PASSED (Rotary-wing zero-speed power verified at $168.49$ W).
- **Gate 9 (Continuous-Time Collision Shield):** PASSED (Mid-slot crossing trajectories with safe endpoints detected and projected to safe separation $d_{\min} \ge 10.0$ m).
- **Gate 10 (Cluster Exclusivity):** PASSED ($\sum_m x_{m,c}(t) \le 1$ enforced deterministically).
- **Gate 11 (PPO Finite Probability Ratios):** PASSED (Finite log-probabilities and entropy under masks; checkpoint load/save verified).
- **Gate 12 (Reward Decomposition Consistency):** PASSED (Plotted and logged reward strictly equals the sum of its decomposed terms in `reward_components.csv`).

---

## 3. Publication Artifacts

All 10 required vector PDF and PNG figures have been generated in `outputs/Milestone_5_final_evaluation/`:
- `fig1_convergence.pdf` / `.png`
- `fig2_energy_comparison.pdf` / `.png`
- `fig3_latency_breakdown.pdf` / `.png`
- `fig4_completion_ratio.pdf` / `.png`
- `fig5_scalability_N.pdf` / `.png`
- `fig6_scalability_M.pdf` / `.png`
- `fig7_trajectories_clusters.pdf` / `.png`
- `fig8_fault_timeline.pdf` / `.png`
- `fig9_ablation_study.pdf` / `.png`
- `fig10_weight_sensitivity.pdf` / `.png`

