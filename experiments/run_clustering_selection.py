"""
Experiment B0: Pre-Deployment Spatial K-Means++ Clustering & Geometric Count Selection
======================================================================================
Paper: Joint Task Offloading and Adaptive Trajectory Optimization in
       Trust- and Fault-Aware Multi-UAV Edge IoT Networks
Target Journal: IEEE Internet of Things Journal

Executes:
  1. Spatial K-Means screening for C in [2 .. 12] on N=80 static IoT nodes.
  2. Numerical planning radius R_plan derivation from ChannelModel.
  3. Manuscript selection rule:
       K_plan = {C in K | r_max(C) <= R_plan}
       C* = argmax_{C in K_plan} S(C); tie-break to smaller C.
  4. Generates:
       - outputs/clustering/cluster_selection_metrics.csv
       - outputs/paper_figures/fig04_cluster_inertia_silhouette_radius.pdf / .png
       - outputs/paper_figures/fig05_cluster_layouts.pdf / .png
       - outputs/clustering/cluster_map_C4.pdf / .png
       - outputs/clustering/cluster_map_auto_Cstar.pdf / .png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clustering import GeometricClusterSelector, compute_r_plan_numerical
from config import AREA_RADIUS

# Visual Styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.titlesize': 12,
    'axes.grid': True,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173'
]


def run_experiment_b0(num_devices=80, area_radius=500.0, seed=42,
                      output_dir="outputs/clustering",
                      figures_dir="outputs/paper_figures"):
    print("=" * 75)
    print("   EXPERIMENT B0: GEOMETRIC CLUSTER-COUNT SCREENING & SELECTION (C_geo)   ")
    print("=" * 75)
    print(f"  IoT Devices N   : {num_devices}")
    print(f"  Area Radius     : {area_radius} m")
    print(f"  Random Seed     : {seed}")
    print(f"  Output CSV Path : {output_dir}")
    print(f"  Figures Path    : {figures_dir}")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Generate Static IoT Device Coordinates
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, num_devices)
    radii = rng.uniform(10.0, area_radius, num_devices)
    positions_xy = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # 2. Geometric Screening
    candidates = list(range(2, 13))
    selector = GeometricClusterSelector(c_candidates=candidates, random_state=seed)
    results, c_geo, models = selector.screen_geometry(positions_xy)

    csv_path = os.path.join(output_dir, "cluster_selection_metrics.csv")
    selector.export_csv(results, csv_path)
    print(f"\n  [✓] Geometry screening complete. Saved table: {csv_path}")
    print(f"  [★] Selected Geometry Cluster Count C_geo = {c_geo} (Planning Radius R_plan = {selector.r_plan:.2f} m)\n")

    for r in results:
        marker = " [★ C_geo SELECTED]" if r['selected_c_geo'] else ""
        print(f"    C={r['C']:2d} | Inertia={r['inertia']:10.1f} m² | Sil={r['silhouette']:5.3f} | "
              f"r_max={r['r_max']:5.1f}m (<= {r['r_plan']:.1f}m: {r['radius_feasible']}){marker}")

    # --------------------------------------------------------------------------
    # Fig 04: Three-Panel Geometry Screening
    # --------------------------------------------------------------------------
    print("\n  [*] Rendering Fig 04: Three-Panel Geometry Screening...")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)

    c_vals = [r['C'] for r in results]
    inertias = [r['inertia'] / 1e6 for r in results]
    silhouettes = [r['silhouette'] for r in results]
    r_maxs = [r['r_max'] for r in results]
    r_plan = selector.r_plan

    # Panel (a): Inertia (Elbow)
    axs[0].plot(c_vals, inertias, marker='o', lw=1.8, color='#1d3557', label=r'Inertia $J_{\mathrm{clu}}(C)$')
    axs[0].axvline(c_geo, color='#e63946', linestyle='--', lw=1.4, label=rf'Selected $C_{{\mathrm{{geo}}}} = {c_geo}$')
    axs[0].set_xlabel('Spatial Cluster Count $C$')
    axs[0].set_ylabel(r'Spatial Inertia $J_{\mathrm{clu}}$ ($\times 10^6$ m$^2$)')
    axs[0].set_title(r'(a) Within-Cluster Inertia $J_{\mathrm{clu}}(C)$')
    axs[0].set_xticks(c_vals)
    axs[0].legend(loc='upper right', fontsize=8)

    # Panel (b): Mean Silhouette S(C)
    colors_sil = ['#2a9d8f' if r['radius_feasible'] else '#adb5bd' for r in results]
    axs[1].bar(c_vals, silhouettes, width=0.55, color=colors_sil, edgecolor='black', alpha=0.85)
    axs[1].axvline(c_geo, color='#e63946', linestyle='--', lw=1.4, label=rf'Selected $C_{{\mathrm{{geo}}}} = {c_geo}$')
    axs[1].set_xlabel('Spatial Cluster Count $C$')
    axs[1].set_ylabel(r'Mean Silhouette Coefficient $S(C)$')
    axs[1].set_title(r'(b) Silhouette Profile $S(C)$ (Feasible Candidates Green)')
    axs[1].set_xticks(c_vals)
    axs[1].legend(loc='lower right', fontsize=8)

    # Panel (c): Maximum Cluster Radius r_max vs R_plan
    axs[2].plot(c_vals, r_maxs, marker='s', lw=1.8, color='#457b9d', label=r'Max Cluster Radius $r_{\max}(C)$')
    axs[2].axhline(r_plan, color='#e76f51', linestyle='--', lw=1.8, label=rf'Planning Radius $R_{{\mathrm{{plan}}}} = {r_plan:.1f}$ m')
    axs[2].fill_between(c_vals, 0, r_plan, color='#2a9d8f', alpha=0.1, label=r'Feasible Region ($r_{\max} \leq R_{\mathrm{plan}}$)')
    axs[2].scatter([c_geo], [models[c_geo]['r_max']], color='#e63946', s=90, zorder=6, marker='*', label=rf'$C_{{\mathrm{{geo}}}}={c_geo}$')
    axs[2].set_xlabel('Spatial Cluster Count $C$')
    axs[2].set_ylabel(r'Maximum Cluster Radius $r_{\max}(C)$ (m)')
    axs[2].set_title(r'(c) Max Cluster Radius vs. Physical Planning Bound')
    axs[2].set_xticks(c_vals)
    axs[2].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    fig4_pdf = os.path.join(figures_dir, "fig04_cluster_inertia_silhouette_radius.pdf")
    fig4_png = os.path.join(figures_dir, "fig04_cluster_inertia_silhouette_radius.png")
    fig.savefig(fig4_pdf)
    fig.savefig(fig4_png)
    plt.close(fig)
    print(f"  [✓] Fig 04 saved: {fig4_pdf}")

    # --------------------------------------------------------------------------
    # Fig 05: Side-by-Side Spatial Cluster Layouts (C=2, 4, 6, 8, 12, C_geo)
    # --------------------------------------------------------------------------
    print("\n  [*] Rendering Fig 05: Multi-Cluster Spatial Layouts on Matched Coordinates...")
    sweep_c_vals = [2, 4, 6, 8, 12]
    if c_geo not in sweep_c_vals:
        sweep_c_vals.append(c_geo)
    sweep_c_vals = sorted(list(set(sweep_c_vals)))

    n_plots = len(sweep_c_vals)
    fig, axs = plt.subplots(1, n_plots, figsize=(3.8 * n_plots, 4.0), dpi=300)
    if n_plots == 1:
        axs = [axs]

    for idx, c_val in enumerate(sweep_c_vals):
        ax = axs[idx]
        mod = models[c_val]
        labels = mod['labels']
        centroids = mod['centroids']

        # Boundary circle
        ax.add_patch(plt.Circle((0, 0), area_radius, color='gray', fill=False, linestyle=':', lw=1.0))

        # Scatter IoT nodes colored by cluster
        for c in range(c_val):
            mask = (labels == c)
            col = PALETTE[c % len(PALETTE)]
            ax.scatter(positions_xy[mask, 0], positions_xy[mask, 1], s=14, color=col, alpha=0.7)

            # Draw coverage radius around centroid
            ax.add_patch(plt.Circle((centroids[c, 0], centroids[c, 1]), mod['cluster_radii'][c],
                                    color=col, fill=False, linestyle='--', alpha=0.3, lw=0.9))

        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=70, color='black', zorder=5)

        title = f"$C = {c_val}$"
        if c_val == 4:
            title += " (Ref)"
        if c_val == c_geo:
            title += r" ($\mathbf{C}_{\mathbf{geo}}$)"
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(-530, 530)
        ax.set_ylim(-530, 530)
        ax.set_xlabel('X Coordinate (m)')
        if idx == 0:
            ax.set_ylabel('Y Coordinate (m)')

    plt.tight_layout()
    fig5_pdf = os.path.join(figures_dir, "fig05_cluster_layouts.pdf")
    fig5_png = os.path.join(figures_dir, "fig05_cluster_layouts.png")
    fig.savefig(fig5_pdf)
    fig.savefig(fig5_png)
    plt.close(fig)
    print(f"  [✓] Fig 05 saved: {fig5_pdf}")

    # --------------------------------------------------------------------------
    # Individual Diagnostic Maps: C=4 and C=C_geo
    # --------------------------------------------------------------------------
    for target_c, fname in [(4, "cluster_map_C4"), (c_geo, "cluster_map_auto_Cstar")]:
        fig, ax = plt.subplots(figsize=(6.0, 5.5), dpi=300)
        mod = models[target_c]
        ax.add_patch(plt.Circle((0, 0), area_radius, color='gray', fill=False, linestyle=':', lw=1.2, label='Service Area (500m)'))

        for c in range(target_c):
            mask = (mod['labels'] == c)
            col = PALETTE[c % len(PALETTE)]
            ax.scatter(positions_xy[mask, 0], positions_xy[mask, 1], s=25, color=col, alpha=0.75, label=f'Cluster {c+1}')
            ax.add_patch(plt.Circle((mod['centroids'][c, 0], mod['centroids'][c, 1]), mod['cluster_radii'][c],
                                    color=col, fill=False, linestyle='--', alpha=0.35, lw=1.2))

        ax.scatter(mod['centroids'][:, 0], mod['centroids'][:, 1], marker='X', s=120, color='black', zorder=6, label=r'Centroids $z_c$')
        ax.set_xlim(-540, 540)
        ax.set_ylim(-540, 540)
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_title(f'Spatial K-Means++ Deployment Map ($C = {target_c}$)')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        plt.tight_layout()

        fig_pdf = os.path.join(output_dir, f"{fname}.pdf")
        fig_png = os.path.join(output_dir, f"{fname}.png")
        fig.savefig(fig_pdf)
        fig.savefig(fig_png)
        plt.close(fig)
        print(f"  [✓] Saved {fname}: {fig_pdf}")

    return c_geo, results, models


if __name__ == "__main__":
    run_experiment_b0()
