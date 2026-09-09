"""
figures.py
==========
All publication-quality figures for the Settling demonstration.

Fig 1 — Environment (two obstacles, valid corridors, forbidden centre)
Fig 2 — Conditional Mean Collapse (upper + lower + invalid mean)
Fig 3 — Settling Dynamics (bifurcation snapshots: k = 0, 10, 30, 60, final)
Fig 4 — Energy Landscape (2-D heatmap, basins, saddle, gradient field)
Fig 5 — Phase Transition (ambiguity sweep: mean collapses, settling bifurcates)
Fig 6 — Quantitative Comparison Table
Fig 7 — All Methods (2×2 comparison, the paper's central figure)
Fig 8 — Energy Convergence (Settling vs Diffusion vs EBM)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from environment import (OBS_CENTRES, OBS_RADIUS, OBS_R, START, GOAL, COLORS,
                          upper_trajectory, lower_trajectory, mean_trajectory,
                          is_collision_free, obstacle_sdf, SEQ_LEN,
                          trajectory_smoothness, min_clearance, _bump)
from energy import obstacle_energy, smooth_energy, total_energy
from inference import (attention_inference, diffusion_inference, ebm_inference,
                       SettlingInferenceModule)

FIGS_DIR = Path("figures")
FIGS_DIR.mkdir(exist_ok=True)

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   10,
    "legend.fontsize":  8.5,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       140,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_obstacles(ax, alpha_core=0.92, alpha_margin=0.55):
    """Draw both circular obstacles (core + safety margin)."""
    for cx, cy in OBS_CENTRES:
        margin_patch = plt.Circle((cx, cy), OBS_R,
                                   color=COLORS['margin'], alpha=alpha_margin,
                                   zorder=2, label='Safety margin')
        core_patch   = plt.Circle((cx, cy), OBS_RADIUS,
                                   color=COLORS['core'],   alpha=alpha_core,
                                   zorder=3)
        ax.add_patch(margin_patch)
        ax.add_patch(core_patch)

    ax.scatter(*START, s=110, color=COLORS['settling'], zorder=7,
               marker='o', edgecolors='k', linewidths=0.8)
    ax.scatter(*GOAL,  s=110, color='#E6550D',           zorder=7,
               marker='*', edgecolors='k', linewidths=0.8)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, color='#BBBBBB', linewidth=0.5)
    ax.set_xlabel('x  (inference state)')
    ax.set_ylabel('y  (inference state)')


# ── Fig 1: Environment ────────────────────────────────────────────────────────

def fig1_environment():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _draw_obstacles(ax)

    upper = upper_trajectory()
    lower = lower_trajectory()

    # Corridor arrows + shading
    xs = np.linspace(-1, 1, 300)
    y_up  = _bump(xs, -0.4, 0.38, 0.72) + _bump(xs, 0.4, 0.38, 0.72)
    y_low = -y_up
    ax.fill_between(xs, y_up + 0.18, 1.0,  alpha=0.08, color=COLORS['upper'])
    ax.fill_between(xs, -1.0, y_low - 0.18, alpha=0.08, color=COLORS['lower'])

    ax.plot(upper[:, 0], upper[:, 1], color=COLORS['upper'], lw=2.2,
            label='Valid Mode A (upper)', zorder=5)
    ax.plot(lower[:, 0], lower[:, 1], color=COLORS['lower'], lw=2.2,
            label='Valid Mode B (lower)', zorder=5)

    # Central forbidden corridor
    ax.axhspan(-OBS_R, OBS_R, alpha=0.07, color='red', zorder=1)
    ax.axhline(0, color='#CC0000', lw=1.0, ls=':', alpha=0.6, zorder=4)

    # Labels
    ax.text(-0.4, 0.0, 'Obs 1', ha='center', va='center',
            color='white', fontsize=8, fontweight='bold', zorder=6)
    ax.text( 0.4, 0.0, 'Obs 2', ha='center', va='center',
            color='white', fontsize=8, fontweight='bold', zorder=6)
    ax.text(0.0,  0.85, 'Valid Mode A', ha='center', color=COLORS['upper'],
            fontsize=9, fontstyle='italic', fontweight='bold')
    ax.text(0.0, -0.85, 'Valid Mode B', ha='center', color=COLORS['lower'],
            fontsize=9, fontstyle='italic', fontweight='bold')
    ax.text(0.82, 0.05, 'Invalid\nMean Region', ha='center', va='bottom',
            color='#CC0000', fontsize=7.5, alpha=0.75)

    ax.text(-1.18, 0.0, 'Start', ha='center', fontsize=8, color=COLORS['settling'])
    ax.text( 1.18, 0.0, 'Goal',  ha='center', fontsize=8, color='#E6550D')

    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='0.8')
    ax.set_title('Controlled Non-Convex Validity Environment\n'
                 'Two designated valid trajectory modes and an invalid mean region',
                 fontweight='bold')
    fig.tight_layout()
    _save(fig, 'fig1_environment')


# ── Fig 2: Conditional Mean Collapse ─────────────────────────────────────────

def fig2_mean_collapse():
    n  = SEQ_LEN
    u  = upper_trajectory(n)
    l  = lower_trajectory(n)
    m  = attention_inference()         # = 0.5*(u+l) = y≈0 = INVALID

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8),
                                   gridspec_kw={'width_ratios': [2.2, 1]})

    _draw_obstacles(ax)

    ax.plot(u[:, 0], u[:, 1], color=COLORS['upper'], lw=2.8,
            label='Mode A — upper (valid)', zorder=5)
    ax.plot(l[:, 0], l[:, 1], color=COLORS['lower'], lw=2.8,
            label='Mode B — lower (valid)', zorder=5)
    ax.plot(m[:, 0], m[:, 1], color=COLORS['attention'], lw=2.8, ls='--',
            label=r'Attention mean $\bar{y} = \frac{1}{2}(y_A+y_B)$  [INVALID]',
            zorder=6)

    # Collision markers at obstacle centres
    for cx, cy in OBS_CENTRES:
        ax.scatter([cx], [cy], s=220, color=COLORS['attention'],
                   marker='X', zorder=8, edgecolors='k', lw=0.8)

    # Annotation
    ax.annotate('Collision', xy=(-0.4, 0.0), xytext=(-0.55, 0.4),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                fontsize=9, color='black')
    ax.annotate('Collision', xy=(0.4, 0.0), xytext=(0.55, 0.4),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                fontsize=9, color='black')

    ax.legend(loc='upper left', frameon=True, framealpha=0.92)
    ax.set_title('Conditional Mean Collapse\n'
                 r'$f^*(c)=\mathbb{E}[y|c]=\bar{y}\;\notin\;\mathcal{V}(c)$',
                 fontweight='bold')

    # Right: y-profile at x = 0 (obstacle band)
    ax2.axhline( 0.72, color=COLORS['upper'],     lw=2.5, label='Mode A')
    ax2.axhline(-0.72, color=COLORS['lower'],     lw=2.5, label='Mode B')
    ax2.axhline(  0.0, color=COLORS['attention'], lw=2.5, ls='--',
                label='Mean (invalid)')
    ax2.axhspan(-OBS_R, OBS_R, alpha=0.20, color=COLORS['core'], label='Obstacle zone')
    ax2.set_xlim(-0.5, 1.5); ax2.set_ylim(-1.05, 1.05)
    ax2.set_xticks([]); ax2.set_ylabel('y at obstacle midplane (x = 0)')
    ax2.set_title('y-profile at x = 0\n(symmetric cross-section)', fontweight='bold')
    ax2.legend(loc='upper right', frameon=False)
    ax2.set_aspect('auto')

    fig.suptitle('Conditional Mean Collapse in a Non-Convex Validity Space',
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    _save(fig, 'fig2_mean_collapse')


# ── Fig 3: Settling Dynamics (bifurcation snapshots) ─────────────────────────

def fig3_settling_dynamics(settling_result):
    history = settling_result['history']
    snap_targets = [0, 10, 30, 60, history[-1][0]]
    snaps = [min(history, key=lambda h: abs(h[0] - t)) for t in snap_targets]

    fig, axes = plt.subplots(1, len(snaps), figsize=(15, 3.8), sharey=True)
    cmap = plt.cm.viridis

    upper = upper_trajectory()
    lower = lower_trajectory()

    for idx, ((k, traj), ax) in enumerate(zip(snaps, axes)):
        _draw_obstacles(ax)
        # Ghost valid modes
        ax.plot(upper[:, 0], upper[:, 1], color=COLORS['upper'], lw=1.0,
                alpha=0.28, ls='--')
        ax.plot(lower[:, 0], lower[:, 1], color=COLORS['lower'], lw=1.0,
                alpha=0.28, ls='--')
        col = cmap(0.15 + 0.7 * idx / (len(snaps)-1))
        ax.plot(traj[:, 0], traj[:, 1], color=col, lw=2.6, zorder=6)

        valid_str = 'valid' if is_collision_free(traj) else 'invalid'
        ax.set_title(f'k = {k}\n({valid_str})', fontweight='bold',
                     color='#1A9850' if valid_str == 'valid' else '#CC0000')
        ax.set_xlabel('x')
        if idx == 0:
            ax.set_ylabel('y')
        ax.set_xticks([-1, 0, 1])

    # Bifurcation annotation on first panel
    axes[0].annotate('Proposal\n(saddle)', xy=(0.0, 0.04), xytext=(-0.6, 0.45),
                     arrowprops=dict(arrowstyle='->', lw=1.2, color='grey'),
                     fontsize=8, color='grey')

    fig.suptitle('Settling Dynamics: Bifurcation and Equilibrium Convergence\n'
                 r'$s^{(k+1)} = s^{(k)} - \eta\,\nabla_s E(s^{(k)}, c)$   '
                 '(initial proposal = invalid conditional mean)',
                 fontweight='bold')
    fig.tight_layout()
    _save(fig, 'fig3_settling_dynamics')


# ── Fig 4: Energy Landscape ───────────────────────────────────────────────────

def fig4_energy_landscape():
    res = 220
    xs = np.linspace(-1.25, 1.25, res)
    ys = np.linspace(-1.05, 1.05, res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    # Obstacle SDF
    sdf = obstacle_sdf(pts).reshape(res, res)

    # Obstacle energy on 2-D grid (evaluate each point as a 1-pt trajectory)
    E_obs = np.zeros((res, res))
    from energy import OBS_R
    for cx, cy in OBS_CENTRES:
        dx = XX - cx; dy = YY - cy
        dist = np.sqrt(dx**2 + dy**2 + 1e-8)
        viol = np.maximum(0.0, OBS_R - dist)
        E_obs += 80.0 * viol**2

    # Goal attraction
    E_goal = 2.0 * ((XX - GOAL[0])**2 + (YY - GOAL[1])**2)
    E_total_grid = E_obs + E_goal

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: SDF with gradient flow
    cmap_sdf = LinearSegmentedColormap.from_list(
        'sdf', ['#D73027','#FEE090','#4575B4'])
    im1 = axes[0].contourf(XX, YY, np.clip(sdf, -0.5, 0.8),
                            levels=60, cmap=cmap_sdf)
    axes[0].contour(XX, YY, sdf, levels=[0],
                    colors='k', linewidths=2.0, linestyles='--')

    # Gradient flow arrows
    step = 16
    Gx = np.gradient(E_obs, axis=1)
    Gy = np.gradient(E_obs, axis=0)
    axes[0].quiver(XX[::step, ::step], YY[::step, ::step],
                   -Gx[::step, ::step], -Gy[::step, ::step],
                   alpha=0.45, color='white', scale=80, width=0.003)

    axes[0].plot(*START, 'o', ms=8, color=COLORS['settling'], zorder=6)
    axes[0].plot(*GOAL,  '*', ms=10, color='#E6550D', zorder=6)
    axes[0].axhline(0, color='gold', lw=1.5, ls=':', label='Saddle locus y=0')
    plt.colorbar(im1, ax=axes[0], label='SDF (neg=inside obstacle)')
    axes[0].set_xlim(-1.25, 1.25); axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_aspect('equal')
    axes[0].set_title('Obstacle Signed Distance Field\n+ Gradient Flow',
                      fontweight='bold')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    axes[0].legend(fontsize=8)

    # Right: composite energy — two basins + central saddle
    cmap_e = LinearSegmentedColormap.from_list(
        'energy', ['#1A9850','#FFFFBF','#D73027'])
    im2 = axes[1].contourf(XX, YY, np.log1p(E_total_grid),
                            levels=60, cmap=cmap_e)
    axes[1].contour(XX, YY, np.log1p(E_total_grid),
                    levels=12, colors='k', linewidths=0.3, alpha=0.3)
    plt.colorbar(im2, ax=axes[1], label='log(1 + E_total)')

    # Basin and saddle labels
    axes[1].text( 0.0,  0.75, 'Upper Basin\n(Valid Mode A)',
                 ha='center', color='white', fontsize=9, fontweight='bold', zorder=7)
    axes[1].text( 0.0, -0.75, 'Lower Basin\n(Valid Mode B)',
                 ha='center', color='white', fontsize=9, fontweight='bold', zorder=7)
    axes[1].text( 0.0,  0.0, 'Saddle\n(unstable mean)',
                 ha='center', va='center', color='gold', fontsize=9,
                 fontweight='bold', zorder=7)

    axes[1].set_xlim(-1.25, 1.25); axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_aspect('equal')
    axes[1].set_title('Composite Energy Landscape\n'
                      'Two valid basins separated by an unstable saddle',
                      fontweight='bold')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')

    fig.suptitle('Reduced Energy Landscape: Non-Convex Validity Structure',
                 fontweight='bold')
    fig.tight_layout()
    _save(fig, 'fig4_energy_landscape')


# ── Fig 5: Phase Transition ───────────────────────────────────────────────────

def fig5_phase_transition():
    """
    Sweep ambiguity parameter α = obstacle y-offset (0 = symmetric, high = separated).
    α=0: both obstacles on centreline → maximum ambiguity.
    α=0.5: obstacles off-centre → less ambiguous (one mode is clearly easier).
    """
    rng = np.random.default_rng(17)
    sbi = SettlingInferenceModule()
    n_alpha = 30
    alphas = np.linspace(0.0, 0.5, n_alpha)

    mean_col, settle_col, settle_commit = [], [], []

    for alpha in alphas:
        # Perturb obstacles symmetrically by alpha (increasing separation)
        obs = OBS_CENTRES.copy()
        obs[:, 1] = alpha        # both obstacles move to y=+alpha

        x = np.linspace(-1.0, 1.0, SEQ_LEN)

        # Upper mode (always above obstacles)
        y_upper = (_bump(x, -0.4, 0.38, 0.72 + alpha) +
                   _bump(x,  0.4, 0.38, 0.72 + alpha))
        upper = np.column_stack([x, y_upper])

        y_lower = -(_bump(x, -0.4, 0.38, 0.72 - alpha) +
                    _bump(x,  0.4, 0.38, 0.72 - alpha))
        lower = np.column_stack([x, y_lower])

        mean_traj = 0.5 * (upper + lower)

        # Mean collision rate
        mean_col.append(0.0 if is_collision_free(mean_traj, obs) else 1.0)

        # Settling
        result = sbi.settle(context=obs, n_steps=100, rng=rng)
        settle_col.append(0.0 if is_collision_free(result['trajectory'], obs) else 1.0)

        # Commitment: did settling commit to one corridor?
        mid_y = result['trajectory'][SEQ_LEN // 2, 1]
        settle_commit.append(float(abs(mid_y) > 0.15))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.plot(alphas, mean_col, color=COLORS['attention'], lw=2.5, marker='o', ms=4,
             label='Attention — collision rate')
    ax1.plot(alphas, settle_col, color=COLORS['settling'], lw=2.5, marker='s', ms=4,
             label='Settling — collision rate')
    ax1.axvline(0.0, color='grey', ls=':', lw=1.5, label='Max ambiguity (α=0)')
    ax1.set_xlabel('Ambiguity α  (obstacle y-offset)')
    ax1.set_ylabel('Collision rate')
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_title('Collision Rate vs. Ambiguity α\n'
                  '(Mean collapses; Settling avoids)', fontweight='bold')
    ax1.legend(frameon=False)

    ax2.plot(alphas, settle_commit, color=COLORS['settling'], lw=2.5, marker='D', ms=4,
             label='Settling — committed')
    ax2.axhline(0, color=COLORS['attention'], lw=2.0, ls='--',
                label='Attention — never commits')
    ax2.axvline(0.12, color='#CC0000', ls=':', lw=1.5, alpha=0.7,
                label='Illustrative regime boundary')
    ax2.fill_betweenx([0, 1.15], 0, 0.12, alpha=0.06, color='#CC0000')
    ax2.set_xlabel('Ambiguity α')
    ax2.set_ylabel('Commitment rate')
    ax2.set_ylim(-0.1, 1.2)
    ax2.set_title('Phase Transition: Symmetry Breaking\n'
                  '(Sharp bifurcation vs. continuous collapse)', fontweight='bold')
    ax2.legend(frameon=False)

    fig.suptitle('Controlled Ambiguity Sweep: Settling vs. Mean Collapse',
                 fontweight='bold')
    fig.tight_layout()
    _save(fig, 'fig5_phase_transition')


# ── Fig 6: Quantitative Comparison Table ─────────────────────────────────────

def fig6_comparison_table(metrics: dict):
    """metrics: output of evaluation.run_quantitative_eval()"""
    methods = ['Attention', 'Diffusion', 'EBM', 'Settling (Ours)']
    keys    = ['success_rate', 'collision_rate', 'smoothness', 'clearance',
               'deterministic', 'mode_commit']
    headers = ['Method', 'Success ↑', 'Collision ↓', 'Smoothness ↓',
               'Min Clear ↑', 'Deterministic', 'Mode Commit']

    rows = []
    for m in methods:
        mk = m.lower().split()[0]
        d  = metrics[mk]
        rows.append([
            m,
            f"{d['success_rate']:.0%}",
            f"{d['collision_rate']:.0%}",
            f"{d['smoothness']:.5f}",
            f"{d['clearance']:.3f}",
            d['deterministic'],
            f"{d['mode_commit']:.0%}",
        ])

    fig, ax = plt.subplots(figsize=(13, 3.0))
    ax.axis('off')
    col_widths = [0.20, 0.11, 0.11, 0.14, 0.12, 0.16, 0.14]
    tbl = ax.table(cellText=rows, colLabels=headers,
                   cellLoc='center', loc='center', colWidths=col_widths)
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.0, 2.1)

    for j in range(len(headers)):
        tbl[0, j].set_facecolor('#2C3E50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    row_colours = ['#FCE4EC', '#FFF3E0', '#EDE7F6', '#E8F5E9']
    for i, rc in enumerate(row_colours, start=1):
        for j in range(len(headers)):
            tbl[i, j].set_facecolor(rc)

    ax.set_title('Figure 6 — Quantitative Comparison (100-seed evaluation)',
                 fontweight='bold', pad=20, fontsize=12)
    fig.tight_layout()
    _save(fig, 'fig6_comparison_table')


# ── Fig 7: All Methods (the paper's central figure) ──────────────────────────

def fig7_all_methods(settling_result, diffusion_result, ebm_result):
    n = SEQ_LEN
    upper = upper_trajectory(n)
    lower = lower_trajectory(n)
    t_att = attention_inference()
    t_sbi = settling_result['trajectory']
    t_diff = diffusion_result['trajectory']
    t_ebm  = ebm_result['trajectory']
    prop   = settling_result['proposal']

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5),
                             sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.34, wspace=0.18)

    panels = [
        (axs[0, 0], 'Attention / Mean-Seeking',   t_att,  COLORS['attention'],
         'Mean Collapse',    'mean-seeking, single-step'),
        (axs[0, 1], 'Diffusion / Stochastic',     t_diff, COLORS['diffusion'],
         'Noisy / Unstable', 'stochastic, non-deterministic'),
        (axs[1, 0], 'EBM / Direct Optimization',  t_ebm,  COLORS['ebm'],
         'Stagnation',       'random init, no proposal'),
        (axs[1, 1], 'Settling (Ours)',             t_sbi,  COLORS['settling'],
         'Stable Equilibrium','deterministic, equilibrium-based'),
    ]

    for ax, title, traj, col, label, sub in panels:
        _draw_obstacles(ax)
        # Ghost valid modes
        ax.plot(upper[:, 0], upper[:, 1], color=COLORS['upper'], lw=1.1,
                alpha=0.30, ls='--', zorder=4)
        ax.plot(lower[:, 0], lower[:, 1], color=COLORS['lower'], lw=1.1,
                alpha=0.30, ls='--', zorder=4)

        if ax is axs[1, 1]:   # Settling: also show proposal
            ax.plot(prop[:, 0], prop[:, 1], color=COLORS['proposal'], lw=1.5,
                    ls=':', zorder=4, alpha=0.7, label='Proposal π_φ(c)')
            ax.legend(loc='upper right', fontsize=7.5, frameon=True, framealpha=0.9)

        ax.plot(traj[:, 0], traj[:, 1], color=col, lw=2.8, zorder=6)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.text(0.0, -0.88, label, ha='center', va='center', color=col,
                fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.88, edgecolor='none', pad=1.5),
                zorder=8)
        ax.text(-1.18, -0.88, sub, ha='left', va='center', color='#555555',
                fontsize=7.5, fontstyle='italic')

        valid = is_collision_free(traj)
        ax.text(1.18, 0.92, 'VALID' if valid else 'INVALID',
                ha='right', va='top', color='#1A9850' if valid else '#CC0000',
                fontsize=8, fontweight='bold', zorder=9,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.0))

    fig.suptitle('Structural Comparison of Inference Paradigms\n'
                 'Operator-level analysis in a non-convex validity space',
                 fontsize=13, fontweight='bold', y=0.98)
    _save(fig, 'fig7_all_methods')


# ── Fig 8: Energy Convergence ─────────────────────────────────────────────────

def fig8_energy_convergence(settling_result, diffusion_result, ebm_result):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    def _plot_curve(result, key, color, label):
        if 'energy_curve' not in result:
            return
        ks, es = zip(*result['energy_curve'])
        ax.semilogy(ks, es, color=color, lw=2.5, label=label)

    _plot_curve(settling_result,  'energy_curve', COLORS['settling'],
                'Settling (numerical trace)')
    _plot_curve(diffusion_result, 'energy_curve', COLORS['diffusion'],
                'Stochastic denoising')
    _plot_curve(ebm_result,       'energy_curve', COLORS['ebm'],
                'Direct energy descent')

    ax.set_xlabel('Iteration k')
    ax.set_ylabel('E_total  (log scale)')
    ax.set_title('Numerical Refinement Diagnostics\n'
                 'Method-specific energy-like traces from the supplied implementation',
                 fontweight='bold')
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, 'fig8_energy_convergence')


# ── Save helper ───────────────────────────────────────────────────────────────

def _save(fig, name: str):
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS_DIR / f'{name}.{ext}')
    plt.close(fig)
    print(f"  ✓  {name}")
