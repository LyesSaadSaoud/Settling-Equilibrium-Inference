"""Reproducible robustness sweeps for the released Settling geometric diagnostic.

This script uses the same SettlingInferenceModule, 120-step budget, seeds,
collision rule, and central finite-difference control-point gradient as the
released implementation. To make the 1,200-run sweep practical, the five
interior central-difference pairs are evaluated in a vectorized batch. A
built-in equivalence check compares the vectorized gradient against the
original inference.cp_gradient before the sweep starts.

Examples
--------
If this file is inside the same code folder as environment.py/inference.py:
    python settling_robustness_sweep_reproducible.py

If the released code is elsewhere:
    python settling_robustness_sweep_reproducible.py --code-dir C:\\path\\to\\code

Outputs
-------
robustness_summary.csv / .json
robustness_combined.png / .pdf
robustness_context_jitter.png / .pdf
robustness_init_noise.png / .pdf
robustness_clearance.png / .pdf
"""
from pathlib import Path
import argparse
import csv
import json
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100, help='seeds per sweep point (default: 100)')
parser.add_argument('--code-dir', type=str, default=None, help='folder containing environment.py and inference.py')
parser.add_argument('--out-dir', type=str, default=None, help='output directory')
args = parser.parse_args()

HERE = Path(__file__).resolve().parent
if args.code_dir:
    CODE = Path(args.code_dir).resolve()
elif (HERE / 'environment.py').exists() and (HERE / 'inference.py').exists():
    CODE = HERE
elif (HERE / 'code' / 'environment.py').exists():
    CODE = HERE / 'code'
else:
    raise FileNotFoundError('Could not find environment.py/inference.py. Pass --code-dir PATH.')

OUT = Path(args.out_dir).resolve() if args.out_dir else HERE / 'robustness_results'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE))

import inference
from environment import random_context, is_collision_free, min_clearance, OBS_R
from inference import SettlingInferenceModule

# ---------------------------------------------------------------------------
# Vectorized implementation of the SAME central finite-difference gradient.
# CubicSpline is linear in the control-point values, so a spline basis lets us
# evaluate all +/- eps perturbations in one batch without changing the method.
# ---------------------------------------------------------------------------
_X_CP = inference._X_CP.copy()
_X_DENSE = inference._X_DENSE.copy()
_N_CP = len(_X_CP)
_BASIS = np.stack([
    CubicSpline(_X_CP, np.eye(_N_CP)[i], bc_type='clamped')(_X_DENSE)
    for i in range(_N_CP)
], axis=1)


def cp_gradient_vectorized(y_cp, obs_centres, lam_obs=80.0, eps=1e-4):
    interior = np.arange(1, len(y_cp) - 1)
    y0 = _BASIS @ y_cp
    ys = np.tile(y0, (2 * len(interior), 1))
    for q, i in enumerate(interior):
        ys[2*q] += eps * _BASIS[:, i]
        ys[2*q + 1] -= eps * _BASIS[:, i]

    energies = np.zeros(ys.shape[0], dtype=float)
    for cx, cy in obs_centres:
        dx = _X_DENSE - cx
        dy = ys - cy
        dist = np.sqrt(dx[None, :]**2 + dy**2 + 1e-8)
        viol = np.maximum(0.0, OBS_R - dist)
        energies += lam_obs * np.sum(viol**2, axis=1)

    grad = np.zeros(len(y_cp), dtype=float)
    for q, i in enumerate(interior):
        grad[i] = (energies[2*q] - energies[2*q + 1]) / (2 * eps)
    return grad


def verify_gradient_equivalence():
    original = inference.cp_gradient
    rng = np.random.default_rng(123)
    obs = np.array([[-0.4, 0.03], [0.4, -0.02]], dtype=float)
    max_err = 0.0
    for _ in range(3):
        y = np.r_[0.0, rng.normal(0.0, 0.3, 5), 0.0]
        g_ref = original(y, obs, 80.0)
        g_fast = cp_gradient_vectorized(y, obs, 80.0)
        max_err = max(max_err, float(np.max(np.abs(g_ref - g_fast))))
    if max_err > 1e-8:
        raise RuntimeError(f'Gradient-equivalence check failed: max error={max_err:.3e}')
    print(f'Gradient-equivalence check: PASS (max abs error {max_err:.3e})')
    inference.cp_gradient = cp_gradient_vectorized


N = int(args.n)
STEPS = 120
JITTERS = np.array([0.00, 0.04, 0.08, 0.12, 0.16, 0.20])
NOISES = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
BASE_JITTER = 0.08


def evaluate(jitter, noise_std, sweep_name):
    sbi = SettlingInferenceModule()
    vals = []
    for seed in range(N):
        obs = random_context(np.random.default_rng(seed), obs_jitter=float(jitter))
        result = sbi.settle(
            context=obs,
            n_steps=STEPS,
            noise_std=float(noise_std),
            rng=np.random.default_rng(seed + 30000),
        )
        traj = result['trajectory']
        vals.append((
            float(is_collision_free(traj, obs)),
            float(min_clearance(traj, obs)),
            float(abs(traj[len(traj)//2, 1]) > 0.12),
        ))
    a = np.asarray(vals, dtype=float)
    return {
        'sweep': sweep_name,
        'jitter': float(jitter),
        'noise_std': float(noise_std),
        'n': N,
        'success': float(a[:, 0].mean()),
        'clearance_mean': float(a[:, 1].mean()),
        'clearance_median': float(np.median(a[:, 1])),
        'clearance_q05': float(np.quantile(a[:, 1], 0.05)),
        'clearance_min': float(a[:, 1].min()),
        'commit_rate': float(a[:, 2].mean()),
    }


def save_plots(rows):
    context_rows = [r for r in rows if r['sweep'] == 'context_jitter']
    noise_rows = [r for r in rows if r['sweep'] == 'init_noise']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    ax1.plot([r['jitter'] for r in context_rows], [r['success'] for r in context_rows], marker='o', lw=2.2)
    ax1.axvline(0.08, ls=':', lw=1.2, label='Nominal jitter = 0.08')
    ax1.set_ylim(0.90, 1.01)
    ax1.set_xlabel('Obstacle vertical jitter range')
    ax1.set_ylabel('Collision-free success rate')
    ax1.set_title(f'Context robustness ({N} seeds per point)', fontweight='bold')
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, fontsize=8.5)
    for r in context_rows:
        ax1.annotate(f"{int(round(100*r['success']))}/{N}", (r['jitter'], r['success']),
                     xytext=(0, 7), textcoords='offset points', ha='center', fontsize=7.5)

    ax2.plot([r['noise_std'] for r in noise_rows], [r['success'] for r in noise_rows], marker='s', lw=2.2)
    ax2.axvline(0.30, ls=':', lw=1.2, label=r'Nominal $\sigma=0.30$')
    ax2.set_ylim(0.90, 1.01)
    ax2.set_xlabel(r'One-time initialization perturbation $\sigma$')
    ax2.set_ylabel('Collision-free success rate')
    ax2.set_title(f'Tie-break sensitivity ({N} seeds per point)', fontweight='bold')
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, fontsize=8.5)
    for r in noise_rows:
        ax2.annotate(f"{int(round(100*r['success']))}/{N}", (r['noise_std'], r['success']),
                     xytext=(0, 7), textcoords='offset points', ha='center', fontsize=7.5)

    fig.suptitle('Robustness of the Released Settling Diagnostic', fontweight='bold', fontsize=13.5)
    fig.savefig(OUT / 'robustness_combined.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'robustness_combined.pdf', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot([r['jitter'] for r in context_rows], [r['success'] for r in context_rows], marker='o', lw=2.2)
    ax.set_ylim(0.90, 1.01)
    ax.set_xlabel('Obstacle vertical jitter range')
    ax.set_ylabel('Collision-free success rate')
    ax.set_title(f'Robustness to context perturbation ({N} seeds per point)')
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT/'robustness_context_jitter.png', dpi=300); fig.savefig(OUT/'robustness_context_jitter.pdf'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot([r['noise_std'] for r in noise_rows], [r['success'] for r in noise_rows], marker='s', lw=2.2)
    ax.set_ylim(0.90, 1.01)
    ax.set_xlabel(r'One-time initialization perturbation $\sigma$')
    ax.set_ylabel('Collision-free success rate')
    ax.set_title(f'Sensitivity to symmetry-breaking perturbation ({N} seeds per point)')
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT/'robustness_init_noise.png', dpi=300); fig.savefig(OUT/'robustness_init_noise.pdf'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot([r['jitter'] for r in context_rows], [r['clearance_median'] for r in context_rows], marker='o', lw=2.2, label='Median')
    ax.plot([r['jitter'] for r in context_rows], [r['clearance_q05'] for r in context_rows], marker='^', lw=1.8, label='5th percentile')
    ax.axhline(0.02, ls='--', lw=1.1, label='Success threshold')
    ax.set_xlabel('Obstacle vertical jitter range')
    ax.set_ylabel('Minimum obstacle clearance')
    ax.set_title('Clearance robustness under context perturbation')
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT/'robustness_clearance.png', dpi=300); fig.savefig(OUT/'robustness_clearance.pdf'); plt.close(fig)


def main():
    verify_gradient_equivalence()
    rows = []
    t0 = time.time()
    for jitter in JITTERS:
        r = evaluate(jitter, 0.30, 'context_jitter')
        rows.append(r); print(r)
    for noise in NOISES:
        r = evaluate(BASE_JITTER, noise, 'init_noise')
        rows.append(r); print(r)

    with open(OUT / 'robustness_summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    with open(OUT / 'robustness_summary.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)

    save_plots(rows)
    print(f'Completed {len(rows)*N} trials in {time.time()-t0:.1f} s')
    print(f'Outputs: {OUT}')

if __name__ == '__main__':
    main()
