"""
evaluation.py
=============
Quantitative comparison of all four inference paradigms.

Metrics (over 100 randomised seeds with obstacle jitter):
  - success_rate      : fraction of trials with collision-free trajectories
  - collision_rate    : fraction colliding with at least one obstacle
  - smoothness        : mean squared second differences (lower = smoother)
  - clearance         : minimum clearance from any obstacle (higher = safer)
  - deterministic     : Yes / No
  - mode_commit       : fraction of trials that commit to a single corridor
"""

import numpy as np
from environment import (OBS_CENTRES, upper_trajectory, lower_trajectory,
                          mean_trajectory, is_collision_free, SEQ_LEN,
                          trajectory_smoothness, min_clearance, random_context)
from inference import (attention_inference, diffusion_inference, ebm_inference,
                       SettlingInferenceModule)


N_SEEDS = 100
OBS_JITTER = 0.08   # small random perturbation to obstacle positions


def run_quantitative_eval(n_seeds: int = N_SEEDS) -> dict:
    """
    Run all four inference paradigms under n_seeds randomised conditions.
    Returns metrics dict suitable for fig6_comparison_table().
    """
    print(f"\n  Running quantitative evaluation ({n_seeds} seeds)...")
    rng = np.random.default_rng(2024)
    sbi = SettlingInferenceModule()

    records = {m: [] for m in ['attention', 'diffusion', 'ebm', 'settling']}

    for seed in range(n_seeds):
        if seed % 20 == 0:
            print(f"    seed {seed}/{n_seeds}")

        seed_rng = np.random.default_rng(seed)
        obs = random_context(seed_rng, obs_jitter=OBS_JITTER)

        # ── Attention ───────────────────────────────────────────────────
        t_att = attention_inference(obs)
        records['attention'].append({
            'traj':  t_att,
            'valid': is_collision_free(t_att, obs),
        })

        # ── Diffusion ───────────────────────────────────────────────────
        d_rng = np.random.default_rng(seed + 10000)
        d_res = diffusion_inference(obs, n_steps=100, rng=d_rng)
        t_diff = d_res['trajectory']
        records['diffusion'].append({
            'traj':  t_diff,
            'valid': is_collision_free(t_diff, obs),
        })

        # ── EBM ─────────────────────────────────────────────────────────
        e_rng = np.random.default_rng(seed + 20000)
        e_res = ebm_inference(obs, n_steps=180, rng=e_rng)
        t_ebm = e_res['trajectory']
        records['ebm'].append({
            'traj':  t_ebm,
            'valid': is_collision_free(t_ebm, obs),
        })

        # ── Settling ─────────────────────────────────────────────────────
        s_rng = np.random.default_rng(seed + 30000)
        s_res = sbi.settle(context=obs, n_steps=120, rng=s_rng)
        t_sbi = s_res['trajectory']
        records['settling'].append({
            'traj':  t_sbi,
            'valid': is_collision_free(t_sbi, obs),
        })

    # ── Aggregate metrics ────────────────────────────────────────────────
    def _mode_commit(traj):
        """True if trajectory midpoint clearly commits to one corridor."""
        mid_y = traj[SEQ_LEN // 2, 1]
        return abs(mid_y) > 0.12

    results = {}
    for method, recs in records.items():
        trajs  = [r['traj']  for r in recs]
        valids = [r['valid'] for r in recs]
        results[method] = {
            'success_rate':   float(np.mean(valids)),
            'collision_rate': float(1.0 - np.mean(valids)),
            'smoothness':     float(np.mean([trajectory_smoothness(t) for t in trajs])),
            'clearance':      float(np.mean([min_clearance(t) for t in trajs])),
            'deterministic':  'Yes' if method in ('attention', 'settling') else 'No',
            'mode_commit':    float(np.mean([_mode_commit(t) for t in trajs])),
        }

    _print_table(results)
    return results


def _print_table(results: dict):
    print("\n  ╔══════════════════╦══════════╦══════════╦══════════════╦════════════╦═══════════╗")
    print("  ║ Method           ║ Success  ║ Collision║ Smoothness   ║ Min Clear  ║ Mode Commit║")
    print("  ╠══════════════════╬══════════╬══════════╬══════════════╬════════════╬═══════════╣")
    for m, d in results.items():
        print(f"  ║ {m:<16} ║ {d['success_rate']:>7.0%}  ║ {d['collision_rate']:>7.0%}  "
              f"║ {d['smoothness']:>11.5f}  ║ {d['clearance']:>9.3f}   ║ {d['mode_commit']:>8.0%}   ║")
    print("  ╚══════════════════╩══════════╩══════════╩══════════════╩════════════╩═══════════╝\n")
