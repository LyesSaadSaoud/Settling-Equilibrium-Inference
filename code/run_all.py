"""
run_all.py
==========
Single command entry point for the Settling demonstration.

Framing: Operator-level analysis of inference geometry in disconnected validity spaces.
         This is NOT robotics / RL / path planning. It is a controlled conceptual
         demonstration of Conditional Mean Collapse and equilibrium-based resolution.

Usage:
    python run_all.py [--quick]   # --quick reduces eval seeds for fast testing

Outputs:
    figures/fig{1-8}_{name}.{pdf,png}
    animations/settling_animation.{gif,mp4}
"""

import sys
import time
import numpy as np

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


def banner(msg: str, width: int = 62):
    print(f"\n{'─'*width}")
    print(f"  {msg}")
    print(f"{'─'*width}")


def main():
    quick = '--quick' in sys.argv
    t0 = time.time()

    # ── Step 1: Verify environment ────────────────────────────────────────
    banner("Step 1 / 5 — Environment verification")
    from environment import (upper_trajectory, lower_trajectory,
                              mean_trajectory, is_collision_free, OBS_CENTRES)

    n = 40
    upper = upper_trajectory(n)
    lower = lower_trajectory(n)
    mean  = mean_trajectory(n)

    u_valid = is_collision_free(upper)
    l_valid = is_collision_free(lower)
    m_valid = is_collision_free(mean)

    print(f"  Mode A (upper)    — collision-free: {u_valid}")
    print(f"  Mode B (lower)    — collision-free: {l_valid}")
    print(f"  Conditional mean  — collision-free: {m_valid}   ← COLLAPSE (Theorem 1)")

    assert u_valid,  "ERROR: upper trajectory should be valid!"
    assert l_valid,  "ERROR: lower trajectory should be valid!"
    assert not m_valid, "ERROR: mean should be INVALID!"
    print("  Environment verified. ✓")

    # ── Step 2: Run all four inference paradigms ──────────────────────────
    banner("Step 2 / 5 — Running four inference paradigms")
    from inference import (attention_inference, diffusion_inference,
                           ebm_inference, SettlingInferenceModule)

    rng = np.random.default_rng(SEED)
    sbi = SettlingInferenceModule()

    # Attention (single-step, analytical)
    t_att = attention_inference(OBS_CENTRES)
    print(f"  Attention   — valid: {is_collision_free(t_att)}"
          f"   (midpoint y = {t_att[n//2, 1]:+.3f})")

    # Settling (full history for Fig 3 + animation)
    print("  Settling    — running dynamics ...")
    sbi_full = sbi.settle_with_snapshots(
        context=OBS_CENTRES, n_steps=120, lr=0.006,
        noise_std=0.025, rng=np.random.default_rng(SEED))
    t_sbi = sbi_full['trajectory']
    print(f"  Settling    — valid: {is_collision_free(t_sbi)}"
          f"   (midpoint y = {t_sbi[n//2, 1]:+.3f})")

    # Diffusion
    print("  Diffusion   — running ...")
    diff_result = diffusion_inference(
        OBS_CENTRES, n_steps=120, rng=np.random.default_rng(SEED + 1))
    t_diff = diff_result['trajectory']
    print(f"  Diffusion   — valid: {is_collision_free(t_diff)}"
          f"   (midpoint y = {t_diff[n//2, 1]:+.3f})")

    # EBM
    print("  EBM         — running ...")
    ebm_result = ebm_inference(
        OBS_CENTRES, n_steps=200, rng=np.random.default_rng(SEED + 2))
    t_ebm = ebm_result['trajectory']
    print(f"  EBM         — valid: {is_collision_free(t_ebm)}"
          f"   (midpoint y = {t_ebm[n//2, 1]:+.3f})")

    # ── Step 3: Figures ───────────────────────────────────────────────────
    banner("Step 3 / 5 — Generating publication-quality figures")
    from figures import (fig1_environment, fig2_mean_collapse,
                          fig3_settling_dynamics, fig4_energy_landscape,
                          fig5_phase_transition, fig6_comparison_table,
                          fig7_all_methods, fig8_energy_convergence)

    # Run 30-seed eval for table (full 100 below)
    quick_metrics = _quick_metrics()

    fig1_environment()
    fig2_mean_collapse()
    fig3_settling_dynamics(sbi_full)
    fig4_energy_landscape()
    fig5_phase_transition()
    fig6_comparison_table(quick_metrics)
    fig7_all_methods(sbi_full, diff_result, ebm_result)
    fig8_energy_convergence(sbi_full, diff_result, ebm_result)

    # ── Step 4: Animations ────────────────────────────────────────────────
    banner("Step 4 / 5 — Generating animations")
    from animation import make_gif, make_mp4
    make_gif(sbi_full)
    make_mp4(sbi_full)

    # ── Step 5: Full quantitative evaluation ─────────────────────────────
    n_seeds = 20 if quick else 100
    banner(f"Step 5 / 5 — Quantitative evaluation ({n_seeds} seeds)")
    from evaluation import run_quantitative_eval
    metrics = run_quantitative_eval(n_seeds)

    # Re-generate table with full metrics
    fig6_comparison_table(metrics)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner(f"Done in {elapsed:.1f}s")
    print(f"  Method            | Valid  | midpoint-y  | notes")
    print(f"  ──────────────────┼────────┼─────────────┼────────────────────────────────")
    for name, traj, note in [
        ('Attention',       t_att,  'Conditional Mean Collapse (structural)'),
        ('Diffusion',       t_diff, 'Stochastic, non-deterministic'),
        ('EBM',             t_ebm,  'Stagnation / local trapping'),
        ('Settling (ours)', t_sbi,  'Deterministic equilibrium selection'),
    ]:
        v = is_collision_free(traj)
        y = traj[n//2, 1]
        sym = '✓' if v else '✗'
        print(f"  {name:<18}| {sym}      | {y:+.4f}      | {note}")

    print(f"\n  Figures    → figures/")
    print(f"  Animations → animations/\n")


def _quick_metrics() -> dict:
    """30-seed eval for intermediate figure generation."""
    from evaluation import run_quantitative_eval
    return run_quantitative_eval(n_seeds=30)


if __name__ == "__main__":
    main()
