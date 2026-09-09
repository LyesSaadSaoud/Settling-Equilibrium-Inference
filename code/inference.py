"""
inference.py
============
Four inference paradigms — using control-point spline parameterization
for SettlingInferenceModule to guarantee smooth, zigzag-free trajectories.

  A. Attention / Mean-Seeking  — convex aggregation → Conditional Mean Collapse
  B. Diffusion / Stochastic    — noisy denoising → inconsistent
  C. EBM / Direct Optimization — gradient descent from random init → stagnation
  D. SettlingInferenceModule   — equilibrium selection, smooth by construction
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from environment import (SEQ_LEN, START, GOAL, OBS_CENTRES, OBS_R,
                          upper_trajectory, lower_trajectory, mean_trajectory,
                          OBS_RADIUS, SAFETY_MARGIN, _bump)
from energy import obstacle_energy, obstacle_gradient, endpoint_energy, endpoint_gradient


# ── Control-point spline helpers ─────────────────────────────────────────────

_X_CP = np.array([-1.0, -0.65, -0.4, 0.0, 0.4, 0.65, 1.0])  # 7 control points
_X_DENSE = np.linspace(-1.0, 1.0, SEQ_LEN)


def traj_from_cp(y_cp: np.ndarray,
                 x_cp: np.ndarray = _X_CP,
                 n_dense: int = SEQ_LEN) -> np.ndarray:
    """
    Cubic spline from y-control-points → dense smooth trajectory.
    C2-smooth by construction: eliminates zigzagging at the parameterisation level.
    """
    x_dense = np.linspace(x_cp[0], x_cp[-1], n_dense)
    cs = CubicSpline(x_cp, y_cp, bc_type='clamped')
    y_dense = cs(x_dense)
    return np.column_stack([x_dense, y_dense])


def cp_gradient(y_cp: np.ndarray,
                obs_centres: np.ndarray,
                lam_obs: float = 80.0,
                eps: float = 1e-4) -> np.ndarray:
    """
    Numerical gradient of E_obstacle w.r.t. interior control-point y-values.
    Only interior points (indices 1..-1) are optimised; endpoints are pinned.
    Cost: 2*(N_CP-2) energy evaluations per step — very cheap.
    """
    n = len(y_cp)
    grad = np.zeros(n)
    for i in range(1, n - 1):       # skip fixed endpoints
        yp = y_cp.copy(); yp[i] += eps
        ym = y_cp.copy(); ym[i] -= eps
        Ep = obstacle_energy(traj_from_cp(yp), obs_centres, lam_obs)
        Em = obstacle_energy(traj_from_cp(ym), obs_centres, lam_obs)
        grad[i] = (Ep - Em) / (2 * eps)
    return grad


# ── A. Attention / Mean-Seeking ──────────────────────────────────────────────

def attention_inference(context: np.ndarray = OBS_CENTRES,
                        n: int = SEQ_LEN) -> np.ndarray:
    """
    Expectation-based (attention-like) inference.

    Under MSE training with symmetric bimodal data, the Bayes-optimal
    predictor is the conditional mean (Theorem 1):

        f*(c) = E[y|c] = 0.5·y_A + 0.5·y_B = 0  (symmetric context)

    Output: y ≡ 0 → passes through both obstacle centres → INVALID.
    This is a provably structural failure, not a capacity issue.
    """
    u = upper_trajectory(n)
    l = lower_trajectory(n)
    return 0.5 * (u + l)          # arithmetic mean → y = 0 everywhere


# ── B. Diffusion / Stochastic Baseline ───────────────────────────────────────

def diffusion_inference(context: np.ndarray = OBS_CENTRES,
                        n: int = SEQ_LEN,
                        n_steps: int = 120,
                        noise_init: float = 0.9,
                        noise_decay: float = 0.94,
                        step_size: float = 0.04,
                        rng: np.random.Generator = None) -> dict:
    """
    Stochastic denoising baseline. Noise persists throughout inference —
    the defining feature that distinguishes diffusion from Settling.
    Shows: stochasticity, non-determinism, inconsistent convergence.
    """
    if rng is None:
        rng = np.random.default_rng(999)

    x_base = np.linspace(-1.0, 1.0, n)
    s = rng.normal(0, noise_init, (n, 2))
    s[:, 0] = x_base
    s[0] = START; s[-1] = GOAL

    history = [(0, s.copy())]
    energy_curve = [(0, float(obstacle_energy(s, context)))]
    current_noise = noise_init * 0.45

    for k in range(1, n_steps + 1):
        # Smooth denoising step
        s_new = s.copy()
        s_new[:, 1] = gaussian_filter1d(s[:, 1], sigma=1.8)

        # Obstacle + endpoint gradient (stochastic)
        g_obs = obstacle_gradient(s_new, context, lam=25.0)
        g_end = endpoint_gradient(s_new, lam=30.0)
        s_new -= step_size * (g_obs + g_end)

        # Persistent noise — what makes diffusion stochastic
        noise = rng.normal(0, current_noise, (n, 2))
        noise[:, 0] = 0; noise[0] = noise[-1] = 0
        s_new += noise
        current_noise *= noise_decay

        s_new[0] = START; s_new[-1] = GOAL
        s = s_new

        if k % 10 == 0:
            history.append((k, s.copy()))
            energy_curve.append((k, float(obstacle_energy(s, context))))

    history.append((n_steps, s.copy()))
    energy_curve.append((n_steps, float(obstacle_energy(s, context))))
    return {'trajectory': s, 'history': history, 'energy_curve': energy_curve}


# ── C. EBM / Direct Optimization ─────────────────────────────────────────────

def ebm_inference(context: np.ndarray = OBS_CENTRES,
                  n: int = SEQ_LEN,
                  n_steps: int = 200,
                  lr: float = 0.008,
                  rng: np.random.Generator = None) -> dict:
    """
    EBM: gradient descent on obstacle energy from RANDOM initialisation.
    No proposal anchoring → context-irrelevant equilibria, local trapping.
    """
    if rng is None:
        rng = np.random.default_rng(77)

    x_base = np.linspace(-1.0, 1.0, n)
    s = rng.normal(0, 0.4, (n, 2))
    s[:, 0] = x_base
    s[0] = START; s[-1] = GOAL

    history = [(0, s.copy())]
    energy_curve = [(0, float(obstacle_energy(s, context)))]

    for k in range(1, n_steps + 1):
        g_obs = obstacle_gradient(s, context, lam=80.0)
        g_end = endpoint_gradient(s, lam=50.0)
        # Light smoothness only
        from energy import smooth_gradient
        g_sm  = smooth_gradient(s, lam=2.0)
        grad  = g_obs + g_end + g_sm

        gnorm = np.linalg.norm(grad)
        if gnorm > 3.0:
            grad = grad / gnorm * 3.0

        s = s - lr * grad
        s[0] = START; s[-1] = GOAL

        if k % 25 == 0:
            history.append((k, s.copy()))
            energy_curve.append((k, float(obstacle_energy(s, context))))

    # Post-smooth
    s[:, 1] = gaussian_filter1d(s[:, 1], sigma=2.0)
    s[0] = START; s[-1] = GOAL

    history.append((n_steps, s.copy()))
    energy_curve.append((n_steps, float(obstacle_energy(s, context))))
    return {'trajectory': s, 'history': history, 'energy_curve': energy_curve}


# ── D. SettlingInferenceModule ────────────────────────────────────────────────

class SettlingInferenceModule:
    """
    Settling-Based Inference (SBI) — Section III of the paper.

    Three-component decomposition:
      Module I   — Proposal π_φ(c):  mean-seeking, analytically y=0
      Module II  — Energy E(s,c):    obstacle barrier (non-convex validity)
      Module III — Dynamics:          s^{k+1} = s^k − η ∇_s E  (Eq. 16)

    Key innovation over bebV3.BicameralEnergyBlock:
      • Control-point parameterisation → smooth by construction (no zigzag)
      • Cubic spline interpolation → C2-smooth output
      • Strong bifurcation: large noise (σ=0.30) + weak anchor + strong obstacle
      • Gradient clipping in GLOBAL (not per-waypoint) sense
    """

    def __init__(self,
                 lam_obs:    float = 80.0,
                 lam_anchor: float = 0.05,   # very weak — let settling be free
                 x_cp:       np.ndarray = None):
        self.lam_obs    = lam_obs
        self.lam_anchor = lam_anchor
        self.x_cp = x_cp if x_cp is not None else _X_CP.copy()

    # ── Module I: Proposal ───────────────────────────────────────────────

    def proposal(self, context: np.ndarray = OBS_CENTRES,
                 n: int = SEQ_LEN) -> np.ndarray:
        """
        π_φ(c): conditional mean — intentionally invalid.
        Analytically: y = 0.5*y_A + 0.5*y_B = 0 for symmetric context.
        This is the saddle point that settling must escape.
        """
        return mean_trajectory(n)

    def proposal_cp(self) -> np.ndarray:
        """Control-point representation of the conditional mean (all zeros)."""
        return np.zeros(len(self.x_cp))     # y = 0 → invalid conditional mean

    # ── Module III: Settling Dynamics (control-point space) ──────────────

    def settle(self, context: np.ndarray = OBS_CENTRES,
               n: int = SEQ_LEN,
               n_steps: int = 100,
               lr: float = 0.025,
               noise_std: float = 0.30,
               rng: np.random.Generator = None,
               record_every: int = 5) -> dict:
        """
        Core settling loop in control-point space.

        s^{(0)} = π_φ(c) + ε,   ε ~ N(0, σ²)   (Algorithm 1, line 1)
        s^{(k+1)} = s^{(k)} − η ∇_{s_cp} E_obstacle(traj(s_cp), c)

        Working in control-point space (7 parameters) instead of full
        trajectory space (80 parameters) gives:
          • Smoothness by construction (cubic spline interpolation)
          • No Laplacian regularisation needed
          • Faster, more stable optimisation
          • No zigzag artefacts possible
        """
        if rng is None:
            rng = np.random.default_rng(42)

        # Initialise: proposal (y=0) + symmetry-breaking noise ε
        y_cp = self.proposal_cp()
        y_cp[1:-1] += rng.normal(0, noise_std, len(self.x_cp) - 2)
        y_cp[0] = y_cp[-1] = 0.0          # hard-pin endpoints

        y_cp_proposal = y_cp.copy()        # for anchoring

        history = [(0, traj_from_cp(y_cp, self.x_cp, n))]
        energy_curve = [(0, obstacle_energy(traj_from_cp(y_cp, self.x_cp, n),
                                            context, self.lam_obs))]

        for k in range(1, n_steps + 1):
            # Obstacle energy gradient w.r.t. control points
            g_obs = cp_gradient(y_cp, context, self.lam_obs)

            # Very weak anchor: allow settling to explore freely
            g_anchor = 2.0 * self.lam_anchor * (y_cp - y_cp_proposal)

            grad = g_obs + g_anchor
            grad[0] = grad[-1] = 0.0      # keep endpoints fixed

            # Global gradient norm clipping
            gnorm = np.linalg.norm(grad)
            if gnorm > 2.0:
                grad = grad * 2.0 / gnorm

            y_cp = y_cp - lr * grad
            y_cp[0] = y_cp[-1] = 0.0      # pin endpoints

            if k % record_every == 0 or k == n_steps:
                traj = traj_from_cp(y_cp, self.x_cp, n)
                history.append((k, traj.copy()))
                energy_curve.append((k, obstacle_energy(traj, context, self.lam_obs)))

        # Final dense smooth trajectory
        final_traj = traj_from_cp(y_cp, self.x_cp, n)

        return {
            'trajectory':   final_traj,
            'proposal':     self.proposal(context, n),
            'history':      history,
            'energy_curve': energy_curve,
            'y_cp_final':   y_cp,
        }

    def settle_with_snapshots(self, context: np.ndarray = OBS_CENTRES,
                              n: int = SEQ_LEN,
                              n_steps: int = 100,
                              lr: float = 0.025,
                              noise_std: float = 0.30,
                              rng: np.random.Generator = None) -> dict:
        """Dense history (every iteration) for Fig 3 and animation."""
        return self.settle(context, n, n_steps, lr, noise_std, rng, record_every=1)
