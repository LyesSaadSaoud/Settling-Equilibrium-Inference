"""
energy.py
=========
Composite energy E_total and analytical gradients for settling dynamics.

E_total(s, c) = λ_obs  · E_obstacle(s, c)      ← non-convex constraint (Eq. 14)
              + λ_prop · ‖s − π_φ(c)‖²          ← proposal anchoring  (Eq. 18)
              + λ_sm   · R_smooth(s)              ← curvature regularization

All gradients are computed analytically to ensure stability.
The Laplacian gradient uses the bilaplacian formula (L^T L).
"""

import numpy as np
from environment import OBS_RADIUS, SAFETY_MARGIN, START, GOAL

OBS_R = OBS_RADIUS + SAFETY_MARGIN


# ── Obstacle energy (smooth barrier) ────────────────────────────────────────

def obstacle_energy(traj: np.ndarray,
                    obs_centres: np.ndarray,
                    lam: float = 80.0,
                    r: float = OBS_R) -> float:
    """
    E_obs = λ Σ_i Σ_j max(0, r − dist(s_i, obs_j))²

    Smooth ReLU-squared barrier — zero outside obstacle margin,
    quadratically increasing inside.  Provides non-convex validity structure.
    """
    E = 0.0
    for cx, cy in obs_centres:
        dx = traj[:, 0] - cx
        dy = traj[:, 1] - cy
        dist = np.sqrt(dx**2 + dy**2 + 1e-8)
        viol = np.maximum(0.0, r - dist)
        E += lam * np.sum(viol**2)
    return E


def obstacle_gradient(traj: np.ndarray,
                      obs_centres: np.ndarray,
                      lam: float = 80.0,
                      r: float = OBS_R) -> np.ndarray:
    """Analytical gradient of E_obs w.r.t. traj waypoints."""
    grad = np.zeros_like(traj)
    for cx, cy in obs_centres:
        dx = traj[:, 0] - cx
        dy = traj[:, 1] - cy
        dist = np.sqrt(dx**2 + dy**2 + 1e-8)
        viol = np.maximum(0.0, r - dist)
        # ∂E/∂traj_i = -2λ · viol_i · (traj_i − obs) / dist_i
        factor = -2.0 * lam * viol / dist
        grad[:, 0] += factor * dx
        grad[:, 1] += factor * dy
    return grad


# ── Smoothness energy (Laplacian / curvature) ────────────────────────────────

def smooth_energy(traj: np.ndarray, lam: float = 8.0) -> float:
    """
    R_smooth = λ Σ_i ‖s_{i+2} − 2s_{i+1} + s_i‖²

    Second-order finite differences penalise curvature.
    High λ forces the trajectory to be a near-polynomial curve,
    eliminating zigzagging artefacts.
    """
    d2 = traj[2:] - 2*traj[1:-1] + traj[:-2]
    return lam * float(np.sum(d2**2))


def smooth_gradient(traj: np.ndarray, lam: float = 8.0) -> np.ndarray:
    """
    Analytical gradient of R_smooth (bilaplacian operator L^T L).

    ∂R_smooth/∂s_i = 2λ (d2[i−2] − 2·d2[i−1] + d2[i])
    where d2[k] = s_{k+2} − 2s_{k+1} + s_k.
    """
    N = traj.shape[0]
    d2 = traj[2:] - 2*traj[1:-1] + traj[:-2]       # (N-2, 2)
    d2_ext = np.zeros((N, 2))
    d2_ext[1:N-1] = d2                               # zero-pad boundaries

    # Bilaplacian: d4[i] = d2[i] - 2*d2[i-1] + d2[i-2]
    d4 = np.zeros_like(traj)
    for i in range(1, N-1):
        v = np.zeros(2)
        if 0 <= i <= N-3:      v  += d2_ext[i]
        if 0 <= i-1 <= N-3:    v  -= 2 * d2_ext[i-1]
        if 0 <= i-2 <= N-3:    v  += d2_ext[i-2]
        d4[i] = v
    return 2.0 * lam * d4


# ── Proposal anchoring energy ────────────────────────────────────────────────

def anchor_energy(traj: np.ndarray, proposal: np.ndarray,
                  lam: float = 0.4) -> float:
    """
    λ_p ‖s − π_φ(c)‖² — prevents convergence to context-irrelevant equilibria.
    Provides semantic coherence with the (possibly invalid) proposal.
    """
    diff = traj - proposal
    return lam * float(np.sum(diff**2))


def anchor_gradient(traj: np.ndarray, proposal: np.ndarray,
                    lam: float = 0.4) -> np.ndarray:
    return 2.0 * lam * (traj - proposal)


# ── Endpoint pinning ─────────────────────────────────────────────────────────

def endpoint_energy(traj: np.ndarray, lam: float = 50.0) -> float:
    e  = lam * float(np.sum((traj[0]  - START)**2))
    e += lam * float(np.sum((traj[-1] - GOAL)**2))
    return e


def endpoint_gradient(traj: np.ndarray, lam: float = 50.0) -> np.ndarray:
    grad = np.zeros_like(traj)
    grad[0]  = 2.0 * lam * (traj[0]  - START)
    grad[-1] = 2.0 * lam * (traj[-1] - GOAL)
    return grad


# ── Total energy + gradient ──────────────────────────────────────────────────

def total_energy(traj: np.ndarray,
                 proposal: np.ndarray,
                 obs_centres: np.ndarray,
                 lam_obs: float = 80.0,
                 lam_smooth: float = 8.0,
                 lam_anchor: float = 0.4,
                 lam_end: float = 50.0) -> float:
    """E_total = E_obs + R_smooth + E_anchor + E_endpoints"""
    return (obstacle_energy(traj, obs_centres, lam_obs)
          + smooth_energy(traj, lam_smooth)
          + anchor_energy(traj, proposal, lam_anchor)
          + endpoint_energy(traj, lam_end))


def total_gradient(traj: np.ndarray,
                   proposal: np.ndarray,
                   obs_centres: np.ndarray,
                   lam_obs: float = 80.0,
                   lam_smooth: float = 8.0,
                   lam_anchor: float = 0.4,
                   lam_end: float = 50.0) -> np.ndarray:
    """Analytical gradient ∇_s E_total."""
    return (obstacle_gradient(traj, obs_centres, lam_obs)
          + smooth_gradient(traj, lam_smooth)
          + anchor_gradient(traj, proposal, lam_anchor)
          + endpoint_gradient(traj, lam_end))
