"""
environment.py
==============
Double-Bifurcation Environment (from bebV3.py, analytically extended).

Two circular obstacles positioned symmetrically at x = ±0.4, y = 0.
This creates the canonical non-convex validity structure:

  Valid Mode A (upper):  trajectory arcs ABOVE both obstacles  →  y > 0.3
  Valid Mode B (lower):  trajectory arcs BELOW both obstacles  →  y < -0.3
  Conditional Mean:      arithmetic mean ≈ y = 0              →  COLLISION

This is the geometric instantiation of Conditional Mean Collapse (Theorem 1).

Obstacle geometry inherits bebV3.py:
  radius = 0.25, safety margin = 0.05
  World:   x ∈ [-1, 1],  y ∈ [-1, 1]
  Start:   (-1, 0)
  Goal:    ( 1, 0)
"""

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
SEQ_LEN       = 40
OBS_RADIUS    = 0.25
SAFETY_MARGIN = 0.05
OBS_R         = OBS_RADIUS + SAFETY_MARGIN   # effective constraint radius

START = np.array([-1.0, 0.0])
GOAL  = np.array([ 1.0, 0.0])

# Fixed test-time obstacle centres (obs1_y = obs2_y = 0 → symmetric, from bebV3)
OBS_CENTRES = np.array([[-0.4, 0.0],   # left obstacle
                         [ 0.4, 0.0]])  # right obstacle

COLORS = {
    "attention": "#1F77B4",   # standard scientific blue
    "diffusion":  "#FF7F0E",  # standard scientific orange
    "ebm":        "#9467BD",  # standard scientific purple
    "settling":   "#2CA02C",  # standard scientific green
    "proposal":   "#7F7F7F",  # neutral gray
    "core":       "#333333",  # dark charcoal (obstacle body)
    "margin":     "#D9D9D9",  # light silver (safety margin)
    "upper":      "#D62728",  # mode A (red)
    "lower":      "#17BECF",  # mode B (teal)
    "mean":       "#8C564B",  # conditional mean (brown/invalid)
}


# ── Trajectory generators ────────────────────────────────────────────────────

def _make_x(n: int = SEQ_LEN) -> np.ndarray:
    return np.linspace(-1.0, 1.0, n)


def _bump(x: np.ndarray, cx: float, sigma: float, amp: float) -> np.ndarray:
    """Gaussian bump centred at cx."""
    return amp * np.exp(-((x - cx) / sigma) ** 2)


def upper_trajectory(n: int = SEQ_LEN, amp: float = 0.72) -> np.ndarray:
    """
    Valid Mode A — arcs above both obstacles.
    y(x) = G(-0.4) + G(+0.4) with positive amplitude.
    """
    x = _make_x(n)
    y = _bump(x, -0.4, 0.38, amp) + _bump(x, 0.4, 0.38, amp)
    return np.column_stack([x, y])


def lower_trajectory(n: int = SEQ_LEN, amp: float = 0.72) -> np.ndarray:
    """
    Valid Mode B — arcs below both obstacles (negated upper).
    """
    t = upper_trajectory(n, amp)
    return np.column_stack([t[:, 0], -t[:, 1]])


def mean_trajectory(n: int = SEQ_LEN) -> np.ndarray:
    """
    Conditional mean of upper and lower modes.
    Arithmetic mean → y ≡ 0.0 → passes through both obstacle centres.
    This is the INVALID trajectory demonstrating Conditional Mean Collapse.
    """
    u = upper_trajectory(n)
    l = lower_trajectory(n)
    return 0.5 * (u + l)           # y = 0 everywhere


# ── Obstacle geometry ────────────────────────────────────────────────────────

def obstacle_sdf(points: np.ndarray,
                 obs: np.ndarray = OBS_CENTRES) -> np.ndarray:
    """
    Signed distance field for circular obstacle union.
    Returns min SDF over all obstacles:
      positive → outside (safe)
      negative → inside (collision)

    points: (N, 2)
    returns: (N,)
    """
    sdfs = []
    for cx, cy in obs:
        d = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2) - OBS_RADIUS
        sdfs.append(d)
    return np.minimum(*sdfs) if len(sdfs) > 1 else sdfs[0]


def is_collision_free(traj: np.ndarray,
                      obs: np.ndarray = OBS_CENTRES,
                      margin: float = 0.02) -> bool:
    """True if every waypoint clears all obstacles by at least margin."""
    for cx, cy in obs:
        dists = np.sqrt((traj[:, 0] - cx)**2 + (traj[:, 1] - cy)**2)
        if np.any(dists < OBS_RADIUS + margin):
            return False
    return True


def min_clearance(traj: np.ndarray,
                  obs: np.ndarray = OBS_CENTRES) -> float:
    """Minimum clearance from any obstacle across the trajectory."""
    min_d = np.inf
    for cx, cy in obs:
        dists = np.sqrt((traj[:, 0] - cx)**2 + (traj[:, 1] - cy)**2) - OBS_RADIUS
        min_d = min(min_d, float(dists.min()))
    return min_d


def trajectory_smoothness(traj: np.ndarray) -> float:
    """Mean squared second differences — lower is smoother."""
    d2 = traj[2:] - 2*traj[1:-1] + traj[:-2]
    return float(np.mean(d2**2))


# ── Randomised evaluation contexts ──────────────────────────────────────────

def random_context(rng: np.random.Generator,
                   obs_jitter: float = 0.1) -> np.ndarray:
    """
    Randomised obstacle centres for quantitative evaluation.
    Jitters around the symmetric (test-time) configuration.
    """
    base = OBS_CENTRES.copy()
    base[:, 1] += rng.uniform(-obs_jitter, obs_jitter, 2)
    return base
