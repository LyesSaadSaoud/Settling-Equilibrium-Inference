"""
animation.py
============
Settling evolution animations — GIF and MP4.

Shows: proposal (invalid mean) → symmetry breaking → bifurcation → valid equilibrium.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from environment import (OBS_CENTRES, OBS_RADIUS, OBS_R, START, GOAL, COLORS,
                          upper_trajectory, lower_trajectory, is_collision_free, SEQ_LEN)
from energy import total_energy

ANIM_DIR = Path("animations")
ANIM_DIR.mkdir(exist_ok=True)


def _draw_obs_ax(ax):
    for cx, cy in OBS_CENTRES:
        ax.add_patch(plt.Circle((cx, cy), OBS_R,
                                color=COLORS['margin'], alpha=0.5, zorder=2))
        ax.add_patch(plt.Circle((cx, cy), OBS_RADIUS,
                                color=COLORS['core'],   alpha=0.9, zorder=3))
    ax.plot(*START, 'o', ms=8, color=COLORS['settling'], zorder=6, mec='k', mew=0.7)
    ax.plot(*GOAL,  '*', ms=10, color='#E6550D',         zorder=6, mec='k', mew=0.7)
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.18, color='#BBBBBB', lw=0.5)
    ax.set_xlabel('x'); ax.set_ylabel('y')


def make_gif(settling_result, filename="animations/settling_animation.gif",
             fps: int = 10):
    """Export settling evolution as animated GIF."""
    try:
        from PIL import Image
        import io
    except ImportError:
        print("  ⚠  Pillow not available — skipping GIF")
        return

    history = settling_result['history']
    proposal = settling_result['proposal']
    upper = upper_trajectory()
    lower = lower_trajectory()

    frames = []
    cmap = plt.cm.plasma
    n_hist = len(history)

    for fi, (k, traj) in enumerate(history):
        fig, ax = plt.subplots(figsize=(7, 4.0), dpi=90)
        _draw_obs_ax(ax)

        ax.plot(upper[:, 0], upper[:, 1], color=COLORS['upper'], lw=1.1,
                alpha=0.25, ls='--')
        ax.plot(lower[:, 0], lower[:, 1], color=COLORS['lower'], lw=1.1,
                alpha=0.25, ls='--')
        ax.plot(proposal[:, 0], proposal[:, 1], color=COLORS['proposal'], lw=1.2,
                ls=':', alpha=0.55, label='Proposal π_φ(c)')

        col = cmap(0.12 + 0.76 * fi / max(n_hist - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], color=col, lw=2.6, zorder=6)

        e = total_energy(traj, proposal, OBS_CENTRES)
        valid = is_collision_free(traj)
        status = 'VALID  ✓' if valid else 'invalid'
        status_col = '#1A9850' if valid else '#CC0000'

        ax.set_title(f'Settling Inference — Iteration k = {k:3d}   '
                     f'E = {e:.3f}', fontsize=10, fontweight='bold')
        ax.text(0.98, 0.96, status, transform=ax.transAxes,
                ha='right', va='top', color=status_col,
                fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, frameon=False)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=90)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())

    frames[0].save(filename, save_all=True,
                   append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"  ✓  GIF saved → {filename}  ({len(frames)} frames)")


def make_mp4(settling_result,
             filename="animations/settling_animation.mp4",
             fps: int = 12):
    """Export settling evolution as MP4 (requires ffmpeg)."""
    try:
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        import shutil
        if not shutil.which('ffmpeg'):
            print("  ⚠  ffmpeg not found — skipping MP4")
            return
    except Exception as exc:
        print(f"  ⚠  MP4 unavailable: {exc}")
        return

    history = settling_result['history']
    proposal = settling_result['proposal']
    upper = upper_trajectory()
    lower = lower_trajectory()
    n_hist = len(history)

    fig, ax = plt.subplots(figsize=(7, 4.0), dpi=90)

    def draw_frame(fi):
        ax.cla()
        _draw_obs_ax(ax)
        ax.plot(upper[:, 0], upper[:, 1], color=COLORS['upper'], lw=1.1,
                alpha=0.25, ls='--')
        ax.plot(lower[:, 0], lower[:, 1], color=COLORS['lower'], lw=1.1,
                alpha=0.25, ls='--')
        ax.plot(proposal[:, 0], proposal[:, 1], color=COLORS['proposal'],
                lw=1.2, ls=':', alpha=0.55)

        k, traj = history[fi]
        col = plt.cm.plasma(0.12 + 0.76 * fi / max(n_hist - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], color=col, lw=2.6)

        e = total_energy(traj, proposal, OBS_CENTRES)
        ax.set_title(f'Settling — k = {k}   E = {e:.3f}', fontweight='bold')
        fig.tight_layout()

    anim = FuncAnimation(fig, draw_frame, frames=n_hist, interval=80)
    writer = FFMpegWriter(fps=fps)
    anim.save(filename, writer=writer)
    plt.close(fig)
    print(f"  ✓  MP4 saved → {filename}")
