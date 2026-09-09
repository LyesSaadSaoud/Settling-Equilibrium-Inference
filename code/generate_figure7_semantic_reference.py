from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap = np.linspace(0.52, 0.83, 26)
    lam_c = 0.72

    # Smooth mean-seeking degradation reference.
    t = (overlap - overlap.min()) / (overlap.max() - overlap.min())
    mean_seeking = 0.27 * (1.0 - t**1.05) + 0.02

    # Canonical supercritical bifurcation branch (order-parameter magnitude).
    settling = 2.1 * np.sqrt(np.clip(overlap - lam_c, 0.0, None))
    settling = np.clip(settling, 0.0, 0.92)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(figsize=(8.4, 5.1), constrained_layout=True)
    band_lo, band_hi = lam_c - 0.012, lam_c + 0.016
    ax.axvspan(band_lo, band_hi, color="#E8D99A", alpha=0.45,
               label="Critical regime", zorder=0)
    ax.axvline(lam_c, color="#B28A00", lw=1.4, ls=":", zorder=1)
    ax.axhline(0.0, color="#666666", lw=0.9, ls=":", zorder=1)

    ax.plot(overlap, settling, color="#2E8B57", lw=2.8,
            label="Settling bifurcation branch", zorder=3)
    ax.plot(overlap, mean_seeking, color="#777777", lw=2.0, ls="--",
            label="Mean-seeking degradation", zorder=3)

    ax.annotate(
        r"critical overlap $\lambda_c$",
        xy=(lam_c, 0.02), xytext=(lam_c + 0.035, 0.16),
        ha="left", va="bottom", color="#2E8B57", fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.9, color="#2E8B57")
    )

    ax.set_xlim(0.50, 0.90)
    ax.set_ylim(-0.04, 1.02)
    ax.set_xlabel(r"Semantic ambiguity (overlap $\lambda$)")
    ax.set_ylabel("Discriminative separation / order-parameter magnitude")
    ax.set_title("Phase Transition: Continuous Collapse vs. Bifurcation", fontweight="bold")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", frameon=False)

    png = out_dir / "figure7_semantic_bifurcation.png"
    pdf = out_dir / "figure7_semantic_bifurcation.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {png}")
    print(f"saved: {pdf}")


if __name__ == "__main__":
    main()
