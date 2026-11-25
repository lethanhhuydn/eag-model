"""Static figures for reports. Colours follow a validated colour-blind-safe palette."""
from __future__ import annotations

import io

import numpy as np

SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
ORDINAL = ["#84b9f3", "#338ced", "#1b64be", "#0e3d7e"]      # light to dark, one hue
INK, INK2, GRID, MUTED = "#191919", "#666666", "#e3e3e3", "#9c9c9c"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "axes.axisbelow": True,
    })
    return plt


def style(ax, grid="y"):
    ax.grid(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8, linestyle="-")
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return buf.getvalue()


def profile_figure(profile, kappa, ci):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    order = np.argsort(profile["kappa"])
    k = profile["kappa"][order]
    rel = profile["loglik"][order] - profile["max"]
    ax.plot(k, rel, color=SERIES[0], linewidth=2, solid_capstyle="round", solid_joinstyle="round")
    ax.plot([kappa], [0], "o", color=SERIES[0], markersize=8, markeredgecolor="white", markeredgewidth=2)
    ax.axhline(-1.92, color=MUTED, linewidth=0.8)
    lo = min(-12.0, rel[np.isfinite(rel)].min() if np.isfinite(rel).any() else -12)
    ax.set_ylim(max(lo, -40) * 1.05, 2.5)
    ax.set_xscale("log")
    ticks = [0.05, 0.1, 0.25, 0.5, 1, 2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    for ref, lab in ((1.0, "sum score"), (0.5, "geometric mean")):
        ax.axvline(ref, color=GRID, linewidth=0.8)
        ax.text(ref, ax.get_ylim()[0], f" {lab}", rotation=90, va="bottom", ha="right", color=INK2, fontsize=8)
    ax.text(k.min(), -1.92, "95% confidence limit", va="bottom", ha="left", color=INK2, fontsize=8)
    ax.annotate(f"estimate {kappa:.2f}\n({ci[0]:.2f} to {ci[1]:.2f})", (kappa, 0), xytext=(10, -30),
                textcoords="offset points", fontsize=8, color=INK, va="top")
    ax.set_xlabel("Compensation index, kappa (log scale)")
    ax.set_ylabel("Log-likelihood relative to maximum")
    ax.set_title("How strongly do weak components hold competence back?")
    style(ax)
    return to_png(fig)


def form_figure(form, bins=5):
    """Mean residual of holistic ratings by profile unevenness, for the sum-score and fitted models."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    labels = {"linear": "Sum-score model", "free": "Fitted model"}
    for i, name in enumerate(("linear", "free")):
        f = form[name]
        edges = np.quantile(f["uneven"], np.linspace(0, 1, bins + 1))
        which = np.clip(np.searchsorted(edges, f["uneven"], side="right") - 1, 0, bins - 1)
        x = np.array([f["uneven"][which == b].mean() for b in range(bins)])
        y = np.array([f["resid"][which == b].mean() for b in range(bins)])
        ax.plot(x, y, color=SERIES[i], linewidth=2, marker="o", markersize=8, markeredgecolor="white",
                markeredgewidth=2, label=labels[name])
        ax.annotate(labels[name], (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Unevenness of the component profile (SD of mastery)")
    ax.set_ylabel("Mean residual (rating points)")
    ax.set_title("Do uneven profiles earn lower ratings than predicted?")
    ax.legend(loc="lower left", fontsize=8)
    ax.margins(x=0.25)
    style(ax)
    return to_png(fig)


def weights_figure(weights):
    plt = _plt()
    n = len(weights)
    fig, ax = plt.subplots(figsize=(5.2, 0.5 * n + 1.0))
    y = np.arange(n)[::-1]
    ax.hlines(y, weights["lower"], weights["upper"], color=SERIES[0], linewidth=2)
    ax.plot(weights["weight"], y, "o", color=SERIES[0], markersize=8, markeredgecolor="white", markeredgewidth=2)
    for yi, (w, up) in zip(y, zip(weights["weight"], weights["upper"])):
        ax.text(up + 0.01, yi, f"{w:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(weights["component"])
    ax.set_xlim(0, max(0.5, float(weights["upper"].max()) + 0.08))
    ax.set_xlabel("Weight (point estimate and 95% confidence interval)")
    ax.set_title("Component weights")
    style(ax, grid="x")
    return to_png(fig)


def trajectory_figure(fit, students):
    plt = _plt()
    data = fit.data
    groups = data.group_labels if fit.has_group else ["all students"]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.4 * len(groups) + 1.2, 3.0), sharey=True, squeeze=False)
    pi = fit.student["pi_mean"]
    for g, ax in enumerate(axes[0]):
        idx = np.flatnonzero(data.group == g) if fit.has_group else np.arange(data.n_students)
        for k, comp in enumerate(data.components):
            ax.plot(data.times, pi[idx, :, k].mean(axis=0), color=SERIES[k % len(SERIES)], linewidth=2,
                    marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.5, label=comp)
        ax.set_title(str(groups[g]).capitalize())
        ax.set_xticks(data.times)
        ax.set_xlabel("Design cycle")
        ax.set_ylim(0, 1)
        style(ax)
    axes[0][0].set_ylabel("Mean mastery")
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    return to_png(fig)
