"""Figures of the article. Figures 1 and 2 follow from the model alone; Figures 3 and 4 use the Monte Carlo output."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from eagmodel.functions import aggregate, leverage, mastery, rho_from_kappa      # noqa: E402
from eagmodel.plots import GRID, INK, INK2, MUTED, ORDINAL, SERIES, _plt, style   # noqa: E402

FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
KAPPAS = [1.0, 0.5, 0.25, 0.1]
GREY = "#9c9c9c"


def paper_style():
    plt = _plt()
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
                         "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=400)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".eps"))            # vector file for journal submission
    import matplotlib.pyplot as plt

    plt.close(fig)


def panel_label(ax, letter, x=-0.16, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, va="bottom", ha="left")


def spread(ys, gap):
    """Push label positions apart so that neighbours are at least `gap` apart, keeping their order."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for _ in range(200):
        moved = False
        for a, b in zip(order[:-1], order[1:]):
            if out[b] - out[a] < gap:
                mid = 0.5 * (out[a] + out[b])
                out[a], out[b] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    return out


# ------------------------------------------------------------------------------------------------ Figure 1
def figure1():
    plt = paper_style()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.1, 4.75))
    ax.set_xlim(-1, 101)
    ax.set_ylim(0, 67)
    ax.axis("off")
    blue, pale, line = SERIES[0], "#eef4fd", "#b6d5f8"
    oy = 9.0                                                    # vertical offset of panel a

    def box(x, y, w, h, text, fc="white", ec=line, fs=6.8, color=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, linespacing=1.4)

    def arrow(p, q, rad=0.0, color=INK2):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.9, color=color,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=1.2, shrinkB=1.2))

    def label(x, y, text):
        ax.text(x, y, text, ha="center", va="center", fontsize=6.9, fontweight="bold", color=blue)

    def note(x, y, text, ha="center", fs=6.2):
        ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=INK2, style="italic", linespacing=1.35)

    ax.text(0, 64.5, "a", fontsize=11, fontweight="bold")
    for x, t in ((11, "Rubric evidence"), (48.5, "Component proficiency"), (75, "Competence"), (94.5, "Use")):
        ax.text(x, 52.5 + oy, t, ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK)
    # rubric evidence
    box(0, 30 + oy, 22, 14, "Analytic rubric\n20 indicators, 4 levels\nscored by the teacher\nin every design cycle")
    arrow((38.5, 37 + oy), (22, 37 + oy))
    ax.text(30.2, 41.0 + oy, "Evidence\nfunction", ha="center", va="center", fontsize=6.9, fontweight="bold",
            color=blue, linespacing=1.25)
    note(30.2, 34.0 + oy, "rating scale\nmodel")
    # components and growth
    comps = ["Problem framing", "System modelling", "Measurement design", "Inference design", "Iterative refinement"]
    ys = [44.6, 39.2, 33.8, 28.4, 23.0]
    for c, y in zip(comps, ys):
        box(38.5, y + oy, 20, 4.2, c, fc=pale, ec=line, fs=6.7)
    label(48.5, 19.6 + oy, "Growth function")
    ax.text(48.5, 14.6 + oy, r"$\theta_k(t)=a_k+b_k\,t$" + "\n" + r"$\pi_k(t)=1/(1+e^{-\theta_k(t)})$", ha="center",
            va="center", fontsize=7, color=INK, linespacing=1.5)
    # competence
    box(65, 30 + oy, 20, 13, "", ec=blue)
    ax.text(75, 38.2 + oy, r"$C(t)=\left(\sum_k w_k\,\pi_k(t)^{\rho}\right)^{1/\rho}$", ha="center", va="center", fontsize=7)
    note(75, 32.4 + oy, r"$\kappa=1/(2-\rho)$ for $\rho\leq 1$", fs=6.2)
    label(75, 45.4 + oy, "Aggregation function")
    for y in ys:
        arrow((58.5, y + 2.1 + oy), (65, 36.5 + oy), color="#9db8d9")
    # holistic review
    box(65, 11 + oy, 20, 10.5, "Holistic design review\nscore 0 to 100\ntwo raters")
    arrow((75, 30 + oy), (75, 21.5 + oy))
    note(76.2, 25.8 + oy, "rating link,\nrater severity", ha="left")
    # uses
    for yc, t in ((45.8, "Teach next:\nlargest\n" + r"$\partial C/\partial\theta_k$"),
                  (36.5, "Readiness:\n" + r"$P(C\geq C^{*})$"), (27.2, "Forecast:\ncycles\nto " + r"$C^{*}$")):
        box(89, yc - 4.1 + oy, 11, 8.2, t, fs=6.4)
        arrow((85, 36.5 + oy), (89, yc + oy))

    # ---------------- panel b: classroom-to-fair pathway
    ax.text(0, 13.2, "b", fontsize=11, fontweight="bold")
    y0, h = 1.6, 5.6
    x = 3.0
    for p in ["Frame", "Model", "Sense", "Infer", "Refine"]:
        box(x, y0, 7.0, h, p, fc=pale, ec=line, fs=6.5)
        arrow((x + 7.0, y0 + h / 2), (x + 8.4, y0 + h / 2))
        x += 8.4
    box(x, y0, 9.0, h, "Design\nreview", fs=6.4)
    arrow((x + 9.0, y0 + h / 2), (x + 10.6, y0 + h / 2))
    xr = x + 10.6
    box(xr, y0, 16.5, h, "EAG report:\nteach next, ready?", ec=blue, fs=6.2)
    ax.add_patch(FancyArrowPatch((xr + 8.2, y0 + h), (6.5, y0 + h), arrowstyle="-|>", mutation_scale=7, lw=0.9,
                                 color=INK2, connectionstyle="arc3,rad=0.11", shrinkA=1, shrinkB=1))
    ax.text(36.5, y0 + h + 5.3, "next design cycle, teaching targeted at each student's highest-leverage component",
            ha="center", va="center", fontsize=6.1, color=INK2, style="italic")
    arrow((xr + 16.5, y0 + h / 2), (xr + 19.1, y0 + h / 2))
    box(xr + 19.1, y0, 8.0, h, "School\nfair", fs=6.4)
    arrow((xr + 27.1, y0 + h / 2), (xr + 29.7, y0 + h / 2))
    box(xr + 29.7, y0, 14.4, h, "Regional and\ninternational fairs", fs=6.2)
    ax.text(xr + 17.8, y0 + h + 1.9, r"$C\geq C^{*}_{\mathrm{school}}$", ha="center", va="center", fontsize=6.2, color=INK2)
    ax.text(xr + 28.6, y0 + h + 1.9, r"$C\geq C^{*}_{\mathrm{fair}}$", ha="center", va="center", fontsize=6.2, color=INK2)
    save(fig, "figure1_model")


# ------------------------------------------------------------------------------------------------ Figure 2
def targeted_trajectories(theta0, w, rho, budget=1.0, cycles=8, steps=10):
    """Competence under uniform growth and under growth allocated to the highest-leverage component."""
    uni = [aggregate(mastery(theta0), w, rho)]
    tar = [uni[0]]
    th_u, th_t = theta0.copy(), theta0.copy()
    for _ in range(cycles):
        th_u = th_u + budget / len(w)
        uni.append(aggregate(mastery(th_u), w, rho))
        for _ in range(steps):
            k = int(np.argmax(leverage(mastery(th_t), w, rho)))
            th_t[k] += budget / steps
        tar.append(aggregate(mastery(th_t), w, rho))
    return np.array(uni), np.array(tar)


def figure2():
    plt = paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), gridspec_kw=dict(wspace=0.52))
    w = np.full(5, 0.2)
    # a: the weakest component
    ax = axes[0]
    x = np.linspace(0.05, 0.95, 181)
    labels = {1.0: "κ = 1", 0.5: "κ = 0.5", 0.25: "κ = 0.25", 0.1: "κ = 0.1"}
    for i, k in enumerate(KAPPAS):
        prof = np.column_stack([x] + [np.full_like(x, 0.8)] * 4)
        ax.plot(x, aggregate(prof, w, float(rho_from_kappa(k))), color=ORDINAL[i], lw=2, solid_capstyle="round",
                label=labels[k])
    ax.set_xlabel("Mastery of the weakest component")
    ax.set_ylabel("Competence, C")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.3, loc="lower right", handlelength=1.4, borderaxespad=0.2)
    style(ax)
    panel_label(ax, "a", x=-0.24)
    # b: same rubric total, different competence
    ax = axes[1]
    even, uneven = np.full(5, 0.6), np.array([0.9, 0.9, 0.6, 0.3, 0.3])
    xs = np.arange(4)
    ce = [aggregate(even, w, float(rho_from_kappa(k))) for k in KAPPAS]
    cu = [aggregate(uneven, w, float(rho_from_kappa(k))) for k in KAPPAS]
    ax.plot(xs, ce, color=GREY, lw=2, marker="o", ms=6.5, mec="white", mew=1.5, label="Even profile")
    ax.plot(xs, cu, color=SERIES[0], lw=2, marker="o", ms=6.5, mec="white", mew=1.5, label="Uneven profile")
    ax.annotate("even profile", (xs[-1], ce[-1]), xytext=(0, 6), textcoords="offset points", ha="right", fontsize=6.5)
    ax.annotate("uneven profile", (xs[-1], cu[-1]), xytext=(0, -10), textcoords="offset points", ha="right", fontsize=6.5)
    ax.legend(loc="lower left", fontsize=6.3, handlelength=1.4, borderaxespad=0.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["1", "0.5", "0.25", "0.1"])
    ax.set_xlabel("Compensation index κ")
    ax.set_ylabel("Competence, C")
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.3, 3.3)
    style(ax)
    panel_label(ax, "b", x=-0.24)
    # c: teaching by leverage
    ax = axes[2]
    theta0 = np.log(uneven / (1 - uneven))
    uni, tar = targeted_trajectories(theta0, w, -2.0)
    t = np.arange(uni.size)
    ax.plot(t, uni, color=GREY, lw=2, marker="o", ms=6.5, mec="white", mew=1.5, label="Uniform")
    ax.plot(t, tar, color=SERIES[0], lw=2, marker="o", ms=6.5, mec="white", mew=1.5, label="Leverage-targeted")
    ax.axhline(0.6, color=MUTED, lw=0.8)
    ax.text(8.3, 0.568, "readiness threshold", ha="right", va="top", fontsize=6.0, color=INK2)
    ax.legend(title="Teaching", loc="lower right", fontsize=6.3, title_fontsize=6.3, handlelength=1.4, borderaxespad=0.2, bbox_to_anchor=(1.0, 0.03))
    ax.set_xlabel("Design cycles")
    ax.set_ylabel("Competence, C")
    ax.set_ylim(0, 1)
    ax.set_xticks(t)
    style(ax)
    panel_label(ax, "c", x=-0.24)
    save(fig, "figure2_implications")
    first = lambda v: int(np.argmax(v >= 0.6)) if (v >= 0.6).any() else None
    return dict(even=ce, uneven=cu, uniform=uni.tolist(), targeted=tar.tolist(),
                cycles_uniform=first(uni), cycles_targeted=first(tar))


# ------------------------------------------------------------------------------------------------ Figure 3
def figure3(d):
    plt = paper_style()
    main = d[(d.block == "main") & (d.failed == 0)]
    sizes = sorted(main.n.unique())
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), gridspec_kw=dict(wspace=0.5))
    xs = np.arange(len(KAPPAS))
    dodge = np.linspace(-0.22, 0.22, len(sizes))
    col = [ORDINAL[0], SERIES[0], ORDINAL[3]]
    # a: sampling distribution of the estimate
    ax = axes[0]
    for xi, k in zip(xs, KAPPAS):
        ax.plot([xi - 0.36, xi + 0.36], [k, k], color=GREY, lw=1.0, zorder=1)
        for j, n in enumerate(sizes):
            g = main[(np.isclose(main.kappa_true, k)) & (main.n == n)].kappa
            lo, md, hi = g.quantile([0.025, 0.5, 0.975])
            ax.plot([xi + dodge[j]] * 2, [lo, hi], color=col[j], lw=2, solid_capstyle="round", zorder=2)
            ax.plot(xi + dodge[j], md, "o", color=col[j], ms=6.5, mec="white", mew=1.5, zorder=3,
                    label=f"{n}" if xi == 0 else None)
    ax.set_yscale("log")
    ax.set_yticks([0.03, 0.1, 0.25, 0.5, 1, 2])
    ax.set_yticklabels(["0.03", "0.1", "0.25", "0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k:g}" for k in KAPPAS])
    ax.set_xlabel("True compensation index κ")
    ax.set_ylabel("Estimated κ (log scale)")
    ax.legend(title="Students", fontsize=6.5, title_fontsize=6.5, loc="lower left", handlelength=0.8, borderaxespad=0.2)
    style(ax)
    panel_label(ax, "a", x=-0.27)
    # b: rejection of the sum-score model
    ax = axes[1]
    for j, n in enumerate(sizes):
        y = [main[(np.isclose(main.kappa_true, k)) & (main.n == n)].reject_linear.mean() for k in KAPPAS]
        ax.plot(xs, y, color=col[j], lw=2, marker="o", ms=6.5, mec="white", mew=1.5)
    ax.axhline(0.05, color=MUTED, lw=0.8)
    ax.text(3.3, 0.065, "nominal 5%", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k:g}" for k in KAPPAS])
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("True compensation index κ")
    ax.set_ylabel("Sum-score model rejected (proportion)")
    style(ax)
    panel_label(ax, "b", x=-0.27)
    # c: coverage
    ax = axes[2]
    for j, n in enumerate(sizes):
        y = [main[(np.isclose(main.kappa_true, k)) & (main.n == n)].cover_kappa.mean() for k in KAPPAS]
        ax.plot(xs + dodge[j], y, "o", color=col[j], ms=6.5, mec="white", mew=1.5)
    ax.axhline(0.95, color=MUTED, lw=0.8)
    ax.text(3.3, 0.955, "nominal 95%", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k:g}" for k in KAPPAS])
    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(0.7, 1.02)
    ax.set_xlabel("True compensation index κ")
    ax.set_ylabel("Coverage of 95% interval for κ")
    style(ax)
    panel_label(ax, "c", x=-0.27)
    save(fig, "figure3_recovery")


# ------------------------------------------------------------------------------------------------ Figure 4
def figure4(d, n=200):
    plt = paper_style()
    main = d[(d.block == "main") & (d.failed == 0) & (d.n == n)]
    xs = np.arange(len(KAPPAS))
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.75), gridspec_kw=dict(wspace=0.95, right=0.87))

    def lines(ax, series, ylabel, ylim, gap):
        ends = []
        for col, lab, color in series:
            y = [main[np.isclose(main.kappa_true, k)][col].mean() for k in KAPPAS]
            ax.plot(xs, y, color=color, lw=2, marker="o", ms=6.5, mec="white", mew=1.5, label=lab)
            ends.append(y[-1])
        pos = spread(ends, gap)
        for (col, lab, color), y_end, y_lab in zip(series, ends, pos):
            ax.annotate(lab, (xs[-1], y_end), xytext=(xs[-1] + 0.3, y_lab), textcoords="data", fontsize=6.4, va="center",
                        ha="left", color=INK, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=GRID, lw=0.7, shrinkA=0, shrinkB=3))
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{k:g}" for k in KAPPAS])
        ax.set_xlim(-0.3, 3.3)
        ax.set_ylim(*ylim)
        ax.set_xlabel("True compensation index κ")
        ax.set_ylabel(ylabel)
        style(ax)

    eag, lin, summ, hol = SERIES[0], SERIES[1], SERIES[2], SERIES[3]
    lines(axes[0], [("top_eag", "EAG", eag), ("top_eaglin", "EAG, κ = 1", lin), ("top_sum", "rubric total", summ),
                    ("top_holistic", "mean rating", hol)], "Top 30% correctly selected", (0.7, 1.0), 0.022)
    panel_label(axes[0], "a", x=-0.3)
    lines(axes[1], [("false_ready_eag", "EAG", eag), ("false_ready_eaglin", "EAG, κ = 1", lin)],
          "Declared ready but not ready", (0.0, 0.4), 0.03)
    panel_label(axes[1], "b", x=-0.3)
    lines(axes[2], [("target_eag", "EAG", eag), ("target_eaglin", "EAG, κ = 1", lin),
                    ("target_lowest", "lowest subscore", summ)], "Teaching target correct", (0.0, 1.0), 0.075)
    panel_label(axes[2], "c", x=-0.3)
    save(fig, "figure4_consequences")


def main():
    info = figure2()
    figure1()
    path = os.path.join(RES, "montecarlo.csv")
    if os.path.exists(path):
        d = pd.read_csv(path)
        figure3(d)
        figure4(d)
    import json

    with open(os.path.join(RES, "figure2_numbers.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print(info)


if __name__ == "__main__":
    main()
