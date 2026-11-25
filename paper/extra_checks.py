"""Additional checks reported in the article, computed from the stored Monte Carlo output.

1. Power of the tests of the growth effects by size of the effect (0.10, 0.15, 0.20 logits per cycle).
2. Replications in which the estimate of the compensation index lies on a bound of the optimiser (0.02 or 20).
3. Replications in which the importance sampling of the second stage broke down, and their influence on the summaries.
4. Ties in the lowest-subscore heuristic and the effect of breaking them at random instead of by listing order.

Usage:  python paper/extra_checks.py            (writes paper/results/extra_checks.json)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from eagmodel.functions import leverage                              # noqa: E402
from eagmodel.simulate import SimConfig, simulate                   # noqa: E402
import summarise                                                    # noqa: E402

EFFECTS = (0.10, 0.15, 0.20, 0.15, 0.20)                            # growth effects of the five components
PROPORTIONS = ["kappa_coverage", "reject_linear", "top_eag", "top_eaglin", "top_sum", "top_holistic", "false_ready_eag",
               "false_ready_eaglin", "missed_ready_eag", "missed_ready_eaglin", "acc_eag", "acc_eaglin", "target_eag",
               "target_eaglin", "target_lowest", "db_coverage", "db_power", "w_coverage", "ppc_flag_free", "ppc_flag_lin",
               "form_flag_free", "form_flag_lin", "link_flag"]


def main():
    d = pd.read_csv(os.path.join(HERE, "results", "montecarlo.csv"))
    main_block = d[d.block == "main"]
    out = {}

    # 1. power by size of the growth effect, averaged over the four structures
    power = {}
    for n, g in main_block.groupby("n"):
        for e in sorted(set(EFFECTS)):
            cols = [f"reject_d_b{k}" for k, v in enumerate(EFFECTS) if v == e]
            power[f"n{int(n)}_effect{e:.2f}"] = float(np.mean([g[c].mean() for c in cols]))
    out["growth_power"] = power

    # 2. estimates on a bound
    at_lower = d[d.kappa <= 0.0201]
    out["kappa_on_lower_bound"] = {"count": int(len(at_lower)), "by_condition": at_lower.groupby("cell").size().to_dict()}
    out["kappa_on_upper_bound"] = int((d.kappa >= 19.9).sum())

    # 3. breakdown of the importance sampling (effective sample size below 10 for at least one student)
    bad = d.ess_min < 10
    s_all = summarise.summarise(d).set_index("cell")
    s_ok = summarise.summarise(d[~bad]).set_index("cell")
    out["breakdown"] = {
        "count": int(bad.sum()), "with_100_students": int((bad & (d.n == 100)).sum()),
        "median_ess_range": [float(d[bad].ess_median.min()), float(d[bad].ess_median.max())],
        "median_ess_elsewhere": float(d[~bad].ess_median.median()),
        "largest_change_in_a_proportion_points": float(100 * (s_all[PROPORTIONS] - s_ok[PROPORTIONS]).abs().max().max()),
        "largest_change_in_relative_bias_points": float((s_all.kappa_rel_bias_pct - s_ok.kappa_rel_bias_pct).abs().max()),
    }

    # 4. ties in the lowest-subscore heuristic (cohorts of 200; data regenerated from the stored seeds)
    ties = {}
    for cell in ("main_rho1_n200", "main_rho0_n200", "main_rho-2_n200", "main_rho-8_n200"):
        g = d[d.cell == cell]
        first, random_, tied_share, diff = [], [], [], []
        for _, r in g.iterrows():
            cfg = SimConfig(n_students=int(r.n), rho=float(r.rho_true))
            data, truth = simulate(cfg, seed=int(r.seed))
            best = leverage(truth["pi"][:, -1, :], np.asarray(cfg.weights), cfg.rho).argmax(axis=1)
            sub = data.X[:, -1].sum(axis=2)
            tied = sub == sub.min(axis=1, keepdims=True)
            m = tied.sum(axis=1)
            first.append(np.mean(sub.argmin(axis=1) == best))
            random_.append(np.mean(tied[np.arange(len(sub)), best] / m))     # expected accuracy with random tie-breaking
            tied_share.append(np.mean(m > 1))
            diff.append(abs(first[-1] - r.target_lowest))
        assert max(diff) < 1e-12, "regenerated data do not match the stored output"
        ties[cell] = {"kappa_true": float(g.kappa_true.iloc[0]), "first_listed": float(np.mean(first)),
                      "random": float(np.mean(random_)), "students_with_a_tie": float(np.mean(tied_share))}
    out["lowest_subscore_ties"] = ties

    path = os.path.join(HERE, "results", "extra_checks.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
