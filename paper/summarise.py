"""Summarise the Monte Carlo output into the tables and key numbers used in the article."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
K = 5
W_TRUE = np.array([0.15, 0.20, 0.20, 0.25, 0.20])
BETA_TRUE = np.array([0.10, 0.15, 0.20, 0.15, 0.20])


def summarise(d):
    rows = []
    for cell, g in d.groupby("cell", sort=False):
        ok = g[g.failed == 0]
        lk = np.log(ok.kappa) - np.log(ok.kappa_true)
        w = ok[[f"w{k}" for k in range(K)]].to_numpy()
        db = ok[[f"d_b{k}" for k in range(K)]].to_numpy()
        row = dict(
            cell=cell, block=g.block.iloc[0], n=int(ok.n.iloc[0]), rho=float(ok.rho_true.iloc[0]),
            kappa_true=float(ok.kappa_true.iloc[0]), reps=len(ok), failed=int((g.failed != 0).sum()),
            converged=float(ok.converged.mean()),
            kappa_median=float(ok.kappa.median()), kappa_mean=float(ok.kappa.mean()),
            kappa_p2_5=float(ok.kappa.quantile(0.025)), kappa_p97_5=float(ok.kappa.quantile(0.975)),
            kappa_rel_bias_pct=float(100 * (np.exp(lk.mean()) - 1)),
            logk_bias=float(lk.mean()), logk_bias_mcse=float(lk.std(ddof=1) / np.sqrt(len(ok))),
            logk_emp_sd=float(lk.std(ddof=1)), logk_model_se=float(ok.log_kappa_se.median()),
            logk_rmse=float(np.sqrt(np.mean(lk ** 2))),
            kappa_coverage=float(ok.cover_kappa.mean()), reject_linear=float(ok.reject_linear.mean()),
            w_abs_bias=float(np.abs(w.mean(axis=0) - W_TRUE).mean()),
            w_rmse=float(np.sqrt(((w - W_TRUE) ** 2).mean())),
            w_coverage=float(ok[[f"cover_w{k}" for k in range(K)]].to_numpy().mean()),
            slope_mean=float(ok.slope.mean()), intercept_mean=float(ok.intercept.mean()),
            sigma_mean=float(ok.sigma.mean()), severity_gap_mean=float(ok.severity_gap.mean()),
            db_bias=float((db.mean(axis=0) - BETA_TRUE).mean()),
            db_abs_bias=float(np.abs(db.mean(axis=0) - BETA_TRUE).mean()),
            db_rmse=float(np.sqrt(((db - BETA_TRUE) ** 2).mean())),
            db_coverage=float(ok[[f"cover_d_b{k}" for k in range(K)]].to_numpy().mean()),
            db_power=float(ok[[f"reject_d_b{k}" for k in range(K)]].to_numpy().mean()),
            reliability=float(ok[[f"rel{k}" for k in range(K)]].to_numpy().mean()),
            corr_a=float(ok.corr_a_mean.mean()),
            ess_median=float(ok.ess_median.mean()), ess_min=float(ok.ess_min.min()),
            seconds=float(ok.seconds.mean()),
        )
        for m in ("eag", "eaglin", "sum", "holistic"):
            row[f"r_{m}"] = float(ok[f"r_{m}"].mean())
            row[f"top_{m}"] = float(ok[f"top_{m}"].mean())
        for m in ("eag", "eaglin", "lowest"):
            row[f"target_{m}"] = float(ok[f"target_{m}"].mean())
        row["rmse_eag"] = float(ok.rmse_eag.mean())
        row["rmse_eaglin"] = float(ok.rmse_eaglin.mean())
        for extra in ("acc_eag", "acc_eaglin", "over_eag", "over_eaglin", "false_ready_eag", "false_ready_eaglin",
                      "missed_ready_eag", "missed_ready_eaglin", "ready_rate", "ppc_flag_free", "ppc_flag_lin",
                      "form_flag_free", "form_flag_lin", "link_flag"):
            if extra in ok.columns:
                row[extra] = float(ok[extra].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "montecarlo.csv")
    d = pd.read_csv(path)
    s = summarise(d)
    stem = os.path.splitext(os.path.basename(path))[0]
    s.to_csv(os.path.join(RES, f"summary_{stem}.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    cols = ["cell", "reps", "kappa_true", "kappa_median", "kappa_rel_bias_pct", "logk_emp_sd", "logk_model_se",
            "kappa_coverage", "reject_linear", "w_rmse", "w_coverage", "db_bias", "db_coverage", "db_power"]
    print(s[cols].round(3).to_string(index=False))
    cols = ["cell", "top_eag", "top_eaglin", "top_sum", "top_holistic", "r_eag", "r_eaglin", "r_sum", "r_holistic",
            "target_eag", "target_eaglin", "target_lowest", "rmse_eag", "rmse_eaglin"]
    print(s[cols].round(3).to_string(index=False))
    with open(os.path.join(RES, f"key_{stem}.json"), "w") as fh:
        json.dump({r["cell"]: r for r in s.to_dict("records")}, fh, indent=1)


if __name__ == "__main__":
    main()
