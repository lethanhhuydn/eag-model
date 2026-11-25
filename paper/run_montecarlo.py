"""Monte Carlo study reported in the paper (aims, data-generating mechanisms, estimands, methods and
performance measures are described in the Methods section).

Usage:  python paper/run_montecarlo.py --reps 200 --sens-reps 100 --workers 2 --out paper/results/montecarlo.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eagmodel import SimConfig, fit, simulate                       # noqa: E402
from eagmodel.estimate import student_summaries                     # noqa: E402
from eagmodel.functions import kappa_from_rho, leverage             # noqa: E402
from eagmodel.validate import _ppc, functional_form, link_linearity # noqa: E402

READY = 0.60      # readiness threshold on the competence scale used for absolute decisions

RHOS = (1.0, 0.0, -2.0, -8.0)
SIZES = (100, 200, 400)


def cells():
    """Main grid (4 x 3) followed by sensitivity cells at N = 200, rho = -2."""
    out = []
    for rho in RHOS:
        for n in SIZES:
            out.append(dict(cell=f"main_rho{rho:g}_n{n}", block="main", n_students=n, rho=rho))
    base = dict(n_students=200, rho=-2.0)
    out.append(dict(cell="sens_corr0.2", block="sensitivity", corr_a=0.2, **base))
    out.append(dict(cell="sens_corr0.8", block="sensitivity", corr_a=0.8, **base))
    out.append(dict(cell="sens_noise12", block="sensitivity", h_sd=12.0, **base))
    out.append(dict(cell="sens_final_only", block="sensitivity", holistic_occasions=(3,), **base))
    out.append(dict(cell="sens_logit_link", block="sensitivity", link="logit", **base))
    return out


def top_share(score, truth, frac=0.30):
    """Share of the truly top-`frac` students who are also in the top `frac` on `score`."""
    n = truth.size
    k = max(int(round(frac * n)), 1)
    top_true = set(np.argsort(-truth, kind="stable")[:k])
    top_est = set(np.argsort(-score, kind="stable")[:k])
    return len(top_true & top_est) / k


def one_replication(task):
    cell, rep, seed = task
    spec = {k: v for k, v in cell.items() if k not in ("cell", "block")}
    cfg = SimConfig(**spec)
    data, truth = simulate(cfg, seed=seed)
    t0 = time.time()
    try:
        f = fit(data, seed=seed + 7)
    except Exception as exc:                                         # keep the run alive and record the failure
        return dict(cell=cell["cell"], block=cell["block"], rep=rep, seed=seed, failed=1, error=str(exc)[:80])
    m, lin = f.free, f.models["linear"]
    K = data.n_components
    w_true = np.asarray(cfg.weights)
    kappa_true = float(kappa_from_rho(cfg.rho))
    row = dict(cell=cell["cell"], block=cell["block"], rep=rep, seed=seed, failed=0, error="",
               n=cfg.n_students, rho_true=cfg.rho, kappa_true=kappa_true,
               kappa=m.kappa, kappa_lo=m.kappa_ci[0], kappa_hi=m.kappa_ci[1], rho=m.rho,
               log_kappa_se=float(np.sqrt(max(m.cov[m.par_names.index("v"), m.par_names.index("v")], 0))),
               cover_kappa=int(m.kappa_ci[0] <= kappa_true <= m.kappa_ci[1]),
               lr=f.lr_linear["statistic"], lr_p=f.lr_linear["p"], reject_linear=int(f.lr_linear["p"] < 0.05),
               intercept=m.intercept, slope=m.slope, sigma=m.sigma,
               severity_gap=float(m.severity[-1] - m.severity[0]),
               converged=int(m.converged and lin.converged), n_eval=m.n_eval + lin.n_eval,
               ess_median=float(np.median(f.ess)), ess_min=float(f.ess.min()), em_iter=f.em_iterations,
               seconds=time.time() - t0)
    for k in range(K):
        row[f"w{k}"] = m.weights[k]
        row[f"w{k}_se"] = m.weights_se[k]
        row[f"cover_w{k}"] = int(abs(m.weights[k] - w_true[k]) <= 1.96 * m.weights_se[k])
        cf = f.components[k]
        row[f"mu_b{k}"] = cf.natural["mu_b"]
        row[f"d_b{k}"] = cf.natural["d_b"]
        row[f"d_b{k}_se"] = cf.se["d_b"]
        row[f"cover_d_b{k}"] = int(abs(cf.natural["d_b"] - cfg.beta[k]) <= 1.96 * cf.se["d_b"])
        row[f"reject_d_b{k}"] = int(abs(cf.natural["d_b"]) > 1.96 * cf.se["d_b"])
        row[f"rel{k}"] = float(cf.reliability.mean())
    corr = f.latent_correlations()
    row["corr_a_mean"] = float(corr[:K, :K][np.triu_indices(K, 1)].mean())
    # ---- consequences for decisions at the final occasion
    c_true = truth["C"][:, -1]
    s_free = f.student
    s_lin = student_summaries(f.problem, lin)
    scores = dict(eag=s_free["C_mean"][:, -1], eaglin=s_lin["C_mean"][:, -1],
                  sum=data.X[:, -1].sum(axis=(1, 2)).astype(float),
                  holistic=np.nanmean(data.Y[:, np.flatnonzero(np.isfinite(data.Y).any(axis=(0, 2)))[-1]], axis=1))
    for name, sc in scores.items():
        row[f"r_{name}"] = float(np.corrcoef(sc, c_true)[0, 1])
        row[f"top_{name}"] = top_share(sc, c_true)
    row["rmse_eag"] = float(np.sqrt(np.mean((scores["eag"] - c_true) ** 2)))
    row["rmse_eaglin"] = float(np.sqrt(np.mean((scores["eaglin"] - c_true) ** 2)))
    lev_true = leverage(truth["pi"][:, -1, :], w_true, cfg.rho).argmax(axis=1)
    row["target_eag"] = float(np.mean(s_free["target"] == lev_true))
    row["target_eaglin"] = float(np.mean(s_lin["target"] == lev_true))
    sub = data.X[:, -1].sum(axis=2)                                   # lowest rubric subscore heuristic
    row["target_lowest"] = float(np.mean(sub.argmin(axis=1) == lev_true))
    # ---- absolute readiness decisions and overstatement of uneven profiles
    ready = c_true >= READY
    row["ready_rate"] = float(ready.mean())
    uneven = truth["pi"][:, -1, :].std(axis=1)
    top = uneven >= np.quantile(uneven, 2 / 3)
    for name in ("eag", "eaglin"):
        pred = scores[name] >= READY
        row[f"acc_{name}"] = float(np.mean(pred == ready))
        row[f"false_ready_{name}"] = float((pred & ~ready).sum() / max(pred.sum(), 1))
        row[f"missed_ready_{name}"] = float((~pred & ready).sum() / max(ready.sum(), 1))
        row[f"over_{name}"] = float(np.mean(scores[name][top] - c_true[top]))
    # ---- operating characteristics of the model checks
    rng = np.random.default_rng(seed + 13)
    for name, agg in (("free", m), ("lin", lin)):
        p = _ppc(f, agg, rng, 200)["p_value"]
        ff = functional_form(f, agg)
        row[f"ppc_p_{name}"] = p
        row[f"ppc_flag_{name}"] = int(not 0.05 <= p <= 0.95)
        row[f"form_r_{name}"] = ff["r"]
        row[f"form_flag_{name}"] = int(not ff["lower"] <= 0 <= ff["upper"])
    link = link_linearity(f, m)
    row["link_p"] = link["p"]
    row["link_flag"] = int(link["p"] < 0.01)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--sens-reps", type=int, default=100, help="replications for the sensitivity cells")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="paper/results/montecarlo.csv")
    ap.add_argument("--only", default="", help="comma-separated cell names or block name")
    ap.add_argument("--seed", type=int, default=20261125)
    args = ap.parse_args()
    all_cells = cells()
    if args.only:
        keep = set(args.only.split(","))
        all_cells = [c for c in all_cells if c["cell"] in keep or c["block"] in keep]
    names = [c["cell"] for c in cells()]
    tasks = []
    for block, reps in (("main", args.reps), ("sensitivity", args.sens_reps)):      # main grid first
        tasks += [(c, r, args.seed + 10000 * names.index(c["cell"]) + r)
                  for r in range(reps) for c in all_cells if c["block"] == block]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            done = {(r["cell"], int(r["rep"])) for r in csv.DictReader(fh)}
    tasks = [t for t in tasks if (t[0]["cell"], t[1]) not in done]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"{len(tasks)} replications to run ({len(done)} already stored)", flush=True)
    if not tasks:
        return
    from multiprocessing import Pool

    fields, t0, n = None, time.time(), 0
    with Pool(args.workers) as pool, open(args.out, "a", newline="") as fh:
        writer = None
        for row in pool.imap_unordered(one_replication, tasks, chunksize=1):
            if writer is None:
                if row.get("failed"):                       # wait for the first complete row to fix the columns
                    print("failed replication:", row, flush=True)
                    continue
                fields = list(row.keys())
                if done:                                    # appending to an existing file: reuse its header
                    with open(args.out) as old:
                        fields = next(csv.reader(old))
                writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                if not done:
                    writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fields})
            fh.flush()
            n += 1
            if n % 50 == 0:
                print(f"{n}/{len(tasks)} done, {time.time() - t0:.0f}s", flush=True)
    print(f"finished {n} replications in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
