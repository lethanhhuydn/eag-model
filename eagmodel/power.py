"""Sample-size planning by simulation."""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from .estimate import fit
from .functions import kappa_from_rho
from .simulate import SimConfig, simulate


def power_analysis(config: Optional[SimConfig] = None, sizes: Sequence[int] = (100, 200, 400), reps: int = 50,
                   alpha: float = 0.05, seed: int = 1, progress=None):
    """Estimate, for each sample size, the power to reject sum-score aggregation and the precision of kappa.

    Returns a pandas data frame with one row per sample size.
    """
    import pandas as pd

    cfg0 = config or SimConfig()
    k_true = float(kappa_from_rho(cfg0.rho))
    rows, done = [], 0
    for n in sizes:
        cfg = replace(cfg0, n_students=int(n))
        est, rej, cover, eff = [], [], [], []
        for r in range(reps):
            data, _ = simulate(cfg, seed=seed + 100000 * int(n) + r)
            f = fit(data, seed=seed + r)
            est.append(f.free.kappa)
            rej.append(f.lr_linear["p"] < alpha)
            cover.append(f.free.kappa_ci[0] <= k_true <= f.free.kappa_ci[1])
            eff.append(np.mean([abs(c.natural.get("d_b", np.nan)) > 1.96 * c.se.get("d_b", np.nan)
                                for c in f.components]) if f.has_group and f.has_growth else np.nan)
            done += 1
            if progress:
                progress(done / (len(sizes) * reps))
        est = np.array(est)
        rows.append(dict(n_students=int(n), true_kappa=k_true, median_kappa=float(np.median(est)),
                         kappa_p2_5=float(np.percentile(est, 2.5)), kappa_p97_5=float(np.percentile(est, 97.5)),
                         rejects_sum_score=float(np.mean(rej)), ci_coverage=float(np.mean(cover)),
                         power_growth_effect=float(np.nanmean(eff)) if np.isfinite(eff).any() else np.nan,
                         replications=reps))
    return pd.DataFrame(rows)
