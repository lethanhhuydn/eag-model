"""Model checking: does a data set support the EAG model, and which aggregation structure does it favour?

`validate` returns a list of checks, each with the statistic, the criterion applied and a verdict
(pass / caution / fail / info), plus the tables behind them. Criteria are conventional rules of thumb,
stated explicitly so that users can apply stricter ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import chi2, norm

from .estimate import EAGFit, _summarise_aggregation, student_summaries
from .functions import rho_from_kappa, rsm_probs


@dataclass
class ValidationReport:
    checks: List[Dict]
    tables: Dict[str, object]
    profile: Optional[Dict[str, np.ndarray]]
    headline: str
    settings: Dict[str, object] = field(default_factory=dict)

    def verdict_counts(self):
        out = {"pass": 0, "caution": 0, "fail": 0, "info": 0}
        for c in self.checks:
            out[c["verdict"]] += 1
        return out

    def checks_frame(self):
        import pandas as pd

        return pd.DataFrame(self.checks)[["layer", "check", "result", "criterion", "verdict"]]


def _add(checks, layer, check, result, criterion, verdict):
    checks.append(dict(layer=layer, check=check, result=result, criterion=criterion, verdict=verdict))


def _grade(value, pass_if, caution_if):
    return "pass" if pass_if(value) else ("caution" if caution_if(value) else "fail")


def item_fit(fit: EAGFit):
    """Infit and outfit mean squares per indicator, with category counts."""
    import pandas as pd

    data = fit.data
    rows = []
    h = np.arange(data.max_score + 1)
    for k, cf in enumerate(fit.components):
        a = cf.post_mean[:, 0]
        b = cf.post_mean[:, 1] if fit.has_growth else np.zeros_like(a)
        theta = a[:, None] + b[:, None] * data.times[None, :]                       # (N, T)
        P = rsm_probs(theta, cf.delta, cf.tau)                                      # (N, T, J, m+1)
        E = (P * h).sum(axis=-1)
        W = (P * h ** 2).sum(axis=-1) - E ** 2
        X = data.X[:, :, k, :]
        obs = (X >= 0) & cf.used[None, None, :]
        resid2 = np.where(obs, (X - E) ** 2, 0.0)
        Wm = np.where(obs, W, 0.0)
        z2 = np.where(obs, resid2 / np.maximum(W, 1e-9), 0.0)
        for j in np.flatnonzero(cf.used):
            n = obs[:, :, j].sum()
            counts = [int(((X[:, :, j] == c) & obs[:, :, j]).sum()) for c in h]
            rows.append(dict(component=cf.name, indicator=data.indicators[k][j], n=int(n),
                             difficulty=float(cf.delta[j]),
                             infit=float(resid2[:, :, j].sum() / max(Wm[:, :, j].sum(), 1e-9)),
                             outfit=float(z2[:, :, j].sum() / max(n, 1)),
                             min_category_count=min(counts), category_counts=str(counts)))
    return pd.DataFrame(rows)


def _rubric_only_prediction(prob, agg):
    """Competence, mean mastery and unevenness predicted from rubric evidence alone (unweighted draws)."""
    post, C, s, _ = prob.evaluate(agg.par, agg.fix_kappa, agg.equal_weights, want="post")
    return C.mean(axis=1), prob.PI.mean(axis=1)                                     # (N, T), (N, T, K)


def _adjusted_ratings(fit, agg):
    """Mean rating per student and occasion after removing estimated rater severity (NaN where unrated)."""
    Y = fit.data.Y + agg.severity[None, None, :]
    n = np.isfinite(Y).sum(axis=2)
    return np.where(n > 0, np.nansum(Y, axis=2) / np.maximum(n, 1), np.nan)


def functional_form(fit, agg):
    """Correlation between rating residuals (rubric-only prediction) and the unevenness of the profile."""
    prob = fit.problem
    N = prob.N
    c_hat, pi_bar = _rubric_only_prediction(prob, agg)
    y_adj = _adjusted_ratings(fit, agg)
    ok = np.isfinite(y_adj)
    predicted = agg.intercept + agg.slope * c_hat[ok]
    resid = y_adj[ok] - predicted
    uneven = pi_bar.std(axis=2)[ok]
    r = float(np.corrcoef(resid, uneven)[0, 1])
    se = 1.0 / np.sqrt(max(N - 3, 1))                                 # conservative: one unit per student
    lo, hi = np.tanh(np.arctanh(r) - 1.96 * se), np.tanh(np.arctanh(r) + 1.96 * se)
    return dict(r=r, lower=float(lo), upper=float(hi), rmse=float(np.sqrt(np.mean(resid ** 2))),
                resid=resid, uneven=uneven, predicted=predicted, observed=y_adj[ok])


def link_linearity(fit, agg):
    """Cluster-robust Wald test (2 df) of quadratic and cubic terms in the rating link."""
    prob = fit.problem
    c_hat, _ = _rubric_only_prediction(prob, agg)
    y_adj = _adjusted_ratings(fit, agg)
    ok = np.isfinite(y_adj)
    student = np.repeat(np.arange(prob.N)[:, None], y_adj.shape[1], axis=1)[ok]
    z = (c_hat[ok] - c_hat[ok].mean()) / c_hat[ok].std()
    X = np.column_stack([np.ones(z.size), z, z ** 2, z ** 3])
    beta = np.linalg.lstsq(X, y_adj[ok], rcond=None)[0]
    e = y_adj[ok] - X @ beta
    bread = np.linalg.inv(X.T @ X)
    G = np.zeros((prob.N, 4))
    np.add.at(G, student, X * e[:, None])                             # score sums per student
    V = bread @ (G.T @ G) @ bread
    b = beta[2:]
    wald = float(b @ np.linalg.solve(V[2:, 2:], b))
    return dict(wald=wald, p=float(chi2.sf(wald, 2)), quadratic=float(beta[2]), cubic=float(beta[3]))


def _partial_slope(y, mean_pi, uneven):
    ok = np.isfinite(y)
    Xd = np.column_stack([np.ones(ok.sum()), mean_pi[ok], uneven[ok]])
    return float(np.linalg.lstsq(Xd, y[ok], rcond=None)[0][2])


def _ppc(fit, agg, rng, n_rep=300):
    """Posterior predictive check on the unevenness signature of holistic ratings."""
    prob, data = fit.problem, fit.data
    post, C, _, _ = prob.evaluate(agg.par, agg.fix_kappa, agg.equal_weights, want="post")
    _, pi_bar = _rubric_only_prediction(prob, agg)
    mean_pi, uneven = pi_bar.mean(axis=2), pi_bar.std(axis=2)
    y_obs = _adjusted_ratings(fit, agg)
    t_obs = _partial_slope(y_obs, mean_pi, uneven)
    mask = np.isfinite(data.Y)
    cum = np.cumsum(post, axis=1)
    cum[:, -1] = 1.0
    N, M = post.shape
    t_rep = np.empty(n_rep)
    for r in range(n_rep):
        pick = (rng.random((N, 1)) > cum).sum(axis=1).clip(0, M - 1)
        c = C[np.arange(N), pick]                                                   # (N, T)
        y = agg.intercept + agg.slope * c[:, :, None] + agg.sigma * rng.standard_normal(data.Y.shape)
        y = np.where(mask, y, np.nan)                                               # severity already removed
        n = mask.sum(axis=2)
        ybar = np.where(n > 0, np.nansum(y, axis=2) / np.maximum(n, 1), np.nan)
        t_rep[r] = _partial_slope(ybar, mean_pi, uneven)
    p = float(np.mean(t_rep <= t_obs))
    return dict(observed=t_obs, replicated_mean=float(t_rep.mean()), replicated_sd=float(t_rep.std()),
                p_value=p, two_sided=float(2 * min(p, 1 - p)))


def _cross_validate(fit, folds, rng):
    """Student-level K-fold comparison of out-of-sample log predictive density: free versus linear."""
    prob = fit.problem
    N = prob.N
    order = rng.permutation(N)
    elpd = {"free": np.zeros(N), "linear": np.zeros(N)}
    for f in range(folds):
        test = np.sort(order[f::folds])
        train = np.setdiff1d(np.arange(N), test)
        ptr, pte = prob.subset(train), prob.subset(test)
        for name, kw, start in (("linear", dict(fix_kappa=1.0), fit.models["linear"].par),
                                ("free", dict(), fit.free.par)):
            _, par, _, _, _ = ptr.fit(start=start, **kw)
            elpd[name][test] = pte.evaluate(par, want="ll", **kw)
    diff = elpd["free"] - elpd["linear"]
    return dict(elpd_free=float(elpd["free"].sum()), elpd_linear=float(elpd["linear"].sum()),
                difference=float(diff.sum()), se=float(np.sqrt(N) * diff.std(ddof=1)), folds=folds)


def profile_kappa(fit: EAGFit, grid=None):
    """Profile log-likelihood of the compensation index (other parameters re-estimated at each value)."""
    prob = fit.problem
    if grid is None:
        grid = np.unique(np.concatenate([np.geomspace(0.04, 1.0, 13), [1.25, 1.6, 2.0, fit.free.kappa]]))
    start = np.delete(fit.free.par, fit.free.par_names.index("v"))
    ll = np.empty(grid.size)
    for i in np.argsort(np.abs(np.log(grid) - np.log(fit.free.kappa))):
        _, _, f, _, _ = prob.fit(fix_kappa=float(grid[i]), start=start)
        ll[i] = f
    return dict(kappa=grid, loglik=ll, max=fit.free.loglik)


def _nz(x, digits=2):
    """Round for display so that a tiny negative number is not printed as -0.00."""
    v = round(float(x), digits)
    return 0.0 if v == 0 else v


def validate(fit: EAGFit, threshold: Optional[float] = None, cv_folds: int = 5, ppc_reps: int = 300,
             profile: bool = True, seed: int = 0) -> ValidationReport:
    """Run all model checks on a fitted EAG model."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    data, prob = fit.data, fit.problem
    free, lin = fit.free, fit.models["linear"]
    N, T, K, J = data.X.shape
    checks: List[Dict] = []
    tables: Dict[str, object] = {}

    # ------------------------------------------------------------------ data
    _add(checks, "Data", "Number of students", f"{N}", "at least 100 for stable estimates of the compensation index",
         _grade(N, lambda v: v >= 100, lambda v: v >= 50))
    n_rated = int(np.isfinite(data.Y).any(axis=2).sum())
    _add(checks, "Data", "Rated student-occasions", f"{n_rated} with {data.n_raters} rater(s)",
         "at least 200 holistic design reviews", _grade(n_rated, lambda v: v >= 200, lambda v: v >= 100))
    miss = float((data.X < 0).mean())
    _add(checks, "Data", "Missing rubric scores", f"{100 * miss:.1f}%", "below 20%",
         _grade(miss, lambda v: v < 0.20, lambda v: v < 0.40))

    # ------------------------------------------------------------------ evidence layer
    items = item_fit(fit)
    tables["items"] = items
    bad = items[(items.infit < 0.5) | (items.infit > 1.5) | (items.outfit < 0.5) | (items.outfit > 1.5)]
    worst = items.loc[(items[["infit", "outfit"]] - 1).abs().max(axis=1).idxmax()]
    _add(checks, "Evidence", "Indicator fit (infit and outfit mean squares)",
         f"{len(items) - len(bad)} of {len(items)} indicators within range; largest deviation "
         f"{worst.indicator} (infit {worst.infit:.2f}, outfit {worst.outfit:.2f})",
         "mean squares between 0.5 and 1.5", "pass" if bad.empty else ("caution" if len(bad) <= 0.2 * len(items) else "fail"))
    sparse = int((items.min_category_count < 10).sum())
    _add(checks, "Evidence", "Use of rubric levels", f"{sparse} indicator(s) with a level used fewer than 10 times",
         "every level of every indicator used at least 10 times", "pass" if sparse == 0 else "caution")
    disordered = [cf.name for cf in fit.components if np.any(np.diff(cf.tau) < 0)]
    _add(checks, "Evidence", "Ordering of rubric thresholds",
         "ordered in all components" if not disordered else "disordered in: " + ", ".join(disordered),
         "thresholds increase with the rubric level", "pass" if not disordered else "caution")
    rel = np.array([cf.reliability.mean() for cf in fit.components])
    _add(checks, "Evidence", "Reliability of component proficiencies",
         "; ".join(f"{cf.name} {r:.2f}" for cf, r in zip(fit.components, rel)),
         "EAP reliability of at least 0.70 in every component (below 0.50 fails)",
         _grade(rel.min(), lambda v: v >= 0.70, lambda v: v >= 0.50))
    comp_rows = []
    for cf, r in zip(fit.components, rel):
        row = dict(component=cf.name, reliability=float(r), initial_mean=cf.natural["mu_a"],
                   initial_sd=cf.natural["sd_a"])
        if fit.has_growth:
            row.update(growth_mean=cf.natural["mu_b"], growth_mean_se=cf.se["mu_b"], growth_sd=cf.natural["sd_b"])
            if fit.has_group:
                z = cf.natural["d_b"] / max(cf.se["d_b"], 1e-12)
                row.update(group_effect_on_growth=cf.natural["d_b"], group_effect_se=cf.se["d_b"],
                           group_effect_p=float(2 * norm.sf(abs(z))))
        comp_rows.append(row)
    tables["components"] = pd.DataFrame(comp_rows)
    corr = fit.latent_correlations()
    labels = [f"initial: {c}" for c in data.components] + ([f"growth: {c}" for c in data.components] if fit.has_growth else [])
    tables["latent_correlations"] = pd.DataFrame(corr, index=labels, columns=labels)
    rmax = float(np.abs(corr[:K, :K][np.triu_indices(K, 1)]).max())
    _add(checks, "Evidence", "Distinctness of components", f"largest correlation between initial proficiencies {rmax:.2f}",
         "below 0.90, otherwise two components cannot be told apart",
         _grade(rmax, lambda v: v < 0.90, lambda v: v < 0.95))

    # ------------------------------------------------------------------ raters
    R = data.n_raters
    rater_rows = []
    for r in range(R):
        rater_rows.append(dict(rater=data.raters[r], ratings=int(np.isfinite(data.Y[:, :, r]).sum()),
                               severity=float(free.severity[r]), severity_se=float(free.severity_se[r])))
    tables["raters"] = pd.DataFrame(rater_rows)
    if R > 1:
        cors = []
        for r1 in range(R):
            for r2 in range(r1 + 1, R):
                ok = np.isfinite(data.Y[:, :, r1]) & np.isfinite(data.Y[:, :, r2])
                if ok.sum() > 10:
                    cors.append(np.corrcoef(data.Y[:, :, r1][ok], data.Y[:, :, r2][ok])[0, 1])
        if cors:
            _add(checks, "Evidence", "Agreement between raters", f"mean correlation between raters {np.mean(cors):.2f}",
                 "at least 0.60", _grade(float(np.mean(cors)), lambda v: v >= 0.60, lambda v: v >= 0.40))
        gap = float(free.severity.max() - free.severity.min())
        _add(checks, "Evidence", "Rater severity", f"range of severity {gap:.1f} rating points "
             f"(residual SD {free.sigma:.1f})", "reported and adjusted for by the model", "info")

    # ------------------------------------------------------------------ computation
    _add(checks, "Computation", "Importance sampling", f"smallest effective sample size {fit.ess.min():.0f} "
         f"of {fit.settings.get('n_prop', 0)} proposals", "at least 30 for every student",
         _grade(float(fit.ess.min()), lambda v: v >= 30, lambda v: v >= 10))
    conv = all(cf.converged for cf in fit.components) and free.converged and lin.converged
    _add(checks, "Computation", "Convergence of all stages", "converged" if conv else "not converged",
         "all optimisers report convergence", "pass" if conv else "fail")

    # ------------------------------------------------------------------ aggregation layer
    width = free.kappa_ci[1] / max(free.kappa_ci[0], 1e-9)
    _add(checks, "Aggregation", "Precision of the compensation index",
         f"kappa = {free.kappa:.2f} (95% CI {free.kappa_ci[0]:.2f} to {free.kappa_ci[1]:.2f}); "
         f"rho = {free.rho:.2f}", "upper limit less than four times the lower limit",
         _grade(width, lambda v: v < 4, lambda v: v < 10))
    lr = fit.lr_linear
    if lr["p"] < 0.05:
        if free.kappa < 1:
            structure = "partly non-compensatory: weak components hold competence back"
        else:
            structure = "more than compensatory: strong components count more than proportionally"
    else:
        structure = "compatible with fully compensatory (sum-score) aggregation"
    _add(checks, "Aggregation", "Test of sum-score aggregation (kappa = 1)",
         f"likelihood ratio {lr['statistic']:.1f}, df 1, p = {lr['p']:.3g}; structure {structure}",
         "p < 0.05 rejects the sum-score assumption", "info")
    n_obs = N
    comp = []
    for name, m in fit.models.items():
        comp.append(dict(model=m.label, kappa=m.kappa, loglik=m.loglik, parameters=m.n_par, AIC=m.aic, BIC=m.bic(n_obs)))
    tables["models"] = pd.DataFrame(comp).sort_values("BIC").reset_index(drop=True)
    if "equal" in fit.models:
        lr_w = max(2 * (free.loglik - fit.models["equal"].loglik), 0.0)
        p_w = float(chi2.sf(lr_w, K - 1))
        _add(checks, "Aggregation", "Test of equal component weights",
             f"likelihood ratio {lr_w:.1f}, df {K - 1}, p = {p_w:.3g}", "p < 0.05 indicates unequal weights", "info")
    tables["weights"] = pd.DataFrame(dict(component=data.components, weight=free.weights, se=free.weights_se,
                                          lower=np.clip(free.weights - 1.96 * free.weights_se, 0, 1),
                                          upper=np.clip(free.weights + 1.96 * free.weights_se, 0, 1)))
    if free.kappa < 0.15:
        _add(checks, "Aggregation", "Interpretability of weights", "near-conjunctive structure",
             "when kappa is below about 0.15 competence follows the weakest component and weights carry little information",
             "caution")

    # functional form: do rubric-only predictions leave an unevenness pattern in the ratings?
    form = {"linear": functional_form(fit, lin), "free": functional_form(fit, free)}
    tables["functional_form"] = form
    f_ = form["free"]
    _add(checks, "Aggregation", "Functional form (residuals versus profile unevenness)",
         f"fitted model r = {_nz(f_['r']):.2f} (95% CI {_nz(f_['lower']):.2f} to {_nz(f_['upper']):.2f}); "
         f"sum-score model r = {_nz(form['linear']['r']):.2f}",
         "confidence interval of the fitted model includes zero",
         "pass" if f_["lower"] <= 0 <= f_["upper"] else "caution")
    link = link_linearity(fit, free)
    tables["link_linearity"] = link
    _add(checks, "Aggregation", "Linearity of the rating link",
         f"Wald test of curvature {link['wald']:.1f}, df 2, p = {link['p']:.3g}",
         "p of at least 0.01; curvature biases the compensation index", "pass" if link["p"] >= 0.01 else "caution")
    ppc = _ppc(fit, free, rng, ppc_reps)
    ppc_lin = _ppc(fit, lin, rng, ppc_reps)
    tables["ppc"] = dict(free=ppc, linear=ppc_lin)
    _add(checks, "Aggregation", "Posterior predictive check (unevenness signature)",
         f"fitted model p = {ppc['p_value']:.2f}; sum-score model p = {ppc_lin['p_value']:.2f}",
         "predictive p-value of the fitted model between 0.05 and 0.95",
         "pass" if 0.05 <= ppc["p_value"] <= 0.95 else "caution")
    if cv_folds and cv_folds > 1 and N >= 2 * cv_folds:
        cv = _cross_validate(fit, cv_folds, rng)
        tables["cross_validation"] = cv
        z = cv["difference"] / max(cv["se"], 1e-9)
        better = "fitted model predicts better" if z > 2 else ("sum-score model predicts better" if z < -2 else "no clear difference")
        _add(checks, "Aggregation", f"{cv_folds}-fold cross-validation against the sum-score model",
             f"difference in out-of-sample log predictive density {cv['difference']:.1f} (SE {cv['se']:.1f}); {better}",
             "difference larger than two standard errors", "info")

    # ------------------------------------------------------------------ growth layer
    if fit.has_growth:
        gm = np.array([cf.natural["mu_b"] for cf in fit.components])
        gse = np.array([cf.se["mu_b"] for cf in fit.components])
        n_pos = int((gm - 1.96 * gse > 0).sum())
        _add(checks, "Growth", "Growth across design cycles", f"{n_pos} of {K} components show reliable growth; "
             f"mean growth {gm.mean():.2f} logits per cycle", "reported", "info")
        if fit.has_group:
            eff = np.array([cf.natural["d_b"] for cf in fit.components])
            ese = np.array([cf.se["d_b"] for cf in fit.components])
            n_sig = int((np.abs(eff) > 1.96 * ese).sum())
            _add(checks, "Growth", f"Difference in growth between groups ({data.group_labels[1]} minus {data.group_labels[0]})",
                 f"{n_sig} of {K} components differ reliably; mean difference {eff.mean():.2f} logits per cycle",
                 "reported", "info")

    # ------------------------------------------------------------------ student table
    summ = student_summaries(prob, free, threshold)
    st = pd.DataFrame({"student": data.students})
    if fit.has_group:
        st["group"] = [data.group_labels[g] for g in data.group]
    for t_i, lab in enumerate(data.occasions):
        st[f"competence_t{lab}"] = summ["C_mean"][:, t_i]
    st["competence_sd_last"] = summ["C_sd"][:, -1]
    for k, c in enumerate(data.components):
        st[f"mastery_{c}"] = summ["pi_mean"][:, -1, k]
    st["teach_next"] = [data.components[i] for i in summ["target"]]
    if threshold is not None:
        st["probability_ready"] = summ["p_ready"]
        if "cycles_to_ready" in summ:
            st["cycles_to_ready"] = summ["cycles_to_ready"]
    tables["students"] = st
    prof = profile_kappa(fit) if profile else None

    counts = {"pass": 0, "caution": 0, "fail": 0, "info": 0}
    for c in checks:
        counts[c["verdict"]] += 1
    if counts["fail"]:
        overall = "The data do not yet support the model: resolve the failed checks before interpreting results."
    elif counts["caution"]:
        overall = "The model is usable with care: read the cautions before interpreting results."
    else:
        overall = "All checks passed: the data support the model."
    headline = (f"{overall} Compensation index kappa = {free.kappa:.2f} "
                f"(95% CI {free.kappa_ci[0]:.2f} to {free.kappa_ci[1]:.2f}); the structure is {structure}.")
    return ValidationReport(checks=checks, tables=tables, profile=prof, headline=headline,
                            settings=dict(threshold=threshold, cv_folds=cv_folds, ppc_reps=ppc_reps, seed=seed))
