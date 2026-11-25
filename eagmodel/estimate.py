"""Three-stage estimation of the EAG model.

Stage 1  Evidence and growth, one component at a time.
         1a  Rating-scale-model item parameters by conditional maximum likelihood (CML).
         1b  Latent growth parameters by marginal maximum likelihood on a two-dimensional grid.
Stage 2  Joint distribution of all component proficiencies and growth rates by an
         importance-sampling EM algorithm (proposals come from stage 1b).
Stage 3  Aggregation function (weights, compensation index, holistic link, rater severity)
         by simulated maximum likelihood with analytic gradients.

Stages are estimated in sequence so that the meaning of the component scales is fixed by
the rubric evidence alone and is not bent towards the holistic ratings.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from scipy.stats import chi2

from .data import EAGData
from .functions import (RHO_EPS, kappa_from_rho, log_mastery, rho_from_kappa, rsm_cumtau,
                        rsm_log_normaliser, time_to_readiness)

LOG2PI = np.log(2.0 * np.pi)


# ===================================================================== stage 1a: CML
def cml_rsm(Xk: np.ndarray, m: int):
    """Conditional ML estimates of rating-scale-model parameters for one component.

    Xk is an (n_records, J) integer array with -1 for missing scores. Returns (delta, tau)
    with sum(delta) = 0 and sum(tau) = 0. The estimates do not depend on any assumption
    about how proficiency is distributed or how it changes over occasions.
    """
    Xk = np.asarray(Xk)
    J = Xk.shape[1]
    obs = Xk >= 0
    used = obs.any(axis=0)
    items = np.flatnonzero(used)
    Ju = items.size
    if Ju < 2:
        raise ValueError("Each component needs at least two scored indicators")
    Xu, ou = Xk[:, items], obs[:, items]
    h = np.arange(m + 1)
    patterns, inv = np.unique(ou, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    groups = []
    for p, pat in enumerate(patterns):
        if pat.sum() < 2:
            continue
        rows = Xu[inv == p][:, pat]
        counts = np.stack([(rows == c).sum(axis=0) for c in h], axis=1).astype(float)   # (Jp, m+1)
        n_s = np.bincount(rows.sum(axis=1), minlength=pat.sum() * m + 1).astype(float)
        groups.append((np.flatnonzero(pat), counts, n_s))
    n_tau = max(m - 1, 0)

    def unpack(par):
        d = np.append(par[:Ju - 1], -par[:Ju - 1].sum())
        t = np.append(par[Ju - 1:], -par[Ju - 1:].sum()) if n_tau else np.zeros(m)
        return d, t

    def nll(par):
        d, t = unpack(par)
        logeps = -np.outer(d, h) - rsm_cumtau(t)[None, :]
        val = 0.0
        for idx, counts, n_s in groups:
            le = logeps[idx]
            val += (counts * le).sum()
            poly = np.ones(1)
            for row in np.exp(le):
                poly = np.convolve(poly, row)
            val -= (n_s * np.log(poly)).sum()
        return -val + 1e-4 * float(par @ par)

    res = minimize(nll, np.zeros(Ju - 1 + n_tau), method="BFGS", options={"gtol": 1e-6})
    d, t = unpack(res.x)
    delta = np.zeros(J)
    delta[items] = d
    return delta, t, used


# ===================================================================== helpers
class _ATable:
    """Fast evaluation of the summed log-normaliser A(theta) = sum_j A_j(theta) by interpolation."""

    LO, HI, N = -20.0, 20.0, 8001

    def __init__(self, delta, tau, used):
        self.grid = np.linspace(self.LO, self.HI, self.N)
        self.m = len(tau)
        self.used = np.asarray(used, bool)
        A = rsm_log_normaliser(self.grid, delta, tau)         # (N, J)
        self.items = A * self.used[None, :]
        self.total_table = self.items.sum(axis=1)

    def _interp(self, theta, table, n_items):
        out = np.interp(theta, self.grid, table)
        hi = theta > self.HI
        if np.any(hi):
            out = out + hi * (theta - self.HI) * self.m * n_items
        return out

    def total(self, theta):
        return self._interp(theta, self.total_table, self.used.sum())

    def item(self, j, theta):
        return self._interp(theta, self.items[:, j], 1.0)


def _bvn_logpdf(a, b, ma, mb, sa, sb, r):
    za, zb = (a - ma) / sa, (b - mb) / sb
    return (-np.log(2 * np.pi * sa * sb) - 0.5 * np.log1p(-r * r)
            - (za * za - 2 * r * za * zb + zb * zb) / (2 * (1 - r * r)))


def _uvn_logpdf(a, ma, sa):
    return -0.5 * LOG2PI - np.log(sa) - 0.5 * ((a - ma) / sa) ** 2


@dataclass
class ComponentFit:
    name: str
    delta: np.ndarray
    tau: np.ndarray
    used: np.ndarray
    par_names: List[str]
    psi: np.ndarray            # unconstrained estimates
    cov_psi: np.ndarray        # OPG covariance of psi
    natural: Dict[str, float]  # mu_a, d_a, mu_b, d_b, sd_a, sd_b, r_ab
    se: Dict[str, float]
    post_mean: np.ndarray      # (N, dpc)
    post_cov: np.ndarray       # (N, dpc, dpc)
    loglik: float
    n_nodes: int
    reliability: np.ndarray    # EAP reliability of theta at each occasion
    converged: bool


# ===================================================================== stage 1b: growth on a grid
def _fit_component(Xk, times, group, m, name, has_group, has_growth):
    """Xk: (N, T, J) scores for one component (-1 missing)."""
    N, T, J = Xk.shape
    delta, tau, used = cml_rsm(Xk.reshape(-1, J), m)
    obs = (Xk >= 0) & used[None, None, :]
    S = np.where(obs, Xk, 0).sum(axis=2).astype(float)          # (N, T)
    n_obs = obs.sum(axis=2)
    complete = bool((n_obs == used.sum()).all())
    obs_f = obs.astype(float)
    tbar = float(times.mean()) if has_growth else 0.0
    tc = times - tbar

    # ---- crude starting values from logit proportions
    pr = (S + 0.5) / (n_obs * m + 1.0)
    th = np.where(n_obs > 0, 1.3 * np.log(pr / (1 - pr)), np.nan)
    if has_growth:
        ok = np.isfinite(th)
        cnt = ok.sum(axis=1)
        tm = np.where(ok, tc, 0).sum(1) / np.maximum(cnt, 1)
        ym = np.where(ok, th, 0).sum(1) / np.maximum(cnt, 1)
        sxx = np.where(ok, (tc - tm[:, None]) ** 2, 0).sum(1)
        sxy = np.where(ok, (tc - tm[:, None]) * (np.nan_to_num(th) - ym[:, None]), 0).sum(1)
        slope = np.where(sxx > 0, sxy / np.where(sxx > 0, sxx, 1), np.nan)
        level = ym - np.nan_to_num(slope) * tm
        mc, sc = np.nanmean(level), max(np.nanstd(level), 0.3)
        mb, sb = np.nanmean(slope), max(np.nanstd(slope), 0.1)
    else:
        mc, sc = np.nanmean(th), max(np.nanstd(th), 0.3)
        mb, sb = 0.0, 0.0

    G = 2 if has_group else 1
    gidx = [np.flatnonzero(group == g) for g in range(G)]

    # ---- parameter bookkeeping
    names = ["mu_a"] + (["d_a"] if has_group else [])
    if has_growth:
        names += ["mu_b"] + (["d_b"] if has_group else [])
    names += ["log_sd_a"] + (["log_sd_b", "atanh_r"] if has_growth else [])
    P = len(names)
    ix = {n: i for i, n in enumerate(names)}

    def lp_nodes(psi, a_n, b_n):
        out = np.empty((G, a_n.size))
        sa = np.exp(psi[ix["log_sd_a"]])
        for g in range(G):
            ma = psi[ix["mu_a"]] + (psi[ix["d_a"]] * g if has_group else 0.0)
            if has_growth:
                mb_ = psi[ix["mu_b"]] + (psi[ix["d_b"]] * g if has_group else 0.0)
                out[g] = _bvn_logpdf(a_n, b_n, ma, mb_, sa, np.exp(psi[ix["log_sd_b"]]),
                                     np.tanh(psi[ix["atanh_r"]]))
            else:
                out[g] = _uvn_logpdf(a_n, ma, sa)
        return out

    def norm(lp):
        return lp - logsumexp(lp, axis=1, keepdims=True)

    def make_nodes(c_lo, c_hi, qc, b_lo, b_hi, qb):
        c = np.linspace(c_lo, c_hi, qc)
        if has_growth:
            b = np.linspace(b_lo, b_hi, qb)
            cc, bb = np.meshgrid(c, b, indexing="ij")
            cc, bb = cc.ravel(), bb.ravel()
        else:
            cc, bb = c, np.zeros_like(c)
        a_n = cc - bb * tbar
        theta = a_n[:, None] + bb[:, None] * times[None, :]            # (Q, T)
        A = rsm_log_normaliser(theta, delta, tau) * used[None, None, :]  # (Q, T, J)
        if complete:
            L = S @ theta.T - A.sum(axis=(1, 2))[None, :]
        else:
            L = S @ theta.T - np.einsum("ntj,qtj->nq", obs_f, A)
        geom = dict(c_lo=c_lo, c_hi=c_hi, b_lo=b_lo, b_hi=b_hi, h_c=(c_hi - c_lo) / max(qc - 1, 1),
                    h_b=(b_hi - b_lo) / max(qb - 1, 1) if has_growth else 0.0)
        return a_n, bb, L, geom

    def box(geom):
        """Keep means inside the grid and standard deviations above its resolution."""
        half_c = 0.5 * (geom["c_hi"] - geom["c_lo"])
        half_b = 0.5 * (geom["b_hi"] - geom["b_lo"])
        out = []
        for n_ in names:
            if n_ == "mu_a":
                out.append((geom["c_lo"] - max(geom["b_hi"], 0) * tbar, geom["c_hi"] - min(geom["b_lo"], 0) * tbar))
            elif n_ == "d_a":
                out.append((-half_c, half_c))
            elif n_ == "mu_b":
                out.append((geom["b_lo"], geom["b_hi"]))
            elif n_ == "d_b":
                out.append((-half_b, half_b))
            elif n_ == "log_sd_a":
                out.append((np.log(max(0.05, 0.6 * geom["h_c"])), np.log(10.0)))
            elif n_ == "log_sd_b":
                out.append((np.log(max(0.02, 0.6 * geom["h_b"])), np.log(5.0)))
            else:
                out.append((-2.5, 2.5))
        return out

    def run(psi0, a_n, b_n, L, maxiter, geom):
        eps = 1e-5

        def fun(psi):
            lpn = norm(lp_nodes(psi, a_n, b_n))
            dl = np.empty((P, G, a_n.size))
            for p in range(P):
                e = np.zeros(P)
                e[p] = eps
                dl[p] = (norm(lp_nodes(psi + e, a_n, b_n)) - norm(lp_nodes(psi - e, a_n, b_n))) / (2 * eps)
            ll, grad = 0.0, np.zeros(P)
            for g in range(G):
                M = L[gidx[g]] + lpn[g]
                mx = M.max(axis=1, keepdims=True)
                E = np.exp(M - mx)
                s = E.sum(axis=1, keepdims=True)
                ll += float((np.log(s) + mx).sum())
                grad += dl[:, g, :] @ (E / s).sum(axis=0)
            return -ll, -grad

        bounds = box(geom)
        psi0 = np.clip(psi0, [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(fun, psi0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6})
        return res

    def posterior(psi, a_n, b_n, L):
        lpn = norm(lp_nodes(psi, a_n, b_n))
        post = np.empty_like(L)
        ll = 0.0
        for g in range(G):
            M = L[gidx[g]] + lpn[g]
            mx = M.max(axis=1, keepdims=True)
            E = np.exp(M - mx)
            s = E.sum(axis=1, keepdims=True)
            post[gidx[g]] = E / s
            ll += float((np.log(s) + mx).sum())
        return post, lpn, ll

    # ---- pass 1: wide, coarse grid
    psi0 = np.zeros(P)
    psi0[ix["mu_a"]] = mc - mb * tbar
    psi0[ix["log_sd_a"]] = np.log(max(0.5, 0.8 * sc))
    if has_growth:
        psi0[ix["mu_b"]] = mb
        psi0[ix["log_sd_b"]] = np.log(max(0.1, 0.5 * sb))
    a_n, b_n, L, geom = make_nodes(mc - 4 * sc - 1.5, mc + 4 * sc + 1.5, 31,
                                   mb - 4 * sb - 0.5, mb + 4 * sb + 0.5, 21)
    res = run(psi0, a_n, b_n, L, 200, geom)
    psi = res.x
    post, _, _ = posterior(psi, a_n, b_n, L)

    # ---- pass 2: grid matched to the estimated distribution and to posterior widths
    def moments(post, a_n, b_n):
        ea, eb = post @ a_n, post @ b_n
        vaa = post @ (a_n ** 2) - ea ** 2
        vbb = post @ (b_n ** 2) - eb ** 2
        vab = post @ (a_n * b_n) - ea * eb
        return ea, eb, vaa, vbb, vab

    ea, eb, vaa, vbb, vab = moments(post, a_n, b_n)
    sa = float(np.exp(psi[ix["log_sd_a"]]))
    if has_growth:
        sb_ = float(np.exp(psi[ix["log_sd_b"]]))
        r = float(np.tanh(psi[ix["atanh_r"]]))
        sd_c = np.sqrt(max(sa ** 2 + tbar ** 2 * sb_ ** 2 + 2 * tbar * r * sa * sb_, 1e-6))
        d_a = psi[ix["d_a"]] if has_group else 0.0
        d_b = psi[ix["d_b"]] if has_group else 0.0
        mu_c, d_c = psi[ix["mu_a"]] + psi[ix["mu_b"]] * tbar, d_a + d_b * tbar
        post_sd_c = np.sqrt(np.maximum(vaa + tbar ** 2 * vbb + 2 * tbar * vab, 1e-8))
        h_c = min(0.8 * float(np.median(post_sd_c)), sd_c / 2)
        h_b = min(0.8 * float(np.median(np.sqrt(np.maximum(vbb, 1e-8)))), 0.7 * sb_)
        c_lo, c_hi = mu_c + min(0, d_c) - 5 * sd_c, mu_c + max(0, d_c) + 5 * sd_c
        b_lo, b_hi = psi[ix["mu_b"]] + min(0, d_b) - 5 * sb_, psi[ix["mu_b"]] + max(0, d_b) + 5 * sb_
        qc = int(np.clip(np.ceil((c_hi - c_lo) / max(h_c, 1e-3)) + 1, 21, 161))
        qb = int(np.clip(np.ceil((b_hi - b_lo) / max(h_b, 1e-3)) + 1, 15, 101))
    else:
        d_a = psi[ix["d_a"]] if has_group else 0.0
        h_c = min(0.8 * float(np.median(np.sqrt(np.maximum(vaa, 1e-8)))), sa / 2)
        c_lo, c_hi = psi[ix["mu_a"]] + min(0, d_a) - 5 * sa, psi[ix["mu_a"]] + max(0, d_a) + 5 * sa
        qc = int(np.clip(np.ceil((c_hi - c_lo) / max(h_c, 1e-3)) + 1, 41, 401))
        b_lo = b_hi = 0.0
        qb = 1
    a_n, b_n, L, geom = make_nodes(c_lo, c_hi, qc, b_lo, b_hi, qb)
    res = run(psi, a_n, b_n, L, 200, geom)
    psi = res.x
    post, lpn, ll = posterior(psi, a_n, b_n, L)

    # ---- OPG covariance from per-student scores
    eps = 1e-5
    dl = np.empty((P, G, a_n.size))
    for p in range(P):
        e = np.zeros(P)
        e[p] = eps
        dl[p] = (norm(lp_nodes(psi + e, a_n, b_n)) - norm(lp_nodes(psi - e, a_n, b_n))) / (2 * eps)
    scores = np.empty((N, P))
    for g in range(G):
        scores[gidx[g]] = post[gidx[g]] @ dl[:, g, :].T
    cov = np.linalg.pinv(scores.T @ scores)

    ea, eb, vaa, vbb, vab = moments(post, a_n, b_n)
    nat = {"mu_a": float(psi[ix["mu_a"]]), "sd_a": float(np.exp(psi[ix["log_sd_a"]]))}
    se = {"mu_a": float(np.sqrt(cov[ix["mu_a"], ix["mu_a"]]))}
    se["sd_a"] = nat["sd_a"] * float(np.sqrt(cov[ix["log_sd_a"], ix["log_sd_a"]]))
    if has_group:
        nat["d_a"] = float(psi[ix["d_a"]])
        se["d_a"] = float(np.sqrt(cov[ix["d_a"], ix["d_a"]]))
    if has_growth:
        nat["mu_b"] = float(psi[ix["mu_b"]])
        se["mu_b"] = float(np.sqrt(cov[ix["mu_b"], ix["mu_b"]]))
        nat["sd_b"] = float(np.exp(psi[ix["log_sd_b"]]))
        se["sd_b"] = nat["sd_b"] * float(np.sqrt(cov[ix["log_sd_b"], ix["log_sd_b"]]))
        nat["r_ab"] = float(np.tanh(psi[ix["atanh_r"]]))
        se["r_ab"] = (1 - nat["r_ab"] ** 2) * float(np.sqrt(cov[ix["atanh_r"], ix["atanh_r"]]))
        if has_group:
            nat["d_b"] = float(psi[ix["d_b"]])
            se["d_b"] = float(np.sqrt(cov[ix["d_b"], ix["d_b"]]))
        pm = np.stack([ea, eb], axis=1)
        pc = np.empty((N, 2, 2))
        pc[:, 0, 0], pc[:, 1, 1] = np.maximum(vaa, 1e-8), np.maximum(vbb, 1e-8)
        pc[:, 0, 1] = pc[:, 1, 0] = vab
    else:
        pm = ea[:, None]
        pc = np.maximum(vaa, 1e-8)[:, None, None]
    # EAP reliability of theta_t
    rel = np.empty(times.size)
    for t_i, tv in enumerate(times):
        e_th = ea + tv * eb
        v_th = vaa + tv ** 2 * vbb + 2 * tv * vab
        rel[t_i] = np.var(e_th) / max(np.var(e_th) + np.mean(v_th), 1e-12)
    return ComponentFit(name=name, delta=delta, tau=tau, used=used, par_names=names, psi=psi, cov_psi=cov,
                        natural=nat, se=se, post_mean=pm, post_cov=pc, loglik=ll, n_nodes=int(a_n.size),
                        reliability=rel, converged=bool(res.success))


# ===================================================================== stage 2: joint distribution
def _joint_draws(data: EAGData, comps: List[ComponentFit], has_group, has_growth, rng,
                 n_prop=300, n_draws=150, df=6.0, max_iter=15, tol=1e-3, prior_n=2.0, fixed=None):
    """Joint latent distribution and posterior draws for every student.

    The cross-component covariances are estimated by an EM algorithm in which the E-step is an
    importance-sampling average. Proposals are multivariate-t distributions centred on a Gaussian
    approximation of each student's joint posterior. A weak inverse-Wishart prior, worth `prior_n`
    pseudo-students and centred on independence between components, keeps the estimate away from
    singular solutions in small samples. A closed-form Gaussian EM is run to convergence first, and
    the importance-sampling EM then corrects for the non-Gaussian shape of the rubric likelihoods.
    """
    N, T, K, J = data.X.shape
    times = data.times
    dpc = 2 if has_growth else 1
    D = K * dpc
    prior_n = float(prior_n)
    group = data.group if has_group else np.zeros(N, int)
    pos = [[k] + ([K + k] if has_growth else []) for k in range(K)]

    # ---- stage-1 quantities: group means, block-diagonal covariance
    mu = np.zeros((2, D))
    S_bd = np.zeros((D, D))
    for k, cf in enumerate(comps):
        nat = cf.natural
        for g in range(2):
            mu[g, k] = nat["mu_a"] + g * nat.get("d_a", 0.0)
            if has_growth:
                mu[g, K + k] = nat["mu_b"] + g * nat.get("d_b", 0.0)
        S_bd[k, k] = nat["sd_a"] ** 2
        if has_growth:
            S_bd[K + k, K + k] = nat["sd_b"] ** 2
            S_bd[k, K + k] = S_bd[K + k, k] = nat["r_ab"] * nat["sd_a"] * nat["sd_b"]
    mu_i = mu[group]                                                   # (N, D)

    # ---- Gaussian summary of each component's rubric likelihood: precision Lam and natural mean eta
    Lam = np.zeros((N, D, D))
    eta = np.zeros((N, D))
    eap = np.zeros((N, D))
    Wbar = np.zeros((D, D))
    for k, cf in enumerate(comps):
        ix = np.ix_(pos[k], pos[k])
        Sk_inv = np.linalg.inv(S_bd[ix])
        V_inv = np.linalg.inv(cf.post_cov)                             # (N, dpc, dpc)
        L_k = V_inv - Sk_inv[None]
        ev, evec = np.linalg.eigh(L_k)
        L_k = np.einsum("nij,nj,nkj->nik", evec, np.clip(ev, 1e-6, None), evec)
        e_k = np.einsum("nij,nj->ni", V_inv, cf.post_mean) - mu_i[:, pos[k]] @ Sk_inv
        for a_, pa in enumerate(pos[k]):
            eta[:, pa] = e_k[:, a_]
            eap[:, pa] = cf.post_mean[:, a_]
            for b_, pb in enumerate(pos[k]):
                Lam[:, pa, pb] = L_k[:, a_, b_]
        Wbar[ix] = np.eye(dpc) - cf.post_cov.mean(axis=0) @ Sk_inv      # average shrinkage of the EAP

    # ---- Gaussian-approximation EM for the joint covariance (closed form, run to convergence).
    #      Each student's rubric likelihood is summarised by (Lam, eta); given Sigma the posterior is
    #      Gaussian with precision Sigma^-1 + Lam, so both EM steps are exact and cheap.
    c0 = np.cov((eap - mu_i).T)
    Winv = np.linalg.pinv(Wbar)
    Sigma = Winv @ c0 @ Winv.T
    for k in range(K):
        ix = np.ix_(pos[k], pos[k])
        Sigma[ix] = S_bd[ix]
    sd = np.sqrt(np.diag(S_bd))
    Rm = np.clip(Sigma / np.outer(sd, sd), -0.9, 0.9)
    np.fill_diagonal(Rm, 1.0)
    for k in range(K):
        ix = np.ix_(pos[k], pos[k])
        Rm[ix] = S_bd[ix] / np.outer(sd[pos[k]], sd[pos[k]])
    ev, evec = np.linalg.eigh(Rm)
    Rm = (evec * np.clip(ev, 0.05, None)) @ evec.T
    dr = np.sqrt(np.diag(Rm))
    Sigma = (Rm / np.outer(dr, dr)) * np.outer(sd, sd)
    gauss_iter = 0
    if fixed is None:
        for gauss_iter in range(1, 3001):
            S_inv = np.linalg.inv(Sigma)
            Pinv = np.linalg.inv(S_inv[None] + Lam)                     # posterior covariances (N, D, D)
            mpost = np.einsum("nij,nj->ni", Pinv, mu_i @ S_inv + eta)
            dev0 = mpost - mu_i
            new = (dev0.T @ dev0 + Pinv.sum(axis=0) + prior_n * S_bd) / (N + prior_n)
            sd_o, sd_n = np.sqrt(np.diag(Sigma)), np.sqrt(np.diag(new))
            change = max(np.abs(new / np.outer(sd_n, sd_n) - Sigma / np.outer(sd_o, sd_o)).max(),
                         np.abs(sd_n / sd_o - 1).max())
            Sigma = new
            if change < 1e-6:
                break
    else:                                                               # known population values (diagnostics only)
        mu, Sigma = np.asarray(fixed[0], float), np.asarray(fixed[1], float)
        mu_i = mu[group]
        max_iter = 0

    # ---- proposals: a defensive mixture of (1) a multivariate t around the Gaussian product
    #      approximation of each joint posterior and (2) independent t blocks around the stage-1 posteriors
    S_inv = np.linalg.inv(Sigma)
    P = S_inv[None] + Lam
    h = mu_i @ S_inv + eta
    m = np.linalg.solve(P, h[:, :, None])[:, :, 0]
    Lp = np.linalg.cholesky(P)                                          # P = Lp Lp'
    n1 = n_prop // 2
    n2 = n_prop - n1
    eps = rng.standard_normal((N, D, n1))
    gsc = rng.chisquare(df, size=(N, 1, n1)) / df
    Z1 = np.swapaxes(m[:, :, None] + np.linalg.solve(np.swapaxes(Lp, 1, 2), eps) / np.sqrt(gsc), 1, 2)
    Z2 = np.empty((N, n2, D))
    Vinv_blocks, logdetV = [], np.zeros(N)
    for k, cf in enumerate(comps):
        Lv = np.linalg.cholesky(cf.post_cov)                            # (N, dpc, dpc)
        e2 = rng.standard_normal((N, dpc, n2))
        g2 = rng.chisquare(df, size=(N, 1, n2)) / df
        zk = cf.post_mean[:, :, None] + (Lv @ e2) / np.sqrt(g2)
        for a_, pa in enumerate(pos[k]):
            Z2[:, :, pa] = zk[:, a_, :]
        Vinv_blocks.append(np.linalg.inv(cf.post_cov))
        logdetV += 2.0 * np.log(np.einsum("nii->ni", Lv)).sum(axis=1)
    Z = np.concatenate([Z1, Z2], axis=1)                                # (N, n_prop, D)
    dev = np.swapaxes(Z - m[:, None, :], 1, 2)                          # (N, D, n_prop)
    d2 = ((np.swapaxes(Lp, 1, 2) @ dev) ** 2).sum(axis=1)
    logdet_cov = -2.0 * np.log(np.einsum("nii->ni", Lp)).sum(axis=1)    # log |P^-1|
    logq1 = (gammaln((df + D) / 2) - gammaln(df / 2) - 0.5 * D * np.log(df * np.pi)
             - 0.5 * logdet_cov[:, None] - 0.5 * (df + D) * np.log1p(d2 / df))
    logq2 = np.full((N, n_prop), K * (gammaln((df + dpc) / 2) - gammaln(df / 2) - 0.5 * dpc * np.log(df * np.pi)))
    logq2 -= 0.5 * logdetV[:, None]
    for k, cf in enumerate(comps):
        dk = Z[:, :, pos[k]] - cf.post_mean[:, None, :]
        d2k = np.einsum("nmi,nij,nmj->nm", dk, Vinv_blocks[k], dk)
        logq2 -= 0.5 * (df + dpc) * np.log1p(d2k / df)
    logq = np.logaddexp(np.log(n1 / n_prop) + logq1, np.log(n2 / n_prop) + logq2)

    # ---- rubric log-likelihood at the draws
    ll = np.zeros((N, n_prop))
    for k, cf in enumerate(comps):
        a = Z[:, :, k]
        b = Z[:, :, K + k] if has_growth else np.zeros_like(a)
        Xk = data.X[:, :, k, :]
        obs = (Xk >= 0) & cf.used[None, None, :]
        S = np.where(obs, Xk, 0).sum(axis=2).astype(float)
        tab = _ATable(cf.delta, cf.tau, cf.used)
        complete = bool((obs.sum(axis=2) == cf.used.sum()).all())
        for t_i in range(T):
            th = a + b * times[t_i]
            if complete:
                ll += S[:, t_i, None] * th - tab.total(th)
            else:
                ll += S[:, t_i, None] * th
                for j in np.flatnonzero(cf.used):
                    ll -= obs[:, t_i, j, None] * tab.item(j, th)

    # ---- EM by reweighting
    R = (Z - mu_i[:, None, :]).reshape(N * n_prop, D)
    base = ll - logq

    def weights(Sig):
        Lc = np.linalg.cholesky(Sig + 1e-9 * np.eye(D))
        sol = solve_triangular(Lc, R.T, lower=True)
        lw = base + (-0.5 * (sol ** 2).sum(axis=0)).reshape(N, n_prop) - np.log(np.diag(Lc)).sum()
        lw -= logsumexp(lw, axis=1, keepdims=True)
        return np.exp(lw)

    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        W = weights(Sigma)
        new = ((R * W.reshape(-1, 1)).T @ R + prior_n * S_bd) / (N + prior_n)
        sd_o, sd_n = np.sqrt(np.diag(Sigma)), np.sqrt(np.diag(new))
        change = max(np.abs(new / np.outer(sd_n, sd_n) - Sigma / np.outer(sd_o, sd_o)).max(),
                     np.abs(sd_n / sd_o - 1).max())
        Sigma = new
        if change < tol:
            break
    W = weights(Sigma)
    ess = 1.0 / (W ** 2).sum(axis=1)
    # ---- systematic resampling to equally weighted draws
    cum = np.cumsum(W, axis=1)
    cum[:, -1] = 1.0
    u = (rng.random((N, 1)) + np.arange(n_draws)[None, :]) / n_draws
    idx = np.empty((N, n_draws), dtype=int)
    for i in range(N):
        idx[i] = np.searchsorted(cum[i], u[i])
    idx = np.minimum(idx, n_prop - 1)
    draws = np.take_along_axis(Z, idx[:, :, None], axis=1)
    return draws, mu, Sigma, ess, n_iter


# ===================================================================== stage 3: aggregation
class AggregationProblem:
    """Simulated likelihood of holistic ratings given equally weighted latent draws."""

    def __init__(self, draws, times, Y, K, has_growth):
        N, M, D = draws.shape
        self.N, self.M, self.K = N, M, K
        a = draws[:, :, :K]
        b = draws[:, :, K:] if has_growth else np.zeros_like(a)
        self.times = np.asarray(times, float)
        theta = a[:, :, None, :] + b[:, :, None, :] * self.times[None, None, :, None]
        self.LP = log_mastery(np.clip(theta, -14.0, None))        # (N, M, T, K)
        self.PI = np.exp(self.LP)
        self.a, self.b = a, b
        self.O = np.isfinite(Y)
        self.R = Y.shape[2]
        self.y_mean = float(np.nanmean(Y))
        self.y_sd = float(np.nanstd(Y))
        self.Ys = np.where(self.O, (Y - self.y_mean) / self.y_sd, 0.0)
        self.Of = self.O.astype(float)
        self.n_obs = self.Of.sum(axis=2)                          # (N, T)
        self.n_tot = self.n_obs.sum(axis=1)                       # (N,)
        self.B = np.vstack([np.eye(self.R - 1), -np.ones((1, self.R - 1))]) if self.R > 1 else np.zeros((1, 0))

    def subset(self, idx):
        """A copy restricted to the students in idx (ratings stay on the full-sample standardised scale)."""
        import copy

        sub = copy.copy(self)
        idx = np.asarray(idx)
        for name in ("LP", "PI", "a", "b", "O", "Ys", "Of", "n_obs", "n_tot"):
            setattr(sub, name, getattr(self, name)[idx])
        sub.N = idx.size
        return sub

    # ---- parameter layout: u (K-1), v = log kappa, a_h, b_h, phi (R-1), log sigma
    def layout(self, fix_kappa=None, equal_weights=False):
        names = [] if equal_weights else [f"u{k}" for k in range(self.K - 1)]
        if fix_kappa is None:
            names.append("v")
        names += ["a_h", "b_h"] + [f"phi{r}" for r in range(self.R - 1)] + ["log_sigma"]
        return names

    def bounds(self, names):
        lo, hi = np.full(len(names), -np.inf), np.full(len(names), np.inf)
        for i, n_ in enumerate(names):
            if n_.startswith("u"):
                lo[i], hi[i] = -6.0, 6.0
            elif n_ == "v":
                lo[i], hi[i] = np.log(0.02), np.log(20.0)
            elif n_ == "log_sigma":
                lo[i], hi[i] = np.log(1e-3), np.log(10.0)
        return lo, hi

    def unpack(self, par, fix_kappa=None, equal_weights=False):
        i = 0
        if equal_weights:
            u = np.zeros(self.K)
        else:
            u = np.append(par[:self.K - 1], 0.0)
            i = self.K - 1
        w = np.exp(u - logsumexp(u))
        if fix_kappa is None:
            kappa = float(np.exp(par[i]))
            i += 1
        else:
            kappa = float(fix_kappa)
        a_h, b_h = par[i], par[i + 1]
        i += 2
        phi = par[i:i + self.R - 1]
        i += self.R - 1
        sigma = float(np.exp(par[i]))
        return w, kappa, a_h, b_h, phi, sigma

    def evaluate(self, par, fix_kappa=None, equal_weights=False, want="grad"):
        """Return (negative log-likelihood, negative gradient) or the pieces named by ``want``."""
        w, kappa, a_h, b_h, phi, sigma = self.unpack(par, fix_kappa, equal_weights)
        rho = float(rho_from_kappa(kappa))
        drho_dv = 1.0 / kappa if kappa <= 1.0 else kappa            # v = log kappa
        if abs(rho) < RHO_EPS:
            rho = RHO_EPS if rho >= 0 else -RHO_EPS
        if rho == 1.0:
            e = self.PI * w
            se = e.sum(axis=-1)
            lse = np.log(se)
            C = se
        else:
            e = np.exp(rho * self.LP)
            e *= w
            se = e.sum(axis=-1)                                     # (N, M, T)
            lse = np.log(se)
            C = np.exp(lse / rho)
        eta = a_h + b_h * C
        lam = self.B @ phi if self.R > 1 else np.zeros(1)
        Yl = (self.Ys + lam[None, None, :]) * self.Of               # (N, T, R)
        ysum = Yl.sum(axis=2)[:, None, :]
        ysq = (Yl ** 2).sum(axis=2)[:, None, :]
        nob = self.n_obs[:, None, :]
        sse_tot = (ysq - 2 * eta * ysum + nob * eta * eta).sum(axis=-1)     # (N, M)
        h = -self.n_tot[:, None] * (0.5 * LOG2PI + np.log(sigma)) - sse_tot / (2 * sigma ** 2)
        mxh = h.max(axis=1, keepdims=True)
        ph = np.exp(h - mxh)
        sh = ph.sum(axis=1, keepdims=True)
        ll_i = (np.log(sh / self.M) + mxh)[:, 0]
        post = ph / sh
        if want == "ll":
            return ll_i
        if want == "post":
            return post, C, e / se[..., None], ll_i
        pe = post[:, :, None] * (ysum - nob * eta) / sigma ** 2      # post * dh / d eta
        bw = pe * C * (b_h / rho)
        cols = []
        if not equal_weights:
            G = (e * (bw / se)[..., None]).sum(axis=(1, 2))          # sum bw * share_k
            bsum = bw.sum(axis=(1, 2))
            for j in range(self.K - 1):
                cols.append(G[:, j] - w[j] * bsum)
        if fix_kappa is None:
            sLP = (e * self.LP).sum(axis=-1) / se
            cols.append((bw * (sLP - lse / rho)).sum(axis=(1, 2)) * drho_dv)
        cols.append(pe.sum(axis=(1, 2)))
        cols.append((pe * C).sum(axis=(1, 2)))
        if self.R > 1:
            eta_post = np.einsum("nm,nmt->nt", post, eta)
            dlam = -(Yl.sum(axis=1) - np.einsum("nt,ntr->nr", eta_post, self.Of)) / sigma ** 2
            dphi = dlam @ self.B
            for r in range(self.R - 1):
                cols.append(dphi[:, r])
        cols.append((post * (-self.n_tot[:, None] + sse_tot / sigma ** 2)).sum(axis=1))
        scores = np.stack(cols, axis=1)                             # (N, P)
        if want == "scores":
            return ll_i, scores
        return -float(ll_i.sum()), -scores.sum(axis=0)

    def start(self, fix_kappa=None, equal_weights=False, kappa0=0.5):
        names = self.layout(fix_kappa, equal_weights)
        par = np.zeros(len(names))
        if fix_kappa is None:
            par[names.index("v")] = np.log(kappa0)
        w = np.full(self.K, 1.0 / self.K)
        kap = kappa0 if fix_kappa is None else fix_kappa
        rho = float(rho_from_kappa(kap))
        if abs(rho) < RHO_EPS:
            rho = RHO_EPS
        C = np.exp(logsumexp(np.log(w) + rho * self.LP, axis=-1) / rho).mean(axis=1)   # (N, T)
        ybar = np.where(self.n_obs > 0, (self.Ys * self.Of).sum(axis=2) / np.maximum(self.n_obs, 1), np.nan)
        ok = np.isfinite(ybar)
        slope, intercept = np.polyfit(C[ok], ybar[ok], 1)
        resid = ybar[ok] - (intercept + slope * C[ok])
        par[names.index("a_h")], par[names.index("b_h")] = intercept, slope
        par[names.index("log_sigma")] = np.log(max(resid.std() * 1.2, 0.05))
        return par

    def _bhhh(self, x0, fix_kappa, equal_weights, lo, hi, maxiter=80, tol=1e-7):
        """Berndt-Hall-Hall-Hausman ascent: Newton steps with the outer product of scores as curvature."""
        x = np.clip(x0, lo, hi)
        ll_i, sc = self.evaluate(x, fix_kappa, equal_weights, want="scores")
        f = float(ll_i.sum())
        damp = 1e-2
        n_eval = 1
        for it in range(maxiter):
            g = sc.sum(axis=0)
            free = ~(((x <= lo + 1e-12) & (g < 0)) | ((x >= hi - 1e-12) & (g > 0)))
            H = sc.T @ sc
            Hd = H + damp * np.diag(np.diag(H)) + 1e-10 * np.eye(H.shape[0])
            step = np.zeros_like(x)
            try:
                step[free] = np.linalg.solve(Hd[np.ix_(free, free)], g[free])
            except np.linalg.LinAlgError:
                step[free] = g[free] / np.maximum(np.diag(H)[free], 1e-8)
            decrement = float(g[free] @ step[free])
            if decrement < tol:
                return x, f, True, n_eval
            t, accepted = 1.0, False
            while t > 1e-4:
                xn = np.clip(x + t * step, lo, hi)
                ll_n, sc_n = self.evaluate(xn, fix_kappa, equal_weights, want="scores")
                n_eval += 1
                fn = float(ll_n.sum())
                if np.isfinite(fn) and fn >= f + 1e-4 * t * decrement:
                    accepted = True
                    break
                t *= 0.4
            if accepted:
                x, f, sc = xn, fn, sc_n
                damp = max(damp / 4, 1e-8) if t == 1.0 else damp
            else:
                damp *= 20
                if damp > 1e6:
                    return x, f, False, n_eval
        return x, f, False, n_eval

    def fit(self, fix_kappa=None, equal_weights=False, start=None, kappa0=0.5):
        names = self.layout(fix_kappa, equal_weights)
        x0 = self.start(fix_kappa, equal_weights, kappa0) if start is None else np.asarray(start, float)
        lo, hi = self.bounds(names)
        x, f, ok, n_eval = self._bhhh(x0, fix_kappa, equal_weights, lo, hi)
        if not ok:   # fall back on a quasi-Newton polish
            res = minimize(lambda p: self.evaluate(p, fix_kappa, equal_weights), x, jac=True, method="L-BFGS-B",
                           bounds=list(zip(np.where(np.isfinite(lo), lo, None), np.where(np.isfinite(hi), hi, None))),
                           options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-6})
            if -res.fun >= f:
                x, f, ok = res.x, -float(res.fun), bool(res.success)
            n_eval += res.nfev
        return names, x, f, ok, n_eval


@dataclass
class AggregationFit:
    label: str
    par_names: List[str]
    par: np.ndarray
    cov: np.ndarray
    loglik: float
    n_par: int
    weights: np.ndarray
    weights_se: np.ndarray
    kappa: float
    kappa_ci: tuple
    rho: float
    rho_ci: tuple
    intercept: float
    slope: float
    sigma: float
    severity: np.ndarray
    severity_se: np.ndarray
    converged: bool
    n_eval: int = 0
    fix_kappa: Optional[float] = None
    equal_weights: bool = False

    @property
    def aic(self):
        return -2 * self.loglik + 2 * self.n_par

    def bic(self, n):
        return -2 * self.loglik + np.log(n) * self.n_par


def _summarise_aggregation(prob: AggregationProblem, label, fix_kappa=None, equal_weights=False,
                           start=None, kappa0=0.5):
    names, par, _, ok, n_eval = prob.fit(fix_kappa, equal_weights, start, kappa0)
    ll_i, scores = prob.evaluate(par, fix_kappa, equal_weights, want="scores")
    cov = np.linalg.pinv(scores.T @ scores)
    w, kappa, a_h, b_h, phi, sigma = prob.unpack(par, fix_kappa, equal_weights)
    K, R = prob.K, prob.R
    if equal_weights:
        w_se = np.zeros(K)
    else:
        Jw = np.array([[w[k] * ((k == j) - w[j]) for j in range(K - 1)] for k in range(K)])
        w_se = np.sqrt(np.maximum(np.einsum("kj,jl,kl->k", Jw, cov[:K - 1, :K - 1], Jw), 0))
    if fix_kappa is None:
        iv = names.index("v")
        se_v = float(np.sqrt(max(cov[iv, iv], 0)))
        k_ci = (float(np.exp(par[iv] - 1.96 * se_v)), float(np.exp(par[iv] + 1.96 * se_v)))
    else:
        k_ci = (kappa, kappa)
    rho = float(rho_from_kappa(kappa))
    rho_ci = (float(rho_from_kappa(k_ci[0])), float(rho_from_kappa(k_ci[1])))
    if R > 1:
        ip = [names.index(f"phi{r}") for r in range(R - 1)]
        lam = prob.B @ phi
        lam_cov = prob.B @ cov[np.ix_(ip, ip)] @ prob.B.T
        lam_se = np.sqrt(np.maximum(np.diag(lam_cov), 0))
    else:
        lam, lam_se = np.zeros(1), np.zeros(1)
    return AggregationFit(
        label=label, par_names=names, par=par, cov=cov, loglik=float(ll_i.sum()), n_par=len(names),
        weights=w, weights_se=w_se, kappa=kappa, kappa_ci=k_ci, rho=rho, rho_ci=rho_ci,
        intercept=prob.y_mean + prob.y_sd * a_h, slope=prob.y_sd * b_h, sigma=prob.y_sd * sigma,
        severity=prob.y_sd * lam, severity_se=prob.y_sd * lam_se, converged=bool(ok), n_eval=int(n_eval),
        fix_kappa=fix_kappa, equal_weights=equal_weights)


# ===================================================================== public interface
@dataclass
class EAGFit:
    data: EAGData
    components: List[ComponentFit]
    has_group: bool
    has_growth: bool
    mu: np.ndarray
    Sigma: np.ndarray
    ess: np.ndarray
    em_iterations: int
    problem: AggregationProblem
    models: Dict[str, AggregationFit]
    lr_linear: Dict[str, float]
    student: Dict[str, np.ndarray]
    seconds: float
    settings: Dict[str, float] = field(default_factory=dict)

    @property
    def free(self) -> AggregationFit:
        return self.models["free"]

    def latent_correlations(self):
        sd = np.sqrt(np.diag(self.Sigma))
        return self.Sigma / np.outer(sd, sd)


def student_summaries(prob: AggregationProblem, agg: AggregationFit, threshold=None, forecast=True):
    """Posterior means and standard deviations of competence, mastery and leverage per student."""
    post, C, s, _ = prob.evaluate(agg.par, agg.fix_kappa, agg.equal_weights, want="post")
    pi = np.exp(prob.LP)
    c_mean = np.einsum("nm,nmt->nt", post, C)
    c_sd = np.sqrt(np.maximum(np.einsum("nm,nmt->nt", post, C ** 2) - c_mean ** 2, 0))
    pi_mean = np.einsum("nm,nmtk->ntk", post, pi)
    lev = C[..., None] * s * (1 - pi)                                 # dC / d theta_k
    lev_mean = np.einsum("nm,nmk->nk", post, lev[:, :, -1, :])
    out = dict(C_mean=c_mean, C_sd=c_sd, pi_mean=pi_mean, leverage=lev_mean, target=lev_mean.argmax(axis=1),
               a_mean=np.einsum("nm,nmk->nk", post, prob.a), b_mean=np.einsum("nm,nmk->nk", post, prob.b))
    if threshold is not None:
        out["p_ready"] = np.einsum("nm,nm->n", post, (C[:, :, -1] >= threshold).astype(float))
        if forecast and prob.times.size > 1:
            t_last = prob.times[-1]
            out["cycles_to_ready"] = np.array([
                max(time_to_readiness(out["a_mean"][i], out["b_mean"][i], agg.weights, agg.rho, threshold) - t_last, 0.0)
                for i in range(prob.N)])
    return out


def fit(data: EAGData, n_draws: int = 150, n_prop: int = 300, seed: int = 0, threshold: Optional[float] = None,
        extra_models: bool = False, verbose: bool = False) -> EAGFit:
    """Fit the EAG model.

    extra_models=True also fits the geometric-mean and equal-weight variants used in model comparison.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    N, T, K, J = data.X.shape
    has_group = data.group is not None and len(np.unique(data.group)) == 2
    has_growth = T > 1
    group = data.group if has_group else np.zeros(N, int)
    comps = []
    for k in range(K):
        comps.append(_fit_component(data.X[:, :, k, :], data.times, group, data.max_score,
                                    data.components[k], has_group, has_growth))
        if verbose:
            print(f"  stage 1  {data.components[k]:<24s} nodes={comps[-1].n_nodes:5d}  {time.time() - t0:5.1f}s")
    draws, mu, Sigma, ess, n_iter = _joint_draws(data, comps, has_group, has_growth, rng, n_prop, n_draws)
    if verbose:
        print(f"  stage 2  EM iterations={n_iter}  median ESS={np.median(ess):.0f}/{n_prop}  {time.time() - t0:5.1f}s")
    prob = AggregationProblem(draws, data.times, data.Y, K, has_growth)
    models = {}
    lin = _summarise_aggregation(prob, "linear (kappa = 1)", fix_kappa=1.0)
    models["linear"] = lin
    best = None
    for k0 in (0.5, 0.15):
        start = prob.start(None, False, k0)
        start[:K - 1] = lin.par[:K - 1]
        cand = _summarise_aggregation(prob, "free", start=start, kappa0=k0)
        if best is None or cand.loglik > best.loglik + 1e-6:
            best = cand
    models["free"] = best
    lr = max(2 * (best.loglik - lin.loglik), 0.0)
    lr_linear = dict(statistic=lr, df=1, p=float(chi2.sf(lr, 1)))
    if extra_models:
        models["geometric"] = _summarise_aggregation(prob, "geometric (kappa = 0.5)", fix_kappa=0.5)
        models["equal"] = _summarise_aggregation(prob, "free kappa, equal weights", equal_weights=True,
                                                 kappa0=float(np.clip(best.kappa, 0.03, 4)))
    student = student_summaries(prob, best, threshold)
    if verbose:
        print(f"  stage 3  kappa={best.kappa:.3f}  LR(linear)={lr:.2f}  {time.time() - t0:5.1f}s")
    return EAGFit(data=data, components=comps, has_group=has_group, has_growth=has_growth, mu=mu, Sigma=Sigma,
                  ess=ess, em_iterations=n_iter, problem=prob, models=models, lr_linear=lr_linear,
                  student=student, seconds=time.time() - t0,
                  settings=dict(n_draws=n_draws, n_prop=n_prop, seed=seed, threshold=threshold))
