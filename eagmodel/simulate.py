"""Data-generating process of the EAG model."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np

from .data import EAGData
from .functions import log_aggregate, log_mastery, rsm_probs

DEFAULT_COMPONENTS = ("Problem framing", "System modelling", "Measurement design",
                      "Inference design", "Iterative refinement")


@dataclass
class SimConfig:
    """Population values used to generate data. Defaults follow the paper's reference scenario."""

    n_students: int = 200
    components: Sequence[str] = DEFAULT_COMPONENTS
    n_indicators: int = 4
    max_score: int = 3
    times: Sequence[float] = (0, 1, 2, 3)
    n_raters: int = 2
    weights: Sequence[float] = (0.15, 0.20, 0.20, 0.25, 0.20)
    rho: float = -2.0
    alpha: Sequence[float] = (-0.3, -0.6, -0.8, -1.0, -0.7)   # initial proficiency means
    sd_a: float = 1.0
    gamma: Sequence[float] = (0.35, 0.30, 0.30, 0.25, 0.30)   # growth per design cycle
    beta: Sequence[float] = (0.10, 0.15, 0.20, 0.15, 0.20)    # added growth in group 1
    sd_b: float = 0.25
    corr_a: float = 0.5      # between-component correlation of initial proficiencies
    corr_b: float = 0.3      # between-component correlation of growth rates
    corr_ab: float = -0.2    # within-component correlation of initial proficiency and growth
    delta: Sequence[float] = (-0.6, -0.2, 0.2, 0.6)
    tau: Sequence[float] = (-1.2, 0.0, 1.2)
    h_intercept: float = 20.0
    h_slope: float = 75.0
    h_sd: float = 8.0
    rater_severity: Sequence[float] = (-3.0, 3.0)
    group_fraction: float = 0.5
    holistic_occasions: Optional[Sequence[int]] = None   # indices of occasions with holistic ratings
    link: str = "identity"   # "identity" (model) or "logit" (misspecified link for robustness checks)

    def latent_covariance(self):
        k = len(self.components)
        eye, ones = np.eye(k), np.ones((k, k))
        saa = self.sd_a ** 2 * ((1 - self.corr_a) * eye + self.corr_a * ones)
        sbb = self.sd_b ** 2 * ((1 - self.corr_b) * eye + self.corr_b * ones)
        sab = self.corr_ab * self.sd_a * self.sd_b * eye
        return np.block([[saa, sab], [sab.T, sbb]])

    def to_dict(self):
        return {k: (list(v) if isinstance(v, (tuple, list, np.ndarray)) else v) for k, v in asdict(self).items()}


def simulate(config: SimConfig = None, seed: Optional[int] = None, return_truth: bool = True):
    """Generate one data set. Returns (EAGData, truth) where truth holds the latent values."""
    cfg = config or SimConfig()
    rng = np.random.default_rng(seed)
    n, k, j, m = cfg.n_students, len(cfg.components), cfg.n_indicators, cfg.max_score
    times = np.asarray(cfg.times, dtype=float)
    t = times.size
    w = np.asarray(cfg.weights, dtype=float)
    if w.size != k or abs(w.sum() - 1) > 1e-8:
        raise ValueError("weights must have one entry per component and sum to one")
    group = (rng.permutation(n) < round(cfg.group_fraction * n)).astype(int)
    mu = np.concatenate([np.asarray(cfg.alpha, float), np.asarray(cfg.gamma, float)])
    sigma = cfg.latent_covariance()
    z = rng.multivariate_normal(mu, sigma, size=n)
    z[:, k:] += group[:, None] * np.asarray(cfg.beta, float)
    a, b = z[:, :k], z[:, k:]
    theta = a[:, None, :] + b[:, None, :] * times[None, :, None]            # (N, T, K)
    delta = np.asarray(cfg.delta, float)
    if delta.size != j:
        delta = np.linspace(-0.6, 0.6, j)
    probs = rsm_probs(theta, delta - delta.mean(), np.asarray(cfg.tau, float)[:m])   # (N, T, K, J, m+1)
    u = rng.random(probs.shape[:-1] + (1,))
    X = (u > np.cumsum(probs, axis=-1)).sum(axis=-1).clip(0, m)
    log_pi = log_mastery(theta)
    C = np.exp(log_aggregate(log_pi, w, cfg.rho))                             # (N, T)
    sev = np.asarray(cfg.rater_severity, float)
    if sev.size != cfg.n_raters:
        sev = np.linspace(-3, 3, cfg.n_raters) if cfg.n_raters > 1 else np.zeros(1)
    sev = sev - sev.mean()
    if cfg.link == "logit":   # same mean and spread at C = 0.5, but curved: used for robustness only
        signal = cfg.h_intercept + cfg.h_slope * (0.5 + (np.log(C / (1 - C))) / 5.0)
    else:
        signal = cfg.h_intercept + cfg.h_slope * C
    Y = signal[:, :, None] - sev[None, None, :] + cfg.h_sd * rng.standard_normal((n, t, cfg.n_raters))
    if cfg.holistic_occasions is not None:
        keep = np.zeros(t, dtype=bool)
        keep[list(cfg.holistic_occasions)] = True
        Y[:, ~keep, :] = np.nan
    data = EAGData(X=X, Y=Y, times=times, max_score=m, group=group,
                   components=list(cfg.components), group_labels=["comparison", "intervention"])
    truth = dict(a=a, b=b, theta=theta, pi=np.exp(log_pi), C=C, group=group, config=cfg.to_dict())
    return (data, truth) if return_truth else data
