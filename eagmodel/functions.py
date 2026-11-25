"""Core mathematical functions of the Evidence-Aggregation-Growth (EAG) model.

The model writes a competence as a composition of three functions:

    evidence     rubric scores  <-  component proficiency theta   (rating scale model)
    growth       theta_k(t) = a_k + b_k * t                       (logistic growth of mastery)
    aggregation  C = (sum_k w_k * pi_k**rho) ** (1 / rho)          (CES / generalised mean)

with mastery pi_k = logistic(theta_k). The substitution parameter rho is reported
through the compensation index kappa:

    kappa = 1 / (2 - rho)   for rho <= 1   (equal to s / (1 + s), s = elasticity of substitution)
    kappa = rho             for rho  > 1   (smooth continuation; strengths count more than proportionally)

    kappa = 1     rho = 1      fully compensatory (weighted arithmetic mean, the sum-score case)
    kappa = 0.5   rho = 0      weighted geometric mean
    kappa -> 0    rho -> -inf  conjunctive (competence equals the weakest component)
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp

RHO_EPS = 1e-3


def kappa_from_rho(rho):
    """Compensation index: 1 / (2 - rho) for rho <= 1 and rho itself above 1 (continuously differentiable)."""
    rho = np.asarray(rho, dtype=float)
    return np.where(rho <= 1.0, 1.0 / (2.0 - np.minimum(rho, 1.0)), rho)


def rho_from_kappa(kappa):
    """Inverse of :func:`kappa_from_rho`."""
    kappa = np.asarray(kappa, dtype=float)
    return np.where(kappa <= 1.0, 2.0 - 1.0 / np.minimum(kappa, 1.0), kappa)


def mastery(theta):
    """Mastery pi = logistic(theta), anchored at the rubric midpoint (theta = 0)."""
    return expit(theta)


def log_mastery(theta):
    """log(logistic(theta)), computed stably."""
    return -np.logaddexp(0.0, -np.asarray(theta, dtype=float))


def growth(a, b, t):
    """Growth function on the proficiency scale: theta(t) = a + b * t."""
    return np.asarray(a) + np.asarray(b) * t


def log_aggregate(log_pi, w, rho):
    """log C for an array of log-mastery values whose last axis indexes components."""
    log_pi = np.asarray(log_pi, dtype=float)
    w = np.asarray(w, dtype=float)
    if abs(rho) < RHO_EPS:  # second-order expansion around the geometric mean
        m = np.sum(w * log_pi, axis=-1)
        v = np.sum(w * log_pi ** 2, axis=-1) - m ** 2
        return m + 0.5 * rho * v
    return logsumexp(np.log(w) + rho * log_pi, axis=-1) / rho


def aggregate(pi, w, rho):
    """Aggregation function C = (sum_k w_k pi_k**rho)**(1/rho) on the mastery scale."""
    return np.exp(log_aggregate(np.log(np.asarray(pi, dtype=float)), w, rho))


def shares(pi, w, rho):
    """Elasticities d log C / d log pi_k (they sum to one across components)."""
    log_pi = np.log(np.asarray(pi, dtype=float))
    q = np.log(np.asarray(w, dtype=float)) + rho * log_pi
    return np.exp(q - logsumexp(q, axis=-1, keepdims=True))


def leverage(pi, w, rho):
    """Instructional leverage dC / d theta_k = C * share_k * (1 - pi_k).

    It is the gain in competence per unit of growth in the proficiency of component k,
    so the component with the largest leverage is the most productive target for teaching.
    """
    pi = np.asarray(pi, dtype=float)
    c = aggregate(pi, w, rho)
    return c[..., None] * shares(pi, w, rho) * (1.0 - pi)


def competence_trajectory(a, b, times, w, rho):
    """C(t) for initial proficiencies a (..., K), growth rates b (..., K) and times (T,)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    times = np.asarray(times, dtype=float)
    theta = a[..., None, :] + b[..., None, :] * times[:, None]
    return np.exp(log_aggregate(log_mastery(theta), w, rho))


def time_to_readiness(a, b, w, rho, threshold, t_max=50.0, tol=1e-4):
    """Smallest t >= 0 with C(t) >= threshold for one student (bisection); inf if not reached."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    def c_at(t):
        return float(np.exp(log_aggregate(log_mastery(a + b * t), w, rho)))

    if c_at(0.0) >= threshold:
        return 0.0
    if c_at(t_max) < threshold:
        return float("inf")
    lo, hi = 0.0, t_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if c_at(mid) >= threshold:
            hi = mid
        else:
            lo = mid
    return hi


def rsm_cumtau(tau):
    """Cumulative thresholds (0, tau_1, tau_1 + tau_2, ...)."""
    return np.concatenate([[0.0], np.cumsum(np.asarray(tau, dtype=float))])


def rsm_log_normaliser(theta, delta, tau):
    """A_j(theta) = log sum_h exp(h (theta - delta_j) - cumtau_h); output shape theta.shape + (J,)."""
    theta = np.asarray(theta, dtype=float)
    delta = np.asarray(delta, dtype=float)
    cum = rsm_cumtau(tau)
    h = np.arange(cum.size)
    eta = h * (theta[..., None, None] - delta[:, None]) - cum
    return logsumexp(eta, axis=-1)


def rsm_probs(theta, delta, tau):
    """Category probabilities of the rating scale model; output shape theta.shape + (J, m + 1)."""
    theta = np.asarray(theta, dtype=float)
    delta = np.asarray(delta, dtype=float)
    cum = rsm_cumtau(tau)
    h = np.arange(cum.size)
    eta = h * (theta[..., None, None] - delta[:, None]) - cum
    eta -= logsumexp(eta, axis=-1, keepdims=True)
    return np.exp(eta)
