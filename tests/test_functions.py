import numpy as np

from eagmodel.functions import (aggregate, kappa_from_rho, leverage, log_mastery, mastery, rho_from_kappa,
                                rsm_probs, time_to_readiness)

W = np.array([0.15, 0.20, 0.20, 0.25, 0.20])
PI = np.array([0.8, 0.6, 0.3, 0.7, 0.5])


def test_special_cases_of_the_aggregation_function():
    assert np.isclose(aggregate(PI, W, 1.0), np.sum(W * PI))                        # sum score
    assert np.isclose(aggregate(PI, W, 0.0), np.exp(np.sum(W * np.log(PI))), atol=1e-6)   # geometric mean
    assert abs(aggregate(PI, W, -400.0) - PI.min()) < 5e-3                          # weakest component
    assert abs(aggregate(PI, W, 400.0) - PI.max()) < 5e-3                           # strongest component


def test_aggregation_is_continuous_at_rho_zero():
    from eagmodel.functions import RHO_EPS

    # the series used near zero and the exact formula agree where they meet
    for sign in (1, -1):
        assert abs(aggregate(PI, W, sign * RHO_EPS * 1.001) - aggregate(PI, W, sign * RHO_EPS * 0.999)) < 1e-6
    assert abs(aggregate(PI, W, 2e-3) - aggregate(PI, W, 0.0)) < 1e-3


def test_flat_profiles_do_not_depend_on_rho():
    flat = np.full(5, 0.6)
    for rho in (1.0, 0.0, -3.0, -20.0):
        assert np.isclose(aggregate(flat, W, rho), 0.6)


def test_competence_increases_with_rho_for_uneven_profiles():
    values = [aggregate(PI, W, r) for r in (-20, -5, -1, 0, 0.5, 1)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_compensation_index_round_trip():
    rho = np.array([-50.0, -8.0, -2.0, 0.0, 0.5, 1.0, 1.4, 3.0])
    assert np.allclose(rho_from_kappa(kappa_from_rho(rho)), rho)
    assert np.allclose(kappa_from_rho([1.0, 0.0, -2.0, -8.0]), [1.0, 0.5, 0.25, 0.1])


def test_leverage_is_the_derivative_of_competence():
    theta = np.log(PI / (1 - PI))
    for rho in (1.0, 0.0, -2.0):
        num = np.empty(5)
        for k in range(5):
            e = np.zeros(5)
            e[k] = 1e-6
            num[k] = (aggregate(mastery(theta + e), W, rho) - aggregate(mastery(theta - e), W, rho)) / 2e-6
        assert np.allclose(leverage(PI, W, rho), num, atol=1e-6)


def test_time_to_readiness():
    a, b = np.array([-1.0, -0.5, -1.5, -0.8, -1.2]), np.full(5, 0.4)
    t = time_to_readiness(a, b, W, -2.0, 0.6)
    assert 0 < t < 50
    assert abs(aggregate(mastery(a + b * t), W, -2.0) - 0.6) < 1e-3
    assert time_to_readiness(a, np.zeros(5), W, -2.0, 0.6) == float("inf")
    assert np.isclose(np.exp(log_mastery(0.3)), mastery(0.3))


def test_rating_scale_probabilities_sum_to_one():
    p = rsm_probs(np.linspace(-3, 3, 7), np.array([-0.5, 0.0, 0.5]), np.array([-1.0, 0.0, 1.0]))
    assert p.shape == (7, 3, 4)
    assert np.allclose(p.sum(axis=-1), 1.0)
