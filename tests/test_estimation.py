import numpy as np
import pytest

from eagmodel import EAGData, SimConfig, fit, simulate
from eagmodel.estimate import cml_rsm
from eagmodel.validate import validate


@pytest.fixture(scope="module")
def fitted():
    data, truth = simulate(SimConfig(n_students=160, rho=-2.0), seed=42)
    return data, truth, fit(data, seed=1, extra_models=True, threshold=0.6)


def test_conditional_ml_recovers_rubric_parameters():
    data, _ = simulate(SimConfig(n_students=600), seed=7)
    delta, tau, used = cml_rsm(data.X[:, :, 0, :].reshape(-1, 4), 3)
    assert used.all()
    assert np.allclose(delta, [-0.6, -0.2, 0.2, 0.6], atol=0.12)
    assert np.allclose(tau, [-1.2, 0.0, 1.2], atol=0.12)


def test_gradient_of_the_simulated_likelihood(fitted):
    _, _, f = fitted
    prob = f.problem
    for kw in (dict(), dict(fix_kappa=1.0), dict(equal_weights=True)):
        par = prob.start(**kw) + 0.03
        _, grad = prob.evaluate(par, **kw)
        num = np.array([(prob.evaluate(par + e, **kw)[0] - prob.evaluate(par - e, **kw)[0]) / 2e-6
                        for e in 1e-6 * np.eye(par.size)])
        assert np.allclose(grad, num, rtol=1e-4, atol=1e-4)


def test_non_compensatory_structure_is_detected(fitted):
    data, truth, f = fitted
    assert 0.12 < f.free.kappa < 0.45                      # true value 0.25
    assert f.lr_linear["p"] < 0.001                        # sum-score aggregation rejected
    assert f.free.loglik >= f.models["linear"].loglik
    assert np.isclose(f.free.weights.sum(), 1.0)
    assert np.corrcoef(f.student["C_mean"][:, -1], truth["C"][:, -1])[0, 1] > 0.93
    assert abs(f.free.slope - 75) < 12 and abs(f.free.sigma - 8) < 1.5


def test_validation_report(fitted):
    _, _, f = fitted
    rep = validate(f, threshold=0.6, cv_folds=3, ppc_reps=100)
    counts = rep.verdict_counts()
    assert counts["fail"] == 0
    assert {"items", "components", "models", "weights", "students", "raters"} <= set(rep.tables)
    assert "probability_ready" in rep.tables["students"].columns
    assert rep.profile["loglik"].max() <= rep.profile["max"] + 1e-3


def test_compensatory_data_do_not_reject_the_sum_score_model():
    rejections = 0
    for seed in range(4):
        data, _ = simulate(SimConfig(n_students=150, rho=1.0), seed=100 + seed)
        rejections += fit(data, seed=seed).lr_linear["p"] < 0.01
    assert rejections <= 1


def test_long_format_round_trip():
    data, _ = simulate(SimConfig(n_students=30), seed=3)
    a, h, s = data.to_frames()
    back = EAGData.from_frames(a, h, s)
    assert (back.X == data.X).all() and np.allclose(back.Y, data.Y) and (back.group == data.group).all()


def test_missing_scores_and_unbalanced_raters():
    data, _ = simulate(SimConfig(n_students=150, rho=-2.0), seed=11)
    rng = np.random.default_rng(0)
    data.X[rng.random(data.X.shape) < 0.05] = -1            # scattered missing rubric scores
    data.X[:10, 2] = -1                                      # ten students miss one whole occasion
    data.Y[rng.random(data.Y.shape) < 0.15] = np.nan         # some ratings missing
    f = fit(data, seed=0)
    assert np.isfinite(f.free.kappa) and f.free.kappa < 0.8
    assert np.isfinite(f.student["C_mean"]).all()


def test_single_occasion_data():
    data, _ = simulate(SimConfig(n_students=200, rho=-2.0, times=(0,), n_raters=3,
                                 rater_severity=(-3.0, 0.0, 3.0)), seed=5)
    f = fit(data, seed=0)
    assert not f.has_growth
    assert np.isfinite(f.free.kappa)
    rep = validate(f, cv_folds=0, ppc_reps=50, profile=False)
    assert rep.verdict_counts()["fail"] == 0


def test_data_errors_are_explained():
    data, _ = simulate(SimConfig(n_students=20), seed=3)
    a, h, s = data.to_frames()
    with pytest.raises(ValueError, match="lacks the column"):
        EAGData.from_frames(a.drop(columns=["indicator"]), h)
    bad = a.copy()
    bad["score"] = bad["score"].astype(float)
    bad.loc[0, "score"] = 1.5
    with pytest.raises(ValueError, match="whole numbers"):
        EAGData.from_frames(bad, h)
