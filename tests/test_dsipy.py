"""Tests for DSISurrogate: noise handling against exact Gaussian conditioning, normal-score refits, the example.

The reference throughout is exact linear-Gaussian conditioning on the ensemble's own sample mean and covariance
(PEST manual part II, eq. 18.1.4-18.1.5, with the noise covariance R added to C_dd):

    mean_h' = mean_h + C_hd (C_dd + R)^-1 (d_obs - mean_d)
    C_hh'   = C_hh - C_hd (C_dd + R)^-1 C_dh

With no transforms and no truncation, DSI's surrogate is exactly this Gaussian, so MAP must equal mean_h' and
RML / ES / ES-MDA samples must reproduce mean_h' and C_hh' to Monte Carlo accuracy.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DSIpy import DSISurrogate  # noqa: E402


# ---------------------------------------------------------------------------------------------- helpers

def linear_gaussian(n=3000, nobs=10, npred=2, k=6, seed=0, obs_scales=None, offset=50.0):
    """A jointly Gaussian prior ensemble: d = A z + mu_d, h = B z + mu_h, z ~ N(0, I_k). obs_scales gives the
    observations very different prior standard deviations (as heads at different wells have)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(nobs, k))
    if obs_scales is not None:
        A *= np.asarray(obs_scales)[:, None]
    B = rng.normal(size=(npred, k))
    Z = rng.normal(size=(n, k))
    D = Z @ A.T + offset
    H = Z @ B.T + 10.0
    zt = rng.normal(size=k)
    return D, H, A @ zt + offset, B @ zt + 10.0, rng


def conditional(D, H, dobs, sd):
    """Exact Gaussian conditioning on the sample mean/covariance of the ensemble (ddof=1)."""
    J = np.cov(np.hstack([D, H]).T)
    no = D.shape[1]
    Cdd, Chd, Chh = J[:no, :no], J[no:, :no], J[no:, no:]
    G = Chd @ np.linalg.inv(Cdd + np.diag(np.broadcast_to(sd, (no,)) ** 2))
    return H.mean(0) + G @ (dobs - D.mean(0)), Chh - G @ Chd.T


def fitted(D, H, **kw):
    kw = dict(dict(obs_transform="none", pred_transform="none", svd_variance=1.0), **kw)
    return DSISurrogate(**kw).fit(D, H)


# ---------------------------------------------------------------------------------------------- correctness


def test_map_equals_gaussian_conditioning_equal_obs_variances():
    """Observations with equal prior variance (so the scaled noise is the same for every observation)."""
    D, H, dobs, _, _ = linear_gaussian()
    D = (D - D.mean(0)) / D.std(0) * 2.0 + 50.0           # every observation: prior sd 2
    sd = 0.3
    m_ref, _ = conditional(D, H, dobs, sd)
    m, _, _ = fitted(D, H).predict(dobs, np.full(dobs.size, sd), inversion_type="map")
    np.testing.assert_allclose(m, m_ref, rtol=0, atol=1e-6 * np.abs(m_ref).max() + 1e-6)


@pytest.mark.parametrize("solver", ["cg", "ls"])
def test_iterative_solvers_match_analytical(solver):
    D, H, dobs, _, _ = linear_gaussian()
    s = fitted(D, H)
    hs = s.obs_scaler_.transform(dobs.reshape(1, -1)).ravel()
    x0 = np.random.default_rng(1).normal(size=s.surrogate_components_["n_components"])
    xa = s._solve_map_analytical(hs, 0.01, x0)
    xi = (s._solve_map_cg if solver == "cg" else s._solve_map_ls)(hs, 0.01, x0)
    np.testing.assert_allclose(xi, xa, rtol=1e-4, atol=1e-5)


def test_map_with_unequal_obs_variances():
    """Equal physical noise on observations with different prior spreads (heads at different wells). The scaled
    noise variance sd^2 / var_i then differs per observation; MAP and RML use its mean for every observation."""
    D, H, dobs, _, _ = linear_gaussian(obs_scales=np.geomspace(0.2, 20.0, 10))
    sd = 0.3
    m_ref, _ = conditional(D, H, dobs, sd)
    m, _, _ = fitted(D, H).predict(dobs, np.full(dobs.size, sd), inversion_type="map")
    np.testing.assert_allclose(m, m_ref, rtol=1e-3, atol=1e-3)


def test_map_with_heteroscedastic_noise():
    """Different noise standard deviations per observation (e.g. two instrument types)."""
    D, H, dobs, _, _ = linear_gaussian()
    D = (D - D.mean(0)) / D.std(0) * 2.0 + 50.0
    sd = np.r_[np.full(5, 0.05), np.full(5, 1.5)]
    m_ref, _ = conditional(D, H, dobs, sd)
    m, _, _ = fitted(D, H).predict(dobs, sd, inversion_type="map")
    np.testing.assert_allclose(m, m_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("method", ["rml", "es", "ies"])
def test_posterior_samples_match_gaussian_conditioning(method):
    """Equal obs variances and equal noise (the case every solver should get right): sample mean and sd within
    Monte Carlo error of the exact posterior."""
    D, H, dobs, _, _ = linear_gaussian(n=4000)
    D = (D - D.mean(0)) / D.std(0) * 2.0 + 50.0
    sd = 0.3
    m_ref, C_ref = conditional(D, H, dobs, sd)
    s_ref = np.sqrt(np.diag(C_ref))
    np.random.seed(3)
    m, s, _ = fitted(D, H).predict(dobs, np.full(dobs.size, sd), inversion_type=method, n_posterior_samples=4000,
                                   n_ies_iterations=4)
    assert np.all(np.abs(m - m_ref) < 4 * s_ref / np.sqrt(4000) + 1e-3), (m, m_ref)
    np.testing.assert_allclose(s, s_ref, rtol=0.08)


@pytest.mark.parametrize("method", ["rml", "es", "ies"])
def test_posterior_samples_unequal_obs_variances(method):
    D, H, dobs, _, _ = linear_gaussian(n=4000, obs_scales=np.geomspace(0.2, 20.0, 10))
    sd = 0.3
    m_ref, C_ref = conditional(D, H, dobs, sd)
    s_ref = np.sqrt(np.diag(C_ref))
    np.random.seed(3)
    m, s, _ = fitted(D, H).predict(dobs, np.full(dobs.size, sd), inversion_type=method, n_posterior_samples=4000,
                                   n_ies_iterations=4)
    assert np.all(np.abs(m - m_ref) < 4 * s_ref / np.sqrt(4000) + 1e-3), (m, m_ref)
    np.testing.assert_allclose(s, s_ref, rtol=0.08)


def test_normal_score_refit_uses_new_data():
    """Fitting the same instance again rebuilds the normal-score transformers from the new data."""
    rng = np.random.default_rng(0)
    s = DSISurrogate(obs_transform="normal_score", pred_transform="normal_score")
    D1, H1 = rng.lognormal(size=(300, 4)), rng.lognormal(size=(300, 1))
    s.fit(D1, H1)
    D2, H2 = D1 * 100.0, H1 * 100.0
    s.fit(D2, H2)
    assert abs(np.median(s._transform_data(D2, "obs"))) < 0.2
    assert abs(np.median(s._transform_data(H2, "pred"))) < 0.2


def test_example_uses_current_constructor():
    ex = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "unconfined aquifer",
                      "run_inversion.py")
    src = open(ex).read()
    assert "log_transform_obs" not in src and "obs_transform=" in src
