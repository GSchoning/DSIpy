"""Checks of DSIpy: run from the repository root with  python tests/test_dsipy.py  or  python -m pytest tests -q"""
import os, sys, io, contextlib, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from DSIpy import DSISurrogate, NormalScoreTransformer

quiet = lambda: contextlib.redirect_stdout(io.StringIO())       # noqa: E731  DSIpy prints its progress


def _toy(n=3000, scale=1e-12, seed=0):
    """A linear-Gaussian prior: 2 parameters, 6 positive observations of magnitude ~scale (log-linear in the
    parameters), the parameters as predictions."""
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((n, 2))
    B = np.array([[1.0, 0.2], [0.8, -0.5], [0.3, 1.0], [-0.4, 0.9], [0.6, 0.6], [0.1, -1.0]])
    return p, scale * 10 ** (p @ B.T * 0.5), B


def test_normal_score_matches_sklearn_on_ordinary_values():
    x = np.random.default_rng(1).lognormal(size=(4000, 3))
    z_new = NormalScoreTransformer(n_quantiles=1000).fit_transform(x)
    z_skl = QuantileTransformer(output_distribution='normal', n_quantiles=1000, subsample=10**9).fit_transform(x)
    assert np.abs(z_new - z_skl).max() < 1e-6


def test_normal_score_works_at_small_magnitudes():
    """sklearn's QuantileTransformer maps every value of a column below ~1e-7 to the same score; this one does not."""
    x = np.exp(np.random.default_rng(2).uniform(np.log(1e-15), np.log(1e-8), (5000, 2)))
    nst = NormalScoreTransformer().fit(x)
    z = nst.transform(x)
    assert np.all(np.abs(z.std(axis=0) - 1.0) < 0.05)
    back = nst.inverse_transform(z)
    assert np.median(np.abs(back / x - 1.0)) < 0.01


def test_log_floor_warns_once_and_suggests_epsilon():
    p, obs, _ = _toy()
    with warnings.catch_warnings(record=True) as w, quiet():
        warnings.simplefilter('always')
        dsi = DSISurrogate(obs_transform='log', pred_transform='none').fit(obs, p)           # default epsilon 1e-10
        dsi._transform_data(obs, kind='obs')
    msgs = [str(m.message) for m in w if 'log transform of obs' in str(m.message)]
    assert len(msgs) == 1 and 'below epsilon' in msgs[0]
    with warnings.catch_warnings(record=True) as w, quiet():
        warnings.simplefilter('always')
        DSISurrogate(obs_transform='log', pred_transform='none', epsilon=1e-30).fit(obs, p)
    assert not [m for m in w if 'log transform' in str(m.message)]


def test_constant_observation_gives_finite_posterior():
    """A column that is constant across the prior used to divide the noise by a zero variance (NaN in ES)."""
    p, obs, _ = _toy()
    obs = np.c_[obs, np.full(len(obs), 3.0)]
    with warnings.catch_warnings(record=True) as w, quiet():
        warnings.simplefilter('always')
        dsi = DSISurrogate(obs_transform='log', pred_transform='none', epsilon=1e-30).fit(obs, p)
        mean, std, _ = dsi.predict(obs[0], 0.05 * obs[0], inversion_type='es', n_posterior_samples=200)
    assert any('constant' in str(m.message) for m in w)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(std))


def test_small_magnitude_data_recover_the_parameters():
    """End to end at 1e-12: with epsilon below the data (log) or the new normal score, the posterior mean is
    close to the truth; with sklearn's quantile transform the normal-score run used to end in NaN."""
    p, obs, _ = _toy()
    truth, h = p[0], obs[0]
    np.random.seed(0)
    for kw in (dict(obs_transform='log', epsilon=1e-30), dict(obs_transform='normal_score')):
        with warnings.catch_warnings(), quiet():
            warnings.simplefilter('ignore')
            dsi = DSISurrogate(pred_transform='none', **kw).fit(obs[1:], p[1:])
            mean, std, _ = dsi.predict(h, 0.01 * h, inversion_type='es', n_posterior_samples=500)
        assert np.all(np.isfinite(mean)) and np.all(np.abs(mean - truth) < 3 * std + 0.05), (kw, mean, truth, std)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
