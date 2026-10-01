"""Pins blend/blend.py: E9's blender - rank each model inside its own fold, map to probit space, fit simplex weights by coordinate descent on OOF AUC."""
import sys, numpy as np, pytest
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "blend")
from blend import fold_rank_probit, fit_weights, blend


def _data(n=4000, seed=0):
    r = np.random.default_rng(seed); y = r.integers(0, 2, n)
    good = y + r.normal(0, 1.0, n); ok = y + r.normal(0, 2.0, n); junk = r.normal(0, 1, n)
    fold = np.arange(n) % 4
    return y, fold, np.c_[good, ok * 100 + 7, junk]  # different scales on purpose


def test_rank_probit_is_per_fold_and_bounded():
    y, fold, M = _data(); Z = fold_rank_probit(M, fold)
    assert Z.shape == M.shape and np.isfinite(Z).all()
    for f in range(4):  # inside each fold every column is a probit of uniform ranks: mean ~0, same spread across columns
        z = Z[fold == f]
        assert np.allclose(z.mean(0), 0, atol=1e-9)
        assert np.allclose(z.std(0), z.std(0)[0])
        # order inside a fold is preserved
        assert (np.argsort(z[:, 1]) == np.argsort(M[fold == f, 1])).all()
    # monotone shift of one fold must not change the result
    M2 = M.copy(); M2[fold == 2, 0] = M2[fold == 2, 0] * 3 + 50
    assert np.allclose(fold_rank_probit(M2, fold), Z)


def test_fit_weights_on_simplex_and_beats_best_single():
    y, fold, M = _data(); Z = fold_rank_probit(M, fold)
    w = fit_weights(Z, y)
    assert w.shape == (3,) and (w >= 0).all() and np.isclose(w.sum(), 1)
    assert w[2] < 0.05 and w[0] > w[1]  # junk gets ~nothing, the good model gets most
    assert roc_auc_score(y, Z @ w) >= max(roc_auc_score(y, Z[:, j]) for j in range(3)) - 1e-12


def test_blend_applies_weights_to_test_ranked_as_one_block():
    y, fold, M = _data(); w = np.array([0.7, 0.3, 0.0])
    T = M[:500].copy()
    out = blend(T, w)
    assert out.shape == (500,) and np.isfinite(out).all()
    Zt = norm.ppf((np.argsort(np.argsort(T, axis=0), axis=0) + 0.5) / 500)
    assert np.allclose(out, Zt @ w)
