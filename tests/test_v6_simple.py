"""v6 members (kNN, one-hot logistic): OOF covers every row, no NaN, test averaged."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "models")
from v6_simple import run


def test_oof_shape_no_nan():
    r = np.random.default_rng(0); n = 400
    X = pd.DataFrame({"t": r.normal(size=n), "q1": r.integers(0, 6, n), "q2": r.integers(0, 6, n)})
    y = (X.t + r.normal(size=n) > 0).astype(int).values
    for kind in ("knn", "lr"):
        oof, pt = run(kind, X, X.iloc[:50], y, ["q1", "q2"], K=4, k=20)
        assert oof.shape == (n,) and pt.shape == (50,) and not np.isnan(oof).any() and not np.isnan(pt).any()
