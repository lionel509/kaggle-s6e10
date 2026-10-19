"""Fast paired DeLong (Sun & Xu 2014), O(n log n). delong_paired(y, a, b) -> (auc_a, auc_b, sd_of_diff).
scale: multiply the covariance by n/scale to get the sd at a smaller test size (e.g. scale=60000 for the public LB)."""
import numpy as np
from scipy.stats import rankdata


def delong_paired(y, a, b, scale=None):
    y = np.asarray(y).astype(bool); P = np.vstack([a, b]).astype(float)
    pos, neg = P[:, y], P[:, ~y]; m, n = pos.shape[1], neg.shape[1]
    tz = np.array([rankdata(np.r_[p, q]) for p, q in zip(pos, neg)])
    tx = np.array([rankdata(p) for p in pos]); ty = np.array([rankdata(q) for q in neg])
    auc = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if scale: var *= (m + n) / scale
    return auc[0], auc[1], float(np.sqrt(max(var, 0)))
