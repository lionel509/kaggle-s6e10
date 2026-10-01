"""E9's blender: rank each model inside its own fold, probit, simplex weights by coordinate descent on OOF AUC."""
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score


def _probit_rank(M):
    return norm.ppf((rankdata(M, axis=0, method="ordinal") - 0.5) / len(M))


def fold_rank_probit(M, fold):
    Z = np.empty(M.shape, float)
    for f in np.unique(fold):
        Z[fold == f] = _probit_rank(M[fold == f])
    return Z


def fit_weights(Z, y, steps=(0.2, 0.1, 0.05, 0.02, 0.01, 0.005), rounds=20):
    k = Z.shape[1]; aucs = [roc_auc_score(y, Z[:, j]) for j in range(k)]
    w = np.eye(k)[int(np.argmax(aucs))]; best = max(aucs)
    for s in steps:
        for _ in range(rounds):
            moved = False
            for i in range(k):  # shift mass s from every other member onto i, and back
                for j in range(k):
                    if i == j or w[j] < 1e-12: continue
                    v = w.copy(); d = min(s, v[j]); v[j] -= d; v[i] += d
                    a = roc_auc_score(y, Z @ v)
                    if a > best + 1e-12: w, best, moved = v, a, True
            if not moved: break
    return w


def blend(T, w):
    return _probit_rank(T) @ w
