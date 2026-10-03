"""Stackers from the S6E9 winners, nested by fold: (a) L2 logistic on member logits, (b) L1 logistic on within-fold ranks.
Usage: python blend/stack.py   (OOF AUC + split-half vs the probit blend; no submission written)"""
import sys, glob, numpy as np
from scipy.stats import rankdata


def nested_oof(X, y, fold, make):
    p = np.empty(len(y))
    for f in np.unique(fold):
        m = make().fit(X[fold != f], y[fold != f]); p[fold == f] = m.predict_proba(X[fold == f])[:, 1]
    return p


def logits(M):
    M = np.clip(M, 1e-6, 1 - 1e-6); return np.log(M / (1 - M))


def fold_ranks(M, fold):
    R = np.empty(M.shape)
    for f in np.unique(fold): R[fold == f] = rankdata(M[fold == f], axis=0) / (fold == f).sum()
    return R


if __name__ == "__main__":
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, "blend"); sys.path.insert(0, "models")
    from blend import fold_rank_probit, fit_weights
    from features import folds
    y = pd.read_csv("data/train.csv", usecols=["satisfaction"]).satisfaction.astype(str).eq("True").astype(int).values
    fold = np.zeros(len(y), int)
    for i, (a, b) in enumerate(folds(y)): fold[b] = i
    names = [f[4:-9] for f in sorted(glob.glob("oof/*_test.npy"))]
    M = np.c_[[np.load(f"oof/{n}.npy") for n in names]].T
    l2 = lambda: LogisticRegression(C=1.0, max_iter=2000)
    l1 = lambda: LogisticRegression(penalty="l1", C=0.05, solver="liblinear")
    h = np.arange(len(y)) % 2 == 0
    for tag, X, mk in (("L2 logit", logits(M), l2), ("L1 rank", fold_ranks(M, fold), l1)):
        p = nested_oof(X, y, fold, mk)
        print(f"{tag:9s} OOF {roc_auc_score(y, p):.6f}  halves {roc_auc_score(y[h], p[h]):.6f} {roc_auc_score(y[~h], p[~h]):.6f}", flush=True)
    Z = fold_rank_probit(M, fold); w = fit_weights(Z, y); p = Z @ w
    print(f"probit CD OOF {roc_auc_score(y, p):.6f}  halves {roc_auc_score(y[h], p[h]):.6f} {roc_auc_score(y[~h], p[~h]):.6f}")
