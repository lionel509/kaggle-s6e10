"""Adds Deotte's 124 public OOFs to own(12)+TFM(11): nested L2-LR on logits, C sweep. Saves best cand + test preds."""
import sys, glob, os, json, numpy as np, pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auc
sys.path.insert(0, "models"); sys.path.insert(0, "scripts")
from features import folds
from delong import delong_paired
y = pd.read_csv("data/train.csv", usecols=["satisfaction"]).satisfaction.astype(str).eq("True").astype(int).values
fold = np.zeros(len(y), int)
for i, (_, b) in enumerate(folds(y)): fold[b] = i
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))
b26 = json.load(open("logs/b26_k20all.json"))["weights"]
O, T = {}, {}
for n in b26: O[n] = np.load(f"oof/{n[4:]}.npy"); T[n] = np.load(f"oof/{n[4:]}_test.npy")
for f in sorted(glob.glob("ext/tfm/oof_*.npy")):
    k = "tfm:" + os.path.basename(f)[4:-4]; O[k] = np.load(f).astype(float); T[k] = np.load(f.replace("oof_", "test_")).astype(float)
for f in sorted(glob.glob("ext/pub124/oof_*.npy")):
    a = np.load(f).astype(float); t = np.load(f.replace("/oof_", "/test_")).astype(float)
    if a.shape != y.shape or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1 or auc(y, a) < 0.9: print("skip", f, a.shape); continue
    k = "pub:" + os.path.basename(f)[4:-4]; O[k] = a; T[k] = t
names = list(O); print(len(names), "members", flush=True)
X = np.c_[[L(O[n]) for n in names]].T; XT = np.c_[[L(T[n]) for n in names]].T
m_, s_ = X.mean(0), X.std(0); X = (X - m_) / s_; XT = (XT - m_) / s_
best = (0, None)
for C in (0.001, 0.003, 0.01, 0.03):
    o = np.zeros(len(y))
    for f in range(5):
        tr = fold != f; o[~tr] = LogisticRegression(C=C, max_iter=500).fit(X[tr], y[tr]).decision_function(X[~tr])
    a = auc(y, o); print(f"all C={C}: nested {a:.6f}", flush=True)
    if a > best[0]: best = (a, C, o)
a, C, o = best; np.save("logs/pub124_best_oof.npy", o)
m = LogisticRegression(C=C, max_iter=500).fit(X, y); np.save("logs/pub124_best_test.npy", m.decision_function(XT))
print("best", a, C)
