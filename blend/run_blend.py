"""Blend own + public members: fold-rank probit, coordinate-descent weights, split-half check vs the incumbent set.
Usage: python blend/run_blend.py <out_name> [--own-only] [--drop REGEX]"""
import sys, glob, json, numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "blend"); sys.path.insert(0, "models")
from blend import fold_rank_probit, fit_weights, blend
from public_lib import load_public
from features import folds

tr = pd.read_csv("data/train.csv", usecols=["id", "satisfaction"]); te = pd.read_csv("data/test.csv", usecols=["id"])
y = tr.satisfaction.astype(str).eq("True").astype(int).values
fold = np.zeros(len(y), int)
for i, (a, b) in enumerate(folds(y)): fold[b] = i
oof, tst = {}, {}
for f in sorted(glob.glob("oof/*_test.npy")):
    n = f[4:-9]; oof["own:" + n], tst["own:" + n] = np.load(f"oof/{n}.npy"), np.load(f)
if "--own-only" not in sys.argv:
    po, pt = load_public(tr.id.values, te.id.values); oof |= po; tst |= pt
import re  # --drop REGEX: leave out members (family-restricted hedge blends)
drop = sys.argv[sys.argv.index("--drop") + 1] if "--drop" in sys.argv else None
names = [n for n in oof if not (drop and re.search(drop, n))]; M = np.c_[[oof[n] for n in names]].T; T = np.c_[[tst[n] for n in names]].T
for n in names: print(f"  {roc_auc_score(y, oof[n]):.6f}  {n}")
Z = fold_rank_probit(M, fold); w = fit_weights(Z, y); auc = roc_auc_score(y, Z @ w)
print(f"\nBLEND OOF {auc:.6f}"); [print(f"  {w[i]:.3f}  {names[i]}") for i in np.argsort(-w) if w[i] > 0]
h = np.arange(len(y)) % 2 == 0
for A, B in ((h, ~h), (~h, h)):
    print("  split-half held-out", round(roc_auc_score(y[B], Z[B] @ fit_weights(Z[A], y[A])), 6))
name = sys.argv[1]
pd.DataFrame({"id": te.id, "satisfaction": norm.cdf(blend(T, w))}).to_csv(f"subs/{name}.csv", index=False)
json.dump({"oof": auc, "weights": {n: float(w[i]) for i, n in enumerate(names) if w[i] > 0}}, open(f"logs/{name}.json", "w"), indent=1)
