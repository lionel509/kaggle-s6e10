"""S6E10 2026-10-19: b26 + Deotte TFM OOFs. Nested-by-fold L2 logistic regression on logits (C sweep) + the existing fold-rank-probit CD stacker.
Run from project root: nohup nice uv run python scripts/tfm_stack.py > logs/tfm_stack.log 2>&1 &"""
import sys, glob, json, os, numpy as np, pandas as pd
from scipy.stats import norm
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auc
sys.path.insert(0, "blend"); sys.path.insert(0, "models"); sys.path.insert(0, "scripts")
from blend import fold_rank_probit, fit_weights, blend
from features import folds
from delong import delong_paired

y = pd.read_csv("data/train.csv", usecols=["satisfaction"]).satisfaction.astype(str).eq("True").astype(int).values
fold = np.zeros(len(y), int)
for i, (_, b) in enumerate(folds(y)): fold[b] = i
b26 = json.load(open("logs/b26_k20all.json"))["weights"]
own_o = {n: np.load(f"oof/{n[4:]}.npy") for n in b26}; own_t = {n: np.load(f"oof/{n[4:]}_test.npy") for n in b26}
tfm_o = {f"tfm:{os.path.basename(f)[4:-4]}": np.load(f).astype(float) for f in sorted(glob.glob("ext/tfm/oof_*.npy"))}
tfm_t = {k: np.load(f"ext/tfm/test_{k[4:]}.npy").astype(float) for k in tfm_o}
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))
res = {}
# baseline b26 (full-OOF-fitted weights, as submitted)
names = list(own_o); M = np.c_[[own_o[n] for n in names]].T
Zo = fold_rank_probit(M, fold); w0 = np.array([b26[n] for n in names]); s_b26 = Zo @ w0
print("b26 OOF", auc(y, s_b26)); res["b26"] = auc(y, s_b26)

def nested_lr(X, C):
    o = np.zeros(len(y))
    for f in range(5):
        tr = fold != f; m = LogisticRegression(C=C, max_iter=500).fit(X[tr], y[tr]); o[~tr] = m.decision_function(X[~tr])
    return o
mu = lambda X: (X - X.mean(0)) / X.std(0)
Xown = mu(np.c_[[L(own_o[n]) for n in names]].T); Xtfm = mu(np.c_[[L(tfm_o[n]) for n in tfm_o]].T)
cands = {}
for tag, X in (("own", Xown), ("own+tfm", np.c_[Xown, Xtfm]), ("tfm", Xtfm)):
    for C in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
        o = nested_lr(X, C); cands[f"lr_{tag}_C{C}"] = o; print(f"lr {tag} C={C}: nested OOF {auc(y, o):.6f}", flush=True)
# CD stacker: own members + TFMs, fold-rank probit
allo = own_o | tfm_o; an = list(allo); Ma = np.c_[[allo[n] for n in an]].T
Za = fold_rank_probit(Ma, fold); wa = fit_weights(Za, y); s_cd = Za @ wa
print("CD own+tfm (weights fit on full OOF, optimistic):", auc(y, s_cd)); cands["cd_own+tfm"] = s_cd
print({an[i]: round(wa[i], 3) for i in np.argsort(-wa) if wa[i] > 0})
# honest CD: nested by fold
o = np.zeros(len(y))
for f in range(5):
    tr = fold != f; w = fit_weights(Za[tr], y[tr]); o[~tr] = Za[~tr] @ w
cands["cd_own+tfm_nested"] = o; print("CD own+tfm nested:", auc(y, o))
o = np.zeros(len(y))
for f in range(5):
    tr = fold != f; w = fit_weights(Zo[tr], y[tr]); o[~tr] = Zo[~tr] @ w
cands["cd_own_nested"] = o; print("CD own-only (b26 recipe) nested:", auc(y, o), flush=True)
print("\nDeLong vs b26 (sd full OOF | sd at n=60000):")
for k, s in sorted(cands.items(), key=lambda kv: -auc(y, kv[1]))[:12]:
    a, b, sd = delong_paired(y, s_b26, s); _, _, sd60 = delong_paired(y, s_b26, s, scale=60000)
    print(f"{k:28s} auc {b:.6f} diff {b-a:+.6f} sd {sd:.6f} z {(b-a)/sd:+.2f} | sd60k {sd60:.6f} z60k {(b-a)/sd60:+.2f}")
np.savez("logs/tfm_stack_cands.npz", **cands)
