"""b29 = the b28 nested L2-LR stack (tfm_stack_gp2008.py, 156 members) + ext/b29/ public members.
Member loading is copied from tfm_stack_gp2008.py. Nested 5-fold OOF at C in (0.01, 0.03, 0.1), DeLong vs b28,
leak guard per new member (solo AUC, full-fit coefficient + rank, nested gain alone on top of b28), and a
with/without-flagged fit. Writes subs/b29_<tag>.csv for the best-OOF C of the candidate set (no submit)."""
import sys, glob, os, json, time, re, numpy as np, pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auc
sys.path.insert(0, "models"); sys.path.insert(0, "scripts")
from features import folds
from delong import delong_paired
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)

y = pd.read_csv("data/train.csv", usecols=["satisfaction"]).satisfaction.astype(str).eq("True").astype(int).values
fold = np.zeros(len(y), int)
for i, (_, b) in enumerate(folds(y)): fold[b] = i
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))
te = pd.read_csv("data/test.csv", usecols=["id"]); ss = pd.read_csv("data/sample_submission.csv")
assert len(te) == len(ss) == 299844 and (te.id.values == ss.id.values).all(), "row/id mismatch vs sample_submission"

# ---- member loading: identical to tfm_stack_gp2008.py ----
b26 = json.load(open("logs/b26_k20all.json"))["weights"]
O, T = {}, {}
for n in b26: O[n] = np.load(f"oof/{n[4:]}.npy"); T[n] = np.load(f"oof/{n[4:]}_test.npy")
for f in sorted(glob.glob("ext/tfm/oof_*.npy")):
    k = "tfm:" + os.path.basename(f)[4:-4]; O[k] = np.load(f).astype(float); T[k] = np.load(f.replace("oof_", "test_")).astype(float)
for f in sorted(glob.glob("ext/pub124/oof_*.npy")):
    a = np.load(f).astype(float); t = np.load(f.replace("/oof_", "/test_")).astype(float)
    if a.shape != y.shape or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1 or auc(y, a) < 0.9: print("skip", f, a.shape); continue
    k = "pub:" + os.path.basename(f)[4:-4]; O[k] = a; T[k] = t
for f in sorted(glob.glob("ext/gp2008/oof_*.npy")):
    n = os.path.basename(f)[4:-4]
    if n == "stack245" or "k10" in n or n in ("glm6_xgb_k",): continue
    O["gp:" + n] = np.load(f).astype(float); T["gp:" + n] = np.load(f.replace("/oof_", "/test_")).astype(float)
base = list(O); assert len(base) == 156, f"b28 member set changed: {len(base)} != 156"
# ---- b29 additions: ext/b29/ members, same sanity rule as the b28 loader, plus a test-side check ----
for f in sorted(glob.glob("ext/b29/oof_*.npy")):
    a = np.load(f).astype(float); t = np.load(f.replace("/oof_", "/test_")).astype(float)
    if a.shape != y.shape or t.shape != (len(te),) or not (np.isfinite(a).all() and np.isfinite(t).all()) \
            or a.min() < 0 or a.max() > 1 or t.min() < 0 or t.max() > 1 or auc(y, a) < 0.9:
        print("skip", f, a.shape, t.shape); continue
    k = "b29:" + os.path.basename(f)[4:-4]; O[k] = a; T[k] = t
new = [n for n in O if n.startswith("b29:")]
log(len(base), "base members", len(new), "new members:", new)

def prep(names):
    X = np.c_[[L(O[n]) for n in names]].T; XT = np.c_[[L(T[n]) for n in names]].T
    m_, s_ = X.mean(0), X.std(0); return (X - m_) / s_, (XT - m_) / s_

def nested(X, C):
    o = np.zeros(len(y))
    for f in range(5):
        tr = fold != f; o[~tr] = LogisticRegression(C=C, max_iter=500).fit(X[tr], y[tr]).decision_function(X[~tr])
    return auc(y, o), o

# ---- reproduce b28 ----
b28_oof = np.load("logs/gp2008_best_oof.npy")
X, _ = prep(base)
a28, o28 = nested(X, 0.03)
log(f"b28 reproduction: nested C=0.03 {a28:.6f} (expected 0.962149); logged OOF AUC {auc(y, b28_oof):.6f}; "
    f"max |diff| of OOF vs logs/gp2008_best_oof.npy {np.abs(o28 - b28_oof).max():.2e}")
del X

# ---- leak guard: each new member alone on top of b28 (nested C=0.03) ----
guard = {}
for n in new:
    Xn, _ = prep(base + [n]); an, _ = nested(Xn, 0.03); del Xn
    guard[n] = {"solo_auc": auc(y, O[n]), "gain_alone": an - a28}
    log(f"guard {n}: solo {guard[n]['solo_auc']:.6f} gain_alone {guard[n]['gain_alone']:+.6f}")

# ---- full set: nested grid + full-fit coefficients at C=0.03 ----
GRID = (0.01, 0.03, 0.1)
def run_set(names):
    X, XT = prep(names); res = {}
    for C in GRID:
        a, o = nested(X, C); res[C] = (a, o); log(f"{len(names)} members C={C}: nested {a:.6f}")
    return res, X, XT
full_names = base + new
res_full, Xf, XTf = run_set(full_names)
coef = LogisticRegression(C=0.03, max_iter=500).fit(Xf, y).coef_.ravel()
order = np.argsort(-coef); rank = {full_names[i]: r + 1 for r, i in enumerate(order)}
for n in new:
    guard[n]["coef"] = float(coef[full_names.index(n)]); guard[n]["coef_rank"] = rank[n]
    log(f"coef {n}: {guard[n]['coef']:+.4f} rank {rank[n]} of {len(full_names)} (1 = largest)")
NAME_RE = re.compile(r"target|route|_te\b|^te_|encod|label", re.I)
flagged = [n for n in new if guard[n]["coef_rank"] == 1 or guard[n]["gain_alone"] > 1e-4 or NAME_RE.search(n)]
log("flagged:", flagged or "none")
del Xf, XTf

# ---- without-flagged set (if any member was flagged) ----
if flagged:
    nf_names = base + [n for n in new if n not in flagged]
    res_nf, _, _ = run_set(nf_names)
else:
    nf_names, res_nf = full_names, res_full
cand_names = nf_names  # gate uses the version without flagged members (identical to full when none flagged)
cand_C, (cand_auc, cand_oof) = max(res_nf.items(), key=lambda kv: kv[1][0])
log(f"candidate set {len(cand_names)} members, best C={cand_C}, nested {cand_auc:.6f}")

# ---- DeLong vs b28 (candidate and full set) ----
a_b28, a_c, sd = delong_paired(y, o28, cand_oof)
z_cand = (a_c - a_b28) / sd
a_b28f, a_f, sdf = delong_paired(y, o28, res_full[cand_C][1])
z_full = (a_f - a_b28f) / sdf
log(f"DeLong candidate vs b28: auc {a_b28:.6f} -> {a_c:.6f}, sd {sd:.6g}, z {z_cand:+.3f}")
log(f"DeLong full-set (incl. flagged) vs b28 at C={cand_C}: z {z_full:+.3f}")

# ---- final fit + submission file (candidate set, best-OOF C) ----
Xc, XTc = prep(cand_names)
t = LogisticRegression(C=cand_C, max_iter=500).fit(Xc, y).decision_function(XTc)
p = (rankdata(t) - .5) / len(t)
b28csv = pd.read_csv("subs/b28_lr_tfm_gp2008.csv")
assert (b28csv.id.values == te.id.values).all(), "b28 id order differs"
rho = spearmanr(p, b28csv.satisfaction.values)[0]
tag = "lr_tfm_gp2008_tabm"
out = f"subs/b29_{tag}.csv"
pd.DataFrame({"id": te.id, "satisfaction": p}).to_csv(out, index=False)
np.save("logs/b29_best_oof.npy", cand_oof); np.save("logs/b29_best_test.npy", t)
log(f"wrote {out} {len(p)} rows; spearman vs b28 test preds {rho:.6f}")

GATE_OOF = 0.962149
gate = cand_auc > GATE_OOF and z_cand >= 2
summary = {"base_members": len(base), "new_members": new, "flagged": flagged, "guard": guard,
           "oof_by_C_full": {str(c): res_full[c][0] for c in GRID},
           "oof_by_C_candidate": {str(c): res_nf[c][0] for c in GRID},
           "candidate_names_n": len(cand_names), "cand_C": cand_C, "cand_oof": cand_auc,
           "b28_repro": a28, "delong_z_cand": z_cand, "delong_z_full": z_full, "spearman_vs_b28": rho,
           "csv": out, "gate_pass": bool(gate)}
json.dump(summary, open("logs/b29_summary.json", "w"), indent=1, default=float)
log("GATE", "PASS" if gate else "FAIL", f"(oof {cand_auc:.6f} > {GATE_OOF}: {cand_auc > GATE_OOF}; z {z_cand:+.3f} >= 2: {z_cand >= 2})")
