"""v4: CatBoost on v1's frame + nested TE, with Flight Distance and Age also as native cat features (ordered target stats).
Usage: python v4_catboost.py [seed=42]"""
import json, os, sys, time, numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "models"); from features import load, folds, label_free, add_te, CAT, FD, AGE

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42; name = f"v4_cat_s{SEED}"; K = int(os.environ.get("K", 5)); name += f"_k{K}" if K != 5 else ""; t0 = time.time()
assert not os.environ.get("BLKTE") and not os.environ.get("SOFT"), "XGB/RealMLP only"  # BLKTE would change the TE keys without changing the name
tr, te, og, y, og_y = load(); X, XT = label_free(tr, te, og, og_y)
if os.environ.get("EXTRA"):
    from features import extra; E, ET = extra(tr, te); X, XT = pd.concat([X, E], axis=1), pd.concat([XT, ET], axis=1); name += "_extra"
if os.environ.get("AUX"):
    from features import aux; E, ET = aux(tr, te); X, XT = pd.concat([X, E], axis=1), pd.concat([XT, ET], axis=1); name += "_aux"
if os.environ.get("TEACH"):
    from features import teachers; E, ET = teachers(tr, te, og, og_y); X, XT = pd.concat([X, E], axis=1), pd.concat([XT, ET], axis=1); name += "_teach" + ("" if os.environ["TEACH"] == "1" else os.environ["TEACH"])
for D, src in ((X, tr), (XT, te)):
    D["fd_c"], D["age_c"] = src[FD].astype(str).values, src[AGE].astype(str).values
    for c in CAT: D[c] = D[c].astype(str)
cats = CAT + ["fd_c", "age_c"]
if os.environ.get("CROSS"):  # 13 ratings x Class / Type of Travel / Customer Type as native cats (Shelton Wang's 39 crosses, +0.00028 on a bare CatBoost)
    R = [c for c in tr.columns if tr[c].dtype.kind == "i" and tr[c].max() == 5 and c != "id"]
    for D, src in ((X, tr), (XT, te)):
        for r in R:
            for g in ("Class", "Type of Travel", "Customer Type"): D[f"{r}__x__{g}"] = src[r].astype(str).values + "|" + src[g].astype(str).values
    cats += [f"{r}__x__{g}" for r in R for g in ("Class", "Type of Travel", "Customer Type")]; name += "_cross"; print("crosses", len(R) * 3, flush=True)
if os.environ.get("BLOCK"):  # a01 pairs: the original survey's two factor blocks + the full 13-rating pattern, as native cats (ordered TS on the competition labels)
    B = {"blk_conv": ["Inflight wifi service", "Gate location", "Departure/Arrival time convenient", "Ease of Online booking"],
         "blk_comf": ["Food and drink", "Inflight entertainment", "Cleanliness", "Seat comfort"],
         "blk_all": [c for c in tr.columns if tr[c].dtype.kind == "i" and tr[c].max() == 5 and c != "id"]}
    for D, src in ((X, tr), (XT, te)):
        for n, cols in B.items(): D[n] = src[cols].astype(str).agg("".join, axis=1).values
    cats += list(B); name += "_block"
SMOKE = int(os.environ.get("SMOKE") or 0); R = "smoke/" if SMOKE else ""  # smoke runs write only under smoke/
if SMOKE: X, XT, y = X.iloc[:SMOKE].reset_index(drop=True), XT.iloc[:SMOKE // 5].reset_index(drop=True), y[:SMOKE]
for d in ("oof/ckpt", "logs", "subs/hedge", "data/soft"): os.makedirs(R + d, exist_ok=True)
P = dict(iterations=50 if SMOKE else 20000, learning_rate=0.06, depth=7, l2_leaf_reg=5, eval_metric="AUC", od_type="Iter", od_wait=300, random_seed=SEED, thread_count=int(os.environ.get("THREADS", 8)), verbose=0)
if os.environ.get("REFIT"):  # E: refit on all rows at the mean best iteration; hedge = 50/50 with the fold-average test
    J = json.load(open(f"{R}logs/{name}.json")); nit = int(round(np.mean(J["iters"]))); Xa, _, Xt = add_te(X, X.iloc[:5], XT, y)
    m = CatBoostClassifier(**({k: v for k, v in P.items() if k not in ("od_type", "od_wait")} | {"iterations": nit})).fit(Xa, y, cat_features=cats)
    p = m.predict_proba(Xt)[:, 1]
    np.save(f"{R}subs/hedge/{name}_refit_test.npy", p); np.save(f"{R}subs/hedge/{name}_hedge_test.npy", 0.5 * p + 0.5 * np.load(f"{R}oof/{name}_test.npy"))
    print(f"REFIT {name} nit {nit}", flush=True); sys.exit(0)
oof, pt, scores, its = np.zeros(len(y)), np.zeros(len(XT)), [], []
for i, (a, b) in enumerate(folds(y, K)):
    ck = f"{R}oof/ckpt/{name}_f{i}.npz"  # per-fold checkpoint: a killed k20 run resumes instead of restarting
    if os.path.exists(ck): z = np.load(ck); ob, ptb, it = z["ob"], z["pt"], int(z["it"])
    else:
        Xa, Xb, Xt = add_te(X.iloc[a], X.iloc[b], XT, y[a])
        m = CatBoostClassifier(**P).fit(Xa, y[a], cat_features=cats, eval_set=(Xb, y[b]), use_best_model=True)
        ob, ptb, it = m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1], m.best_iteration_
        np.savez(ck, ob=ob, pt=ptb, it=it)
    oof[b] = ob; pt += ptb / K
    scores.append(roc_auc_score(y[b], oof[b])); its.append(it); print(f"fold {i} {scores[-1]:.6f} it {its[-1]} ({time.time()-t0:.0f}s)", flush=True)
auc = roc_auc_score(y, oof); print(f"OOF {auc:.6f}  ({time.time()-t0:.0f}s)")
np.save(f"{R}oof/{name}.npy", oof); np.save(f"{R}oof/{name}_test.npy", pt)
json.dump({"name": name, "oof": auc, "folds": scores, "iters": its, "params": P}, open(f"{R}logs/{name}.json", "w"), indent=1, default=int)
