"""v4: CatBoost on v1's frame + nested TE, with Flight Distance and Age also as native cat features (ordered target stats).
Usage: python v4_catboost.py [seed=42]"""
import json, os, sys, time, numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "models"); from features import load, folds, label_free, add_te, CAT, FD, AGE

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42; name = f"v4_cat_s{SEED}"; t0 = time.time()
tr, te, og, y, og_y = load(); X, XT = label_free(tr, te, og, og_y)
if os.environ.get("EXTRA"):
    from features import extra; E, ET = extra(tr, te); X, XT = pd.concat([X, E], axis=1), pd.concat([XT, ET], axis=1); name += "_extra"
if os.environ.get("TEACH"):
    from features import teachers; E, ET = teachers(tr, te, og, og_y); X, XT = pd.concat([X, E], axis=1), pd.concat([XT, ET], axis=1); name += "_teach"
for D, src in ((X, tr), (XT, te)):
    D["fd_c"], D["age_c"] = src[FD].astype(str).values, src[AGE].astype(str).values
    for c in CAT: D[c] = D[c].astype(str)
cats = CAT + ["fd_c", "age_c"]
P = dict(iterations=20000, learning_rate=0.06, depth=7, l2_leaf_reg=5, eval_metric="AUC", od_type="Iter", od_wait=300, random_seed=SEED, thread_count=int(os.environ.get("THREADS", 8)), verbose=0)
oof, pt, scores, its = np.zeros(len(y)), np.zeros(len(XT)), [], []
for i, (a, b) in enumerate(folds(y)):
    Xa, Xb, Xt = add_te(X.iloc[a], X.iloc[b], XT, y[a])
    m = CatBoostClassifier(**P).fit(Xa, y[a], cat_features=cats, eval_set=(Xb, y[b]), use_best_model=True)
    oof[b] = m.predict_proba(Xb)[:, 1]; pt += m.predict_proba(Xt)[:, 1] / 5
    scores.append(roc_auc_score(y[b], oof[b])); its.append(m.best_iteration_); print(f"fold {i} {scores[-1]:.6f} it {its[-1]} ({time.time()-t0:.0f}s)", flush=True)
auc = roc_auc_score(y, oof); print(f"OOF {auc:.6f}  ({time.time()-t0:.0f}s)")
np.save(f"oof/{name}.npy", oof); np.save(f"oof/{name}_test.npy", pt)
json.dump({"name": name, "oof": auc, "folds": scores, "iters": its, "params": P}, open(f"logs/{name}.json", "w"), indent=1, default=int)
