"""v7: original rows as weighted *training rows* (kaggle-s6e10#1 idea 2). Raw features only, so no og-label-derived
feature can leak into og rows. OOF is scored on synthetic rows only.  Usage: W=<og weight, 0 = off> K=5 python v7_rows.py"""
import json, os, sys, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "models"); from features import load, folds

W, K = float(os.environ.get("W", 0)), int(os.environ.get("K", 5)); name = f"v7_rows_w{W:g}" + (f"_k{K}" if K != 5 else ""); t0 = time.time()
tr, te, og, y, og_y = load(); raw = [c for c in tr.columns if c != "id" and c in og.columns]
cat = [c for c in raw if tr[c].dtype == object]
X, XT, XO = (pd.concat([d[raw] for d in (tr, te, og)], keys=range(3)).astype({c: "category" for c in cat}).xs(i) for i in range(3))
P = dict(n_estimators=10000, learning_rate=0.03, num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
         reg_lambda=3, verbose=-1, random_state=42, n_jobs=int(os.environ.get("THREADS", 8)))
oof, pt, scores = np.zeros(len(y)), np.zeros(len(XT)), []
for i, (a, b) in enumerate(folds(y, K)):
    Xa, ya, wa = X.iloc[a], y[a], np.ones(len(a))
    if W: Xa, ya, wa = pd.concat([Xa, XO]), np.r_[ya, og_y], np.r_[wa, np.full(len(XO), W)]
    m = lgb.LGBMClassifier(**P).fit(Xa, ya, sample_weight=wa, eval_set=[(X.iloc[b], y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[b] = m.predict_proba(X.iloc[b])[:, 1]; pt += m.predict_proba(XT)[:, 1] / K
    scores.append(roc_auc_score(y[b], oof[b])); print(f"fold {i} {scores[-1]:.6f} it {m.best_iteration_}", flush=True)
auc = roc_auc_score(y, oof); print(f"OOF {auc:.6f}  ({time.time()-t0:.0f}s)")
np.save(f"oof/{name}.npy", oof); np.save(f"oof/{name}_test.npy", pt)
json.dump({"name": name, "oof": auc, "folds": scores, "W": W}, open(f"logs/{name}.json", "w"), indent=1)
