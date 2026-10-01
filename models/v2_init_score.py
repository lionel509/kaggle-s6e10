"""v2: E9's winning model (v34) ported - v1's frame, but boosting starts from logit(prior) instead of the base rate.
KEY = te column(s) joined by '+' (mean of their logits), or 'teacher' (public notebooks report teacher-logit boosting hurts; measured here anyway).
Usage: python v2_init_score.py [n_folds=5] [seed=42] [key=te_fd]"""
import json, sys, time, numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import logit
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "models"); from features import load, folds, label_free, add_te

K, SEED, KEY = int(sys.argv[1]) if len(sys.argv) > 1 else 5, int(sys.argv[2]) if len(sys.argv) > 2 else 42, sys.argv[3] if len(sys.argv) > 3 else "te_fd"
name = f"v2_init_{KEY}_k{K}_s{SEED}"; t0 = time.time()
tr, te, og, y, og_y = load(); X, XT = label_free(tr, te, og, og_y)
ini = lambda D: np.mean([logit(np.clip(D[k].to_numpy(float), 1e-4, 1 - 1e-4)) for k in KEY.split("+")], axis=0)
oof, pt, scores, its = np.zeros(len(y)), np.zeros(len(XT)), [], []
P = dict(n_estimators=10000, learning_rate=0.03, num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=3, verbose=-1, random_state=SEED)
for i, (a, b) in enumerate(folds(y, K, 42)):  # split stays fixed at seed 42; SEED varies the model only
    Xa, Xb, Xt = add_te(X.iloc[a], X.iloc[b], XT, y[a])
    m = lgb.LGBMClassifier(**P).fit(Xa, y[a], init_score=ini(Xa), eval_set=[(Xb, y[b])], eval_init_score=[ini(Xb)], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[b] = m.predict_proba(Xb, raw_score=True) + ini(Xb); pt += (m.predict_proba(Xt, raw_score=True) + ini(Xt)) / K
    scores.append(roc_auc_score(y[b], oof[b])); its.append(m.best_iteration_); print(f"fold {i} {scores[-1]:.6f} it {its[-1]}", flush=True)
auc = roc_auc_score(y, oof); print(f"OOF {auc:.6f}  ({time.time()-t0:.0f}s)")
np.save(f"oof/{name}.npy", oof); np.save(f"oof/{name}_test.npy", pt)
json.dump({"name": name, "oof": auc, "folds": scores, "iters": its, "params": P, "key": KEY}, open(f"logs/{name}.json", "w"), indent=1, default=int)
