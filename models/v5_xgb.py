"""v5: XGBoost (hist, native categoricals) on v1's frame + nested TE.  Usage: python v5_xgb.py [seed=42] [depth=8]"""
import json, os, sys, time, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "models"); from features import load, folds, label_free, add_te

SEED, DEPTH = (int(a) for a in (sys.argv[1:] + ["42", "8"][len(sys.argv) - 1:])[:2])
K = int(os.environ.get("K", 5)); name = f"v5_xgb_d{DEPTH}_s{SEED}" + (f"_k{K}" if K != 5 else ""); t0 = time.time()
tr, te, og, y, og_y = load(); X, XT = label_free(tr, te, og, og_y)
if os.environ.get("EXTRA"):
    from features import extra; E, ET = extra(tr, te); X, XT = pd.concat([X, E], axis=1), pd.concat([XT, ET], axis=1); name += "_extra"
if os.environ.get("TEACH"):
    from features import teachers; E, ET = teachers(tr, te, og, og_y); X, XT = pd.concat([X, E], axis=1), pd.concat([XT, ET], axis=1); name += "_teach" + ("2" if os.environ["TEACH"] == "2" else "")
P = dict(n_estimators=20000, learning_rate=0.02, max_depth=DEPTH, min_child_weight=20, subsample=0.8, colsample_bytree=0.5, reg_lambda=5,
         tree_method="hist", enable_categorical=True, max_cat_to_onehot=4, eval_metric="auc", early_stopping_rounds=300, random_state=SEED, n_jobs=8)
oof, pt, scores, its = np.zeros(len(y)), np.zeros(len(XT)), [], []
for i, (a, b) in enumerate(folds(y, K)):
    Xa, Xb, Xt = add_te(X.iloc[a], X.iloc[b], XT, y[a])
    m = xgb.XGBClassifier(**P).fit(Xa, y[a], eval_set=[(Xb, y[b])], verbose=False)
    oof[b] = m.predict_proba(Xb)[:, 1]; pt += m.predict_proba(Xt)[:, 1] / K
    scores.append(roc_auc_score(y[b], oof[b])); its.append(m.best_iteration); print(f"fold {i} {scores[-1]:.6f} it {its[-1]} ({time.time()-t0:.0f}s)", flush=True)
auc = roc_auc_score(y, oof); print(f"OOF {auc:.6f}  ({time.time()-t0:.0f}s)")
np.save(f"oof/{name}.npy", oof); np.save(f"oof/{name}_test.npy", pt)
json.dump({"name": name, "oof": auc, "folds": scores, "iters": its, "params": P}, open(f"logs/{name}.json", "w"), indent=1, default=int)
