"""v1: LightGBM on raw + counts + original means + route profile + digits + teacher + nested TE."""
import json, sys, time, numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "models"); from features import load, folds, label_free, add_te

name = "v1_lgb"; t0 = time.time()
tr, te, og, y, og_y = load(); X, XT = label_free(tr, te, og, og_y)
print("features", X.shape[1], "teacher AUC on train", round(roc_auc_score(y, X.teacher), 6), flush=True)
oof, pt, scores, its = np.zeros(len(y)), np.zeros(len(XT)), [], []
P = dict(n_estimators=10000, learning_rate=0.03, num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=3, verbose=-1, random_state=42)
for i, (a, b) in enumerate(folds(y)):
    Xa, Xb, Xt = add_te(X.iloc[a], X.iloc[b], XT, y[a])
    m = lgb.LGBMClassifier(**P).fit(Xa, y[a], eval_set=[(Xb, y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[b] = m.predict_proba(Xb)[:, 1]; pt += m.predict_proba(Xt)[:, 1] / 5
    scores.append(roc_auc_score(y[b], oof[b])); its.append(m.best_iteration_); print(f"fold {i} {scores[-1]:.6f} it {its[-1]}", flush=True)
auc = roc_auc_score(y, oof); print(f"OOF {auc:.6f}  ({time.time()-t0:.0f}s)")
np.save(f"oof/{name}.npy", oof); np.save(f"oof/{name}_test.npy", pt)
import pandas as pd; pd.DataFrame({"id": te.id, "satisfaction": pt}).to_csv(f"subs/{name}.csv", index=False)
imp = pd.Series(m.feature_importances_, X.columns.tolist() + [c for c in Xa.columns if c not in X.columns]).sort_values(ascending=False)
json.dump({"name": name, "oof": auc, "folds": scores, "iters": its, "params": P, "top_imp": imp.head(15).to_dict()}, open(f"logs/{name}.json", "w"), indent=1, default=int)
