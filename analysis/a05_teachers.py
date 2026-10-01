"""Which teacher (model trained only on the original) best ranks OUR labels? Current: LGB 600x0.03, AUC 0.9544."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "models"); from features import load, CAT
tr, te, og, y, og_y = load(); raw = [c for c in tr.columns if c != "id"]
A, O = tr[raw], og[raw]
out = {}
def rep(n, p): out[n] = p; print(f"{n:28s} {roc_auc_score(y, p):.6f}", flush=True)
rep("lgb_600 (current)", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0).fit(O, og_y).predict_proba(A)[:, 1])
rep("lgb_1500_cs0.5", lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, random_state=0).fit(O, og_y).predict_proba(A)[:, 1])
rep("lgb_300_leaves15", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=0).fit(O, og_y).predict_proba(A)[:, 1])
rep("xgb_d8", xgb.XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.8, enable_categorical=True, tree_method="hist", n_jobs=8).fit(O, og_y).predict_proba(A)[:, 1])
Os, As = O.copy(), A.copy()
for c in CAT: Os[c], As[c] = Os[c].astype(str), As[c].astype(str)
rep("cat_d8", CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=8, verbose=0, thread_count=8, random_seed=0).fit(Os, og_y, cat_features=CAT).predict_proba(As)[:, 1])
from scipy.special import logit
L = np.c_[[logit(np.clip(v, 1e-6, 1 - 1e-6)) for v in out.values()]].T
rep("mean logit of all", L.mean(1))
np.save("data/teachers_train.npy", L)
