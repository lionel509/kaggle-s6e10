"""v5: XGBoost (hist, native categoricals) on v1's frame + nested TE.  Usage: python v5_xgb.py [seed=42] [depth=8]"""
import json, os, sys, time, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "models"); from features import load, folds, label_free, add_te

SEED, DEPTH = (int(a) for a in (sys.argv[1:] + ["42", "8"][len(sys.argv) - 1:])[:2])
K = int(os.environ.get("K", 5)); name = f"v5_xgb_d{DEPTH}_s{SEED}" + (f"_k{K}" if K != 5 else ""); t0 = time.time()
tr, te, og, y, og_y = load(); X, XT = label_free(tr, te, og, og_y)
if os.environ.get("EXTRA"):
    from features import extra; E, ET = extra(tr, te); X, XT = pd.concat([X, E], axis=1), pd.concat([XT, ET], axis=1); name += "_extra"
if os.environ.get("HAM"):
    from features import ham; E, ET = ham(tr, te); X, XT = pd.concat([X, E], axis=1), pd.concat([XT, ET], axis=1); name += "_ham"
if os.environ.get("AUX"):
    from features import aux; E, ET = aux(tr, te); X, XT = pd.concat([X, E], axis=1), pd.concat([XT, ET], axis=1); name += "_aux"
if os.environ.get("TEACH"):
    from features import teachers; E, ET = teachers(tr, te, og, og_y); X, XT = pd.concat([X, E], axis=1), pd.concat([XT, ET], axis=1); name += "_teach" + ("" if os.environ["TEACH"] == "1" else os.environ["TEACH"])
if os.environ.get("BLKTE"): name += "_blkte"
SOFT = bool(os.environ.get("SOFT")); name += "_soft" if SOFT else ""
SMOKE = int(os.environ.get("SMOKE", 0)); R = "smoke/" if SMOKE else ""  # smoke runs write only under smoke/
if SMOKE: X, XT, y = X.iloc[:SMOKE].reset_index(drop=True), XT.iloc[:SMOKE // 5].reset_index(drop=True), y[:SMOKE]
for d in ("oof/ckpt", "logs", "subs/hedge", "data/soft"): os.makedirs(R + d, exist_ok=True)
P = dict(n_estimators=50 if SMOKE else 20000, learning_rate=0.02, max_depth=DEPTH, min_child_weight=20, subsample=0.8, colsample_bytree=0.5, reg_lambda=5,
         tree_method="hist", enable_categorical=True, max_cat_to_onehot=4, eval_metric="auc", early_stopping_rounds=300, random_state=SEED, n_jobs=int(os.environ.get("THREADS", 8)))
if os.environ.get("LEAVE"):  # B teacher: fit on the k5 folds outside S, predict S. No eval_set, so S's labels never steer it.
    assert K == 5 and not SOFT and not os.environ.get("BLKTE") and os.environ.get("TEACH") == "1" and os.environ.get("AUX") and os.environ.get("EXTRA")
    from soft import rows_for, teacher_path
    S = os.environ["LEAVE"]; a, b = rows_for(y, S)
    Xa, Xb, _ = add_te(X.iloc[a], X.iloc[b], XT.iloc[:5], y[a])
    m = xgb.XGBClassifier(**(P | dict(n_estimators=50 if SMOKE else 888, early_stopping_rounds=None))).fit(Xa, y[a])
    np.save(teacher_path("xgb", S, R + "data/soft"), m.predict_proba(Xb)[:, 1].astype("float32"))
    print(f"LEAVE {S} rows {len(b)}", flush=True); sys.exit(0)
if os.environ.get("REFIT"):  # E: refit on all rows at the mean best iteration; hedge = 50/50 with the fold-average test
    assert not SOFT
    J = json.load(open(f"{R}logs/{name}.json")); nit = int(round(np.mean(J["iters"]))); Xa, _, Xt = add_te(X, X.iloc[:5], XT, y)
    m = xgb.XGBClassifier(**(P | dict(n_estimators=nit, early_stopping_rounds=None))).fit(Xa, y)
    p = m.predict_proba(Xt)[:, 1]
    np.save(f"{R}subs/hedge/{name}_refit_test.npy", p); np.save(f"{R}subs/hedge/{name}_hedge_test.npy", 0.5 * p + 0.5 * np.load(f"{R}oof/{name}_test.npy"))
    print(f"REFIT {name} nit {nit}", flush=True); sys.exit(0)
if SOFT:  # B student: soft targets from leave-two-folds-out teachers; early stopping stays on the hard labels of b, as in the baseline
    from soft import nested_folds, soft_target
    splits = nested_folds(y, K)
else: splits = [(a, b, None) for a, b in folds(y, K)]
oof, pt, scores, its = np.zeros(len(y)), np.zeros(len(XT)), [], []
for i, (a, b, j) in enumerate(splits):
    Xa, Xb, Xt = add_te(X.iloc[a], X.iloc[b], XT, y[a])
    if SOFT:
        t = soft_target(y, a, j, d=R + "data/soft")
        m = xgb.XGBRegressor(**(P | {"objective": "binary:logistic"})).fit(Xa, t, eval_set=[(Xb, y[b])], verbose=False)
        oof[b] = m.predict(Xb); pt += m.predict(Xt) / K
    else:
        m = xgb.XGBClassifier(**P).fit(Xa, y[a], eval_set=[(Xb, y[b])], verbose=False)
        oof[b] = m.predict_proba(Xb)[:, 1]; pt += m.predict_proba(Xt)[:, 1] / K
    scores.append(roc_auc_score(y[b], oof[b])); its.append(m.best_iteration); print(f"fold {i} {scores[-1]:.6f} it {its[-1]} ({time.time()-t0:.0f}s)", flush=True)
auc = roc_auc_score(y, oof); print(f"OOF {auc:.6f}  ({time.time()-t0:.0f}s)")
np.save(f"{R}oof/{name}.npy", oof); np.save(f"{R}oof/{name}_test.npy", pt)
json.dump({"name": name, "oof": auc, "folds": scores, "iters": its, "params": P, **({"soft": True} if SOFT else {})}, open(f"{R}logs/{name}.json", "w"), indent=1, default=int)
