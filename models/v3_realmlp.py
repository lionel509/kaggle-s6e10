"""v3: RealMLP (pytabkit) on our frame - v1's features + nested TE, plus categorical twins of the raw numerics (yekenot/goodpjw recipe).
Usage: python v3_realmlp.py [epochs=6] [n_ens=4] [seed=42]    SMOKE=1: 40k rows, 1 fold"""
import json, os, sys, time, numpy as np, pandas as pd, torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "models"); from features import load, folds, label_free, add_te, FD, AGE, DD, AD

EP, NE, SEED = (int(a) for a in (sys.argv[1:] + ["6", "4", "42"][len(sys.argv) - 1:])[:3])
SMOKE = os.environ.get("SMOKE"); name = f"v3_realmlp_e{EP}_n{NE}_s{SEED}"; t0 = time.time()
tr, te, og, y, og_y = load(); X, XT = label_free(tr, te, og, og_y)
for D, src in ((X, tr), (XT, te)):
    for c in (FD, AGE, DD, AD): D[c + "_cat"] = src[c].fillna(-1).astype(int).astype(str).astype("category").values
    D[AD] = D[AD].fillna(D[DD])
cats = [c for c in X.columns if str(X[c].dtype) == "category"]
num = [c for c in X.columns if c not in cats]; X[num] = X[num].fillna(-1); XT[num] = XT[num].fillna(-1)
for c in cats:  # shared category sets so train/test codes agree
    u = pd.api.types.union_categoricals([X[c], XT[c]]).categories; X[c] = X[c].cat.set_categories(u); XT[c] = XT[c].cat.set_categories(u)
P = dict(random_state=SEED, verbosity=0, val_metric_name="1-auc_ovr", n_ens=NE, n_epochs=EP, batch_size=256, use_early_stopping=False,
         lr=0.053, wd=0.015, sq_mom=0.988, lr_sched="flat_anneal", wd_sched="cos_log_15", first_layer_lr_factor=0.25, embedding_size=5,
         max_one_hot_cat_size=18, hidden_sizes=[512, 256, 128], act="silu", p_drop=0.05, p_drop_sched="invsqrtp1e-3",
         plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151, plr_sigma=2.33, device="mps")
F = folds(y)
if SMOKE: F = [(a[a < 40000], b[b < 40000][:8000]) for a, b in F[:1]]
oof, pt, scores = np.zeros(len(y)), np.zeros(len(XT)), []
for i, (a, b) in enumerate(F):
    Xa, Xb, Xt = add_te(X.iloc[a], X.iloc[b], XT, y[a])
    m = RealMLP_TD_Classifier(**P).fit(Xa, y[a], Xb, y[b])
    oof[b] = m.predict_proba(Xb)[:, 1]; pt += m.predict_proba(Xt)[:, 1] / len(F)
    scores.append(roc_auc_score(y[b], oof[b])); print(f"fold {i} {scores[-1]:.6f} ({time.time()-t0:.0f}s)", flush=True)
if SMOKE: sys.exit()
auc = roc_auc_score(y, oof); print(f"OOF {auc:.6f}  ({time.time()-t0:.0f}s)")
np.save(f"oof/{name}.npy", oof); np.save(f"oof/{name}_test.npy", pt)
json.dump({"name": name, "oof": auc, "folds": scores, "params": P}, open(f"logs/{name}.json", "w"), indent=1, default=str)
