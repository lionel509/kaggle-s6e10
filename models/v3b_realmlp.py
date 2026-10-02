"""v3b: RealMLP with goodpjw2008's input recipe on our frame: teacher as a logit, categorical twins of every rating too, their tuned params; + our extra features.
Usage: python v3b_realmlp.py [epochs=6] [n_ens=8] [seed=42]"""
import json, os, sys, time, numpy as np, pandas as pd
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "models"); from features import load, folds, label_free, add_te, extra, CAT, FD, AGE, DD, AD

EP, NE, SEED = (int(a) for a in (sys.argv[1:] + ["6", "8", "42"][len(sys.argv) - 1:])[:3])
name = f"v3b_realmlp_e{EP}_n{NE}_s{SEED}"; K = int(os.environ.get("K", 5)); name += f"_k{K}" if K != 5 else ""; t0 = time.time()
tr, te, og, y, og_y = load(); X, XT = label_free(tr, te, og, og_y); E, ET = extra(tr, te)
X, XT = pd.concat([X, E], axis=1), pd.concat([XT, ET], axis=1)
if os.environ.get("TEACH"):
    from features import teachers; E, ET = teachers(tr, te, og, og_y); X, XT = pd.concat([X, E], axis=1), pd.concat([XT, ET], axis=1); name += "_teach" + ("2" if os.environ["TEACH"] == "2" else "")
twins = [FD, AGE, DD, AD] + [c for c in tr.columns if tr[c].dtype.kind == "i" and tr[c].max() <= 5]
for D, src in ((X, tr), (XT, te)):
    D["teacher"] = np.log(D.teacher.clip(1e-6, 1 - 1e-6) / (1 - D.teacher.clip(1e-6, 1 - 1e-6)))
    D[AD] = D[AD].fillna(0.0)
    for c in twins: D[c + "_cat_"] = src[c].fillna(0).astype(int).astype(str).values
cats = CAT + [c + "_cat_" for c in twins]
for c in cats:
    u = pd.api.types.union_categoricals([X[c].astype("category"), XT[c].astype("category")]).categories
    X[c] = pd.Categorical(X[c].astype(str), categories=u.astype(str)); XT[c] = pd.Categorical(XT[c].astype(str), categories=u.astype(str))
num = [c for c in X.columns if c not in cats]; X[num] = X[num].fillna(-1); XT[num] = XT[num].fillna(-1)
P = dict(n_ens=NE, n_epochs=EP, batch_size=256, use_early_stopping=False, lr=0.053, wd=0.0236, sq_mom=0.988, lr_sched="lin_cos_log_15",
         wd_sched="cos_log_15", first_layer_lr_factor=0.25, embedding_size=5, max_one_hot_cat_size=18, hidden_sizes=[512, 256, 128], act="silu",
         p_drop=0.05, p_drop_sched="expm4t", plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151, plr_sigma=2.33,
         ls_eps=0.01, ls_eps_sched="sqrt_cos", add_front_scale=False, bias_init_mode="neg-uniform-dynamic-2",
         tfms=["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
         device="mps", random_state=SEED, verbosity=0, val_metric_name="1-auc_ovr")
oof, pt, scores = np.zeros(len(y)), np.zeros(len(XT)), []
for i, (a, b) in enumerate(folds(y, K)):
    Xa, Xb, Xt = add_te(X.iloc[a], X.iloc[b], XT, y[a])
    m = RealMLP_TD_Classifier(**P).fit(Xa, y[a], X_val=Xb, y_val=y[b], cat_col_names=cats)
    oof[b] = m.predict_proba(Xb)[:, 1]; pt += m.predict_proba(Xt)[:, 1] / K
    scores.append(roc_auc_score(y[b], oof[b])); print(f"fold {i} {scores[-1]:.6f} ({time.time()-t0:.0f}s)", flush=True)
auc = roc_auc_score(y, oof); print(f"OOF {auc:.6f}  ({time.time()-t0:.0f}s)")
np.save(f"oof/{name}.npy", oof); np.save(f"oof/{name}_test.npy", pt)
json.dump({"name": name, "oof": auc, "folds": scores, "params": P}, open(f"logs/{name}.json", "w"), indent=1, default=str)
