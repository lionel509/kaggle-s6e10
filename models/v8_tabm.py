"""v8 (F): TabM (pytabkit, arch_type tabm) on the v3b aux+teach frame. The 4 high-cardinality twins stay numeric (TabM one-hot-encodes every categorical).
Usage: [K=20] [SMOKE=N] [DEVICE=cpu|mps] python v8_tabm.py [seed=42]"""
import json, os, sys, time, numpy as np, pandas as pd
from pytabkit import TabM_D_Classifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "models"); from features import load, folds, label_free, add_te, extra, aux, teachers, CAT, AD

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
K = int(os.environ.get("K", 5)); name = f"v8_tabm_s{SEED}" + (f"_k{K}" if K != 5 else "") + "_aux_teach"; t0 = time.time()
SMOKE = int(os.environ.get("SMOKE") or 0); R = "smoke/" if SMOKE else ""
tr, te, og, y, og_y = load(); X, XT = label_free(tr, te, og, og_y); E, ET = extra(tr, te)
X, XT = pd.concat([X, E], axis=1), pd.concat([XT, ET], axis=1)
E, ET = aux(tr, te); X, XT = pd.concat([X, E], axis=1), pd.concat([XT, ET], axis=1)
os.environ["TEACH"] = "1"  # the name is pinned to *_aux_teach: an exported TEACH must not change this frame
E, ET = teachers(tr, te, og, og_y); X, XT = pd.concat([X, E], axis=1), pd.concat([XT, ET], axis=1)
for d in ("oof/ckpt", "logs", "subs/hedge", "data/soft"): os.makedirs(R + d, exist_ok=True)
twins = [c for c in tr.columns if tr[c].dtype.kind == "i" and tr[c].max() <= 5]
for D, src in ((X, tr), (XT, te)):
    D["teacher"] = np.log(D.teacher.clip(1e-6, 1 - 1e-6) / (1 - D.teacher.clip(1e-6, 1 - 1e-6)))
    D[AD] = D[AD].fillna(0.0)
    for c in twins: D[c + "_cat_"] = src[c].fillna(0).astype(int).astype(str).values
if SMOKE: X = X.iloc[:SMOKE].reset_index(drop=True); XT = XT.iloc[:SMOKE // 5].reset_index(drop=True); y = y[:SMOKE]
cats = CAT + [c + "_cat_" for c in twins]
for c in cats:
    u = pd.api.types.union_categoricals([X[c].astype("category"), XT[c].astype("category")]).categories
    X[c] = pd.Categorical(X[c].astype(str), categories=u.astype(str)); XT[c] = pd.Categorical(XT[c].astype(str), categories=u.astype(str))
num = [c for c in X.columns if c not in cats]; X[num] = X[num].fillna(-1); XT[num] = XT[num].fillna(-1)
P = dict(device=os.environ.get("DEVICE", "mps"), random_state=SEED, verbosity=0, arch_type="tabm", tabm_k=32, num_emb_type="pwl", num_emb_n_bins=48,
         batch_size=1024, lr=2e-3, weight_decay=0.0, n_epochs=1 if SMOKE else 20, patience=3, d_embedding=16, d_block=512, n_blocks="auto",
         dropout=0.1, tfms=["quantile_tabr"], val_metric_name="1-auc_ovr")
oof, pt, scores = np.zeros(len(y)), np.zeros(len(XT)), []
for i, (a, b) in enumerate(folds(y, K)):
    Xa, Xb, Xt = add_te(X.iloc[a], X.iloc[b], XT, y[a])
    m = TabM_D_Classifier(**P).fit(Xa, y[a], X_val=Xb, y_val=y[b], cat_col_names=cats)
    oof[b] = m.predict_proba(Xb)[:, 1]; pt += m.predict_proba(Xt)[:, 1] / K
    scores.append(roc_auc_score(y[b], oof[b])); print(f"fold {i} {scores[-1]:.6f} ({time.time()-t0:.0f}s)", flush=True)
auc = roc_auc_score(y, oof); print(f"OOF {auc:.6f}  ({time.time()-t0:.0f}s)")
np.save(f"{R}oof/{name}.npy", oof); np.save(f"{R}oof/{name}_test.npy", pt)
json.dump({"name": name, "oof": auc, "folds": scores, "params": P}, open(f"{R}logs/{name}.json", "w"), indent=1, default=str)
