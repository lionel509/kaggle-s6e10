"""RealMLP teacher: trained on the original 130k rows only (3 seeds x 8 epochs, goodpjw2008's og_realmlp), scored on train+test as a logit.
Own process because torch and lightgbm must not share one. Writes data/feat_teacher_mlp.parquet."""
import sys, time, numpy as np, pandas as pd
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "models"); from features import load, CAT, FD, AGE, DD, AD

t0 = time.time(); tr, te, og, y, og_y = load(); raw = [c for c in tr.columns if c != "id"]
al = pd.concat([tr[raw], te[raw], og[raw]], ignore_index=True)
al[AD] = al[AD].fillna(0.0)
twins = [FD, AGE, DD, AD] + [c for c in raw if al[c].dtype.kind == "i" and al[c].max() <= 5]
for c in twins: al[c + "_cat_"] = al[c].astype(int).astype(str)
cats = CAT + [c + "_cat_" for c in twins]
for c in cats: al[c] = al[c].astype(str).astype("category")
n = len(tr) + len(te); A, O = al.iloc[:n], al.iloc[n:]
P = dict(n_ens=8, n_epochs=8, batch_size=256, use_early_stopping=False, lr=0.053, wd=0.0236, sq_mom=0.988, lr_sched="lin_cos_log_15",
         wd_sched="cos_log_15", first_layer_lr_factor=0.25, embedding_size=5, max_one_hot_cat_size=18, hidden_sizes=[512, 256, 128], act="silu",
         p_drop=0.05, p_drop_sched="expm4t", plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151, plr_sigma=2.33,
         ls_eps=0.01, ls_eps_sched="sqrt_cos", add_front_scale=False, bias_init_mode="neg-uniform-dynamic-2",
         tfms=["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"], device="mps", verbosity=0, val_fraction=0.05)
p = np.zeros(n)
for s in range(3):
    p += RealMLP_TD_Classifier(**P, random_state=s).fit(O, og_y, cat_col_names=cats).predict_proba(A)[:, 1] / 3
    print(f"seed {s} done ({time.time()-t0:.0f}s)", flush=True)
q = np.clip(p, 1e-6, 1 - 1e-6)
pd.DataFrame({"t_mlp": np.log(q / (1 - q)).astype("float32")}).to_parquet("data/feat_teacher_mlp.parquet")
print(f"t_mlp vs our labels AUC {roc_auc_score(y, p[:len(tr)]):.6f}  (lgb teacher 0.954386)")
