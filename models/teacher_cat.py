"""5th teacher (#3 C): CatBoost on the original only with v4's CROSS + BLOCK native cats, scored on train+test as a logit; read by teachers() under TEACH=3. catboost only, no torch.
Writes data/feat_teacher_cat2.parquet (smoke/feat_teacher_cat2.parquet when SMOKE is set)."""
import os, sys, time, numpy as np, pandas as pd
from functools import reduce
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "models"); from features import load, CAT, RATINGS, BLOCKS

t0 = time.time(); SMOKE = int(os.environ.get("SMOKE") or 0)
tr, te, og, y, og_y = load(); raw = [c for c in tr.columns if c != "id"]
if SMOKE: og, og_y = og.iloc[:SMOKE], og_y[:SMOKE]
CROSS_WITH = ["Class", "Type of Travel", "Customer Type"]


def frame(d):
    out = d[raw].copy()
    for c in CAT: out[c] = out[c].astype(str).values
    for r in RATINGS:
        for g in CROSS_WITH:
            out[f"{r}__x__{g}"] = (d[r].astype(str) + "|" + d[g].astype(str)).values
    for n, cols in BLOCKS.items():
        out[n] = reduce(lambda a, b: a + b, (d[c].astype(str) for c in cols)).values
    return out


cats = CAT + [f"{r}__x__{g}" for r in RATINGS for g in CROSS_WITH] + list(BLOCKS)
m = CatBoostClassifier(iterations=50 if SMOKE else 2000, learning_rate=0.05, depth=8, verbose=0,
                       thread_count=int(os.environ.get("THREADS") or 8), random_seed=0, allow_writing_files=False)
m.fit(frame(og), og_y, cat_features=cats)
p = m.predict_proba(frame(pd.concat([tr, te], ignore_index=True)))[:, 1]
q = np.clip(p, 1e-6, 1 - 1e-6)
out = pd.DataFrame({"t_catx": np.log(q / (1 - q)).astype("float32")})
path = "smoke/feat_teacher_cat2.parquet" if SMOKE else "data/feat_teacher_cat2.parquet"
os.makedirs(os.path.dirname(path), exist_ok=True); out.to_parquet(path)
print(f"t_catx vs our labels AUC {roc_auc_score(y, p[:len(tr)]):.6f}  ({time.time()-t0:.0f}s, rows {len(out)})", flush=True)
