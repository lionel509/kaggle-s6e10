"""Step 6: group a model's OOF residuals by every key - anything still structured is signal the model misses.
Usage: python a04_residuals.py oof/v1_lgb.npy"""
import sys, numpy as np, pandas as pd
from scipy.special import expit
tr = pd.read_csv("data/train.csv"); y = tr.satisfaction.astype(str).eq("True").astype(int).values
p = np.load(sys.argv[1]); p = expit(p) if (p.min() < 0 or p.max() > 1) else p
r = y - p; base = (r ** 2).mean()
keys = {c: tr[c] for c in tr.columns if c not in ("id", "satisfaction")}
FD = tr["Flight Distance"]
keys |= {"fd_x_class": FD.astype(str) + tr.Class, "fd_x_age": FD.astype(str) + "_" + tr.Age.astype(str), "age_x_class_tt": tr.Age.astype(str) + tr.Class + tr["Type of Travel"],
         "dep_eq_arr": (tr["Departure Delay in Minutes"] == tr["Arrival Delay in Minutes"]), "arr_nan": tr["Arrival Delay in Minutes"].isna(),
         "n_zero_ratings": (tr[[c for c in tr.columns if tr[c].dtype.kind == "i" and tr[c].max() == 5]] == 0).sum(1),
         "ratings_sum": tr[[c for c in tr.columns if tr[c].dtype.kind == "i" and tr[c].max() == 5]].sum(1)}
rows = []
for k, s in keys.items():
    g = pd.DataFrame({"r": r, "k": s.values}).groupby("k").r.agg(["mean", "count"]); g = g[g["count"] >= 30]
    noise = (base / g["count"]).mean()  # expected var of a group mean under no structure
    rows.append((k, len(g), g["mean"].var(), noise, g["mean"].abs().max()))
d = pd.DataFrame(rows, columns=["key", "groups", "var_mean_res", "noise_var", "max_abs"]).assign(ratio=lambda d: d.var_mean_res / d.noise_var).sort_values("ratio", ascending=False)
print(d.round(6).to_string(index=False)); print("\nratio ~1 = no structure left; >>1 = the model misses signal on that key")
