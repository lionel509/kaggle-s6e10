"""Step 4: every pair of low-card columns - how much does the joint cell rate beat the additive (logit) model? Plus delay & Inflight-service checks."""
import pandas as pd, numpy as np, itertools
from sklearn.linear_model import LogisticRegression
tr = pd.read_csv("data/train.csv"); y = tr.satisfaction.astype(str).eq("True").astype(int)
og = pd.concat([pd.read_csv("data/orig/train.csv"), pd.read_csv("data/orig/test.csv")])
low = [c for c in tr.columns if c not in ("id", "satisfaction") and tr[c].nunique() <= 8]
lg = lambda p: np.log(p.clip(1e-4, 1-1e-4) / (1 - p.clip(1e-4, 1-1e-4)))
base = lg(y.mean()); res = []
for a, b in itertools.combinations(low, 2):
    ra = lg(y.groupby(tr[a]).transform("mean")) - base; rb = lg(y.groupby(tr[b]).transform("mean")) - base
    add = 1 / (1 + np.exp(-(base + ra + rb)))
    g = pd.DataFrame({"y": y, "add": add, "k": tr[a].astype(str) + "|" + tr[b].astype(str)}).groupby("k").agg(n=("y", "size"), r=("y", "mean"), a=("add", "mean"))
    g = g[g.n >= 200]
    res.append((a, b, float((g.n * (g.r - g.a) ** 2).sum() / g.n.sum()) ** .5))
print("Top interaction pairs (weighted RMS of cell rate minus additive-logit prediction):")
for a, b, s in sorted(res, key=lambda t: -t[2])[:15]: print(f"  {s:.4f}  {a} x {b}")
# delays: target rate by delay bucket, train vs orig
for c in ("Departure Delay in Minutes", "Arrival Delay in Minutes"):
    bk = lambda s: pd.cut(s, [-1, 0, 5, 15, 30, 60, 120, 2000])
    print(f"\n{c}"); print(pd.DataFrame({"train_rate": y.groupby(bk(tr[c])).mean(), "train_n": bk(tr[c]).value_counts(), "orig_rate": (og.satisfaction=="satisfied").groupby(bk(og[c])).mean()}).round(4).to_string())
d = tr["Arrival Delay in Minutes"] - tr["Departure Delay in Minutes"]
print("\narr-dep diff: corr train %.3f orig %.3f; |diff|<=5 share train %.3f orig %.3f" % (tr["Arrival Delay in Minutes"].corr(tr["Departure Delay in Minutes"]), og["Arrival Delay in Minutes"].corr(og["Departure Delay in Minutes"]), (d.abs()<=5).mean(), ((og["Arrival Delay in Minutes"]-og["Departure Delay in Minutes"]).abs()<=5).mean()))
# dropped column: how predictable is Inflight service from the other ratings in orig?
rat = [c for c in low if tr[c].dtype.kind == "i" and c not in ("Departure Delay in Minutes",)]
print("\norig corr of dropped 'Inflight service' with kept ratings:"); print(og[rat + ["Inflight service"]].corr()["Inflight service"].drop("Inflight service").sort_values(ascending=False).head(5).round(3).to_string())
