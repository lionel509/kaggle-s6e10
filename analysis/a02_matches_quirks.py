"""Step 5 (row matches vs original), step 1 (per-value quirk on high-card numerics), step 2 (which groupings share a quirk)."""
import pandas as pd, numpy as np, tiktoken
tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
og = pd.concat([pd.read_csv("data/orig/train.csv"), pd.read_csv("data/orig/test.csv")])
og["y"] = (og.satisfaction == "satisfied").astype(int); tr["y"] = tr.satisfaction.astype(str).eq("True").astype(int)
feats = [c for c in tr.columns if c not in ("id", "satisfaction", "y")]
print("zero-delay share: dep train %.3f orig %.3f | arr train %.3f orig %.3f" % ((tr["Departure Delay in Minutes"]==0).mean(), (og["Departure Delay in Minutes"]==0).mean(), (tr["Arrival Delay in Minutes"]==0).mean(), (og["Arrival Delay in Minutes"]==0).mean()))
# whole-row and partial-row matches
key = lambda d, cols: d[cols].fillna(-1).astype(str).agg("|".join, axis=1)
for name, cols in [("all", feats), ("no-delays", [c for c in feats if "Delay" not in c]), ("ratings-only", [c for c in feats if tr[c].nunique() <= 6 and tr[c].dtype.kind == "i"])]:
    ko = key(og, cols); kt = key(tr, cols); ke = key(te, cols)
    m = kt.isin(set(ko))
    print(f"\nmatch[{name}] ({len(cols)} cols): train {m.mean():.4f}  test {ke.isin(set(ko)).mean():.4f}  train-dup-within {kt.duplicated(keep=False).mean():.4f}")
    if m.any():
        orate = og.groupby(ko).y.mean()
        r = kt[m].map(orate)
        print("  on matched train rows: corr(orig label, synth label) %.4f ; agree %.4f ; synth rate %.4f" % (np.corrcoef(r, tr.y[m])[0, 1], ((r > .5) == tr.y[m]).mean(), tr.y[m].mean()))
# per-value quirk on Flight Distance: real share of variance across values
def quirk(col, minn=30):
    g = tr.groupby(col).y.agg(["mean", "count"]); g = g[g["count"] >= minn]
    p = tr.y.mean(); noise = (p*(1-p)/g["count"]).mean()
    # residual quirk after a smooth trend: use rolling median over sorted values
    sm = g["mean"].rolling(25, center=True, min_periods=5).median()
    res = g["mean"] - sm
    print(f"\n{col}: {len(g)} values n>={minn}; var(rate)={g['mean'].var():.5f} var(res vs smooth)={res.var():.5f} expected sampling var={noise:.5f} -> real share of residual {(1-noise/res.var()):.2f}")
    return g.assign(res=res).dropna()
g = quirk("Flight Distance")
# step 2: LOO correlation of per-value residual with its group under various groupings
enc = tiktoken.get_encoding("gpt2")
def loo(groups, name):
    df = g.assign(k=groups.reindex(g.index).values)
    s = df.groupby("k").res.transform("sum"); c = df.groupby("k").res.transform("count")
    ok = c > 1; lo = (s - df.res)[ok] / (c - 1)[ok]
    print(f"  {name:22s} groups={df.k.nunique():5d}  loo-corr={np.corrcoef(lo, df.res[ok])[0,1]:.3f}")
v = pd.Series(g.index, index=g.index); st = v.astype(str)
print("LOO quirk sharing, Flight Distance:")
for n in (1, 2, 3): loo(st.str[:n] + "_" + st.str.len().astype(str), f"first{n}+len")
for n in (1, 2, 3): loo(st.str[-n:], f"last{n}")
loo(v % 7, "mod7"); loo(v // 100, "div100"); loo(v // 10, "div10")
tok = st.map(lambda s: enc.encode(" " + s))
loo(tok.map(lambda t: t[0]), "gpt2 first tok"); loo(tok.map(lambda t: t[-1]), "gpt2 last tok"); loo(tok.map(len), "gpt2 n tok")
cl = tiktoken.get_encoding("cl100k_base"); loo(st.map(lambda s: cl.encode(s)[0]), "cl100k first tok")
