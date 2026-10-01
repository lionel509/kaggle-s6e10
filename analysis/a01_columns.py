"""Day-one pass, step 1 + 5: every column alone, and train vs test vs original."""
import pandas as pd, numpy as np
tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
og = pd.concat([pd.read_csv("data/orig/train.csv"), pd.read_csv("data/orig/test.csv")])
print("shapes", tr.shape, te.shape, og.shape)
print("orig cols", list(og.columns))
print("target", tr.satisfaction.value_counts(normalize=True).round(4).to_dict(), "| orig", og.satisfaction.value_counts(normalize=True).round(4).to_dict())
print("NaN train", tr.isna().sum()[lambda s: s > 0].to_dict(), "test", te.isna().sum()[lambda s: s > 0].to_dict(), "orig", og.isna().sum()[lambda s: s > 0].to_dict())
y = tr.satisfaction.astype(str).eq("True").astype(int)
for c in tr.columns.drop(["id", "satisfaction"]):
    s = tr[c]; u = s.nunique()
    tv, ov = set(te[c].dropna().unique()), set(og[c].dropna().unique())
    trv = set(s.dropna().unique())
    print(f"\n## {c}: dtype={s.dtype} nuniq={u} min={s.min()} max={s.max()} | test-unseen={len(tv-trv)} | train-vals-not-in-orig={len(trv-ov)} ({tr[c].isin(ov).mean():.4f} rows in orig)")
    if u <= 25:
        g = pd.DataFrame({"n": s.value_counts(), "rate": y.groupby(s).mean()}).sort_index()
        og_r = og.groupby(c).satisfaction.apply(lambda v: (v == "satisfied").mean())
        g["orig_rate"] = og_r; g["orig_n"] = og[c].value_counts()
        print(g.round(4).to_string())
    else:
        vc = s.value_counts()
        print("top values", vc.head(8).to_dict())
        # non-integers?
        if s.dtype.kind == "f": print("non-integer share", (s.dropna() % 1 != 0).mean().round(4))
