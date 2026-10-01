"""Shared feature frame. Label-free features are built on train+test; label-based ones are fold-safe."""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

CAT = ["Gender", "Customer Type", "Type of Travel", "Class"]
FD, AGE, DD, AD = "Flight Distance", "Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"


def load():
    tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
    og = pd.concat([pd.read_csv("data/orig/train.csv"), pd.read_csv("data/orig/test.csv")]).drop(columns=["Unnamed: 0", "id", "Inflight service"])
    y = tr.pop("satisfaction").astype(str).eq("True").astype(int).values
    og_y = (og.pop("satisfaction") == "satisfied").astype(int).values
    for d in (tr, te, og):
        for c in CAT: d[c] = d[c].astype("category")
    return tr, te, og, y, og_y


def folds(y, k=5, seed=42):
    """The one CV split for the whole month."""
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def label_free(tr, te, og, og_y, cache="data/feat_lf.parquet"):
    """Cached: neural-net scripts must not import lightgbm (two OpenMP runtimes segfault next to torch on macOS)."""
    import os
    if os.path.exists(cache):
        out = pd.read_parquet(cache)
        for c in CAT: out[c] = out[c].astype("category")
        return out.iloc[: len(tr)].reset_index(drop=True), out.iloc[len(tr):].reset_index(drop=True)
    a, b = _label_free(tr, te, og, og_y); pd.concat([a, b], ignore_index=True).to_parquet(cache)
    return a, b


def _label_free(tr, te, og, og_y):
    raw = [c for c in tr.columns if c != "id"]
    al = pd.concat([tr, te], ignore_index=True)
    out = al[raw].copy()
    for c in raw:
        if c in CAT: continue
        out[c + "_cnt"] = al[c].map(al[c].value_counts()).astype("float32")
        out[c + "_om"] = al[c].map(pd.Series(og_y).groupby(og[c].values).mean()).astype("float32")
    # route profile: per exact distance, mean of every other column (label-free)
    num = al[raw].copy()
    for c in CAT: num[c] = num[c].cat.codes
    prof = num.groupby(al[FD]).transform("mean").drop(columns=[FD]).add_prefix("rt_")
    out = pd.concat([out, prof.astype("float32")], axis=1)
    out["fd_mod10"], out["fd_mod100"], out["age_mod10"] = al[FD] % 10, al[FD] % 100, al[AGE] % 10
    # teacher: a model trained on the original only, scored on everything
    import lightgbm as lgb
    t = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    t.fit(og[raw], og_y)
    out["teacher"] = t.predict_proba(al[raw])[:, 1]
    return out.iloc[: len(tr)].reset_index(drop=True), out.iloc[len(tr):].reset_index(drop=True)


def te_keys(df):
    s = lambda c: df[c].astype(str)
    return {"te_fd": s(FD), "te_fd_cls": s(FD) + s("Class"), "te_fd_tt": s(FD) + s("Type of Travel"),
            "te_age_cls_tt_ct": s(AGE) + s("Class") + s("Type of Travel") + s("Customer Type"), "te_fd10": (df[FD] // 10).astype(str)}


def add_te(Xtr, Xva, Xte, ytr, m=20, inner=5, seed=0):
    """Nested target encoding: inner-fold OOF on the fit rows, full-fit-rows map for val/test."""
    p = ytr.mean(); ktr, kva, kte = te_keys(Xtr), te_keys(Xva), te_keys(Xte)
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    enc = lambda k, yy: (lambda g: (g["sum"] + m * p) / (g["count"] + m))(pd.Series(yy).groupby(k.values).agg(["sum", "count"]))
    for name in ktr:
        o = np.full(len(Xtr), p)
        for a, b in StratifiedKFold(inner, shuffle=True, random_state=seed).split(Xtr, ytr):
            o[b] = ktr[name].iloc[b].map(enc(ktr[name].iloc[a], ytr[a])).fillna(p).values
        e = enc(ktr[name], ytr)
        Xtr[name] = o; Xva[name] = kva[name].map(e).fillna(p).values; Xte[name] = kte[name].map(e).fillna(p).values
    return Xtr, Xva, Xte
