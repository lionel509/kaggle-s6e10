"""Shared feature pipeline for S6E9. Everything the v5 model uses, plus optional
pair / rounded target-encoding keys. All target encoding is nested: fit rows get
inner-OOF values, apply rows get values from the whole fit set."""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold

ORIG = "data/orig/EV_Adoption_and_Range_Anxiety_Dataset.csv"
SEED = 42

def load():
    tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
    o = pd.read_csv(ORIG).drop(columns="Buyer_ID")
    y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
    feats = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]
    return tr, te, o, y, feats

def folds(tr, y, n=5, seed=SEED):
    return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(tr, y))

class Base:
    """Deterministic, target-free features: categoricals, counts over train+test, original lookup."""
    def __init__(self, tr, te, o, feats):
        self.feats = feats; self.o = o
        allX = pd.concat([tr[feats], te[feats], o[feats]], ignore_index=True)
        self.cats = [c for c in feats if str(allX[c].dtype) in ("object", "str", "string")]
        self.cat_levels = {c: sorted(allX[c].dropna().unique()) for c in self.cats}
        synth = pd.concat([tr[feats], te[feats]], ignore_index=True)
        self.counts = {c: synth[c].value_counts() for c in ["Annual_Income_USD", "Daily_Commute_km"]}
        self.orig = {c: o.groupby(c)["Will_Buy_EV"].agg(cnt="size", rate=lambda s: (s == "Yes").mean())
                     for c in ["Annual_Income_USD", "Daily_Commute_km"]}
    def __call__(self, df):
        X = df[self.feats].copy()
        for c in self.cats: X[c] = pd.Categorical(X[c], categories=self.cat_levels[c])
        for c, vc in self.counts.items(): X[c + "_cnt"] = df[c].map(vc).astype(float)
        for c, nm in [("Annual_Income_USD", "inc"), ("Daily_Commute_km", "com")]:
            g = self.orig[c]
            X[nm + "_orig_cnt"] = df[c].map(g["cnt"]).fillna(0).astype(float)
            X[nm + "_orig_rate"] = df[c].map(g["rate"]).astype(float)
        return X

def te_keys(df, spec):
    """spec: a column name, a tuple of columns (pair key), or ('round', col, unit). Vectorised."""
    if isinstance(spec, str): return df[spec].astype(str)
    if spec[0] == "round": return (np.round(df[spec[1]] / spec[2]) * spec[2]).astype(str)
    k = df[spec[0]].astype(str)
    for c in spec[1:]: k = k + "|" + df[c].astype(str)
    return k

def te_name(spec):
    if isinstance(spec, str): return spec + "_te"
    if spec[0] == "round": return f"{spec[1]}_r{spec[2]}_te"
    return "x".join(s.split("_")[0] for s in spec) + "_te"

def _fit(keys, target, prior, m):
    g = pd.DataFrame({"k": keys.values, "y": target}).groupby("k")["y"].agg(["sum", "size"])
    return (g["sum"] + prior * m) / (g["size"] + m)

class TE:
    """Nested target encoding. Keys are built once per spec for train and test, then indexed per fold.
    fit rows get inner-OOF values, validation and test rows get values from the whole fit set."""
    def __init__(self, tr, te, specs, m=20, inner=5, seed=0):
        self.m, self.inner, self.seed = m, inner, seed
        self.ktr = {te_name(s): te_keys(tr, s) for s in specs}
        self.kte = {te_name(s): te_keys(te, s) for s in specs}
    def fold(self, a, b, y, Btr, Bte):
        ya = y[a]; prior = ya.mean(); Xa, Xb, Xt = Btr.iloc[a].copy(), Btr.iloc[b].copy(), Bte.copy()
        inner = list(StratifiedKFold(self.inner, shuffle=True, random_state=self.seed).split(a, ya))
        for nm in self.ktr:
            ka = self.ktr[nm].iloc[a]; col = np.zeros(len(a))
            for ia, ib in inner:
                col[ib] = ka.iloc[ib].map(_fit(ka.iloc[ia], ya[ia], prior, self.m)).fillna(prior).values
            Xa[nm] = col; full = _fit(ka, ya, prior, self.m)
            Xb[nm] = self.ktr[nm].iloc[b].map(full).fillna(prior).values
            Xt[nm] = self.kte[nm].map(full).fillna(prior).values
        return Xa, Xb, Xt

def run_cv(model_fn, tr, te, y, Btr, Bte, specs, skf, name, fit_kwargs_fn=None):
    """Generic 5-fold loop: model_fn() -> estimator with fit/predict_proba; saves OOF and a submission."""
    import time
    from sklearn.metrics import roc_auc_score
    tenc = TE(tr, te, specs) if specs else None
    oof = np.zeros(len(tr)); pred = np.zeros(len(te)); its = []; t = time.time()
    for a, b in skf:
        Xa, Xb, Xt = tenc.fold(a, b, y, Btr, Bte) if tenc else (Btr.iloc[a], Btr.iloc[b], Bte)
        m = model_fn(); m.fit(Xa, y[a], **(fit_kwargs_fn(Xb, y[b]) if fit_kwargs_fn else {}))
        oof[b] = m.predict_proba(Xb)[:, 1]; pred += m.predict_proba(Xt)[:, 1] / len(skf)
        its.append(getattr(m, "best_iteration_", None) or getattr(m, "best_iteration", None) or (m.get_best_iteration() if hasattr(m, "get_best_iteration") else -1))
    auc = roc_auc_score(y, oof)
    print(f"{name:12s} OOF {auc:.6f}  iters {its}  {time.time()-t:.0f}s", flush=True)
    np.save(f"submissions/oof_{name}.npy", oof)
    pd.DataFrame({"id": te.id, "Will_Buy_EV": pred}).to_csv(f"submissions/{name}.csv", index=False)
    return oof, pred, auc

V5_SPECS = ["Annual_Income_USD", "Daily_Commute_km", "Age"]
PAIR_SPECS = [("Annual_Income_USD", "Environmental_Concern_Level"), ("Annual_Income_USD", "Subsidy_Available"),
              ("Annual_Income_USD", "Home_Charging_Possible"), ("Daily_Commute_km", "City_Type"),
              ("Age", "Environmental_Concern_Level")]
ROUND_SPECS = [("round", "Annual_Income_USD", 100), ("round", "Annual_Income_USD", 1000)]

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=18, min_data_in_leaf=152, feature_fraction=0.463, bagging_fraction=0.957,
                  bagging_freq=1, lambda_l1=0.0034, lambda_l2=0.0022, min_gain_to_split=0.329, max_cat_threshold=37,
                  verbose=-1, n_estimators=4000)
