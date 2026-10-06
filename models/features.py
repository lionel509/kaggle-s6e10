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


def extra(tr, te, cache="data/feat_extra.parquet"):
    """Leads from our own analysis (a01-a04): original label rate on exact rating patterns, zero-rating count, the dropped Inflight service column imputed."""
    import os
    if os.path.exists(cache):
        out = pd.read_parquet(cache); return out.iloc[: len(tr)].reset_index(drop=True), out.iloc[len(tr):].reset_index(drop=True)
    og = pd.concat([pd.read_csv("data/orig/train.csv"), pd.read_csv("data/orig/test.csv")])
    oy = (og.satisfaction == "satisfied").astype(float)
    al = pd.concat([tr, te], ignore_index=True)
    rat = [c for c in al.columns if al[c].dtype.kind == "i" and c not in ("id", AGE, FD, DD)]
    groups = {"ratings": rat, "ratings_cat": rat + CAT, "nodelay": rat + CAT + [AGE, FD]}
    out = pd.DataFrame(index=al.index)
    for n, cols in groups.items():
        ko = og[cols].astype(str).agg("|".join, axis=1); ka = al[cols].astype(str).agg("|".join, axis=1)
        g = oy.groupby(ko.values).agg(["sum", "count"])
        out[f"om_{n}"] = ka.map((g["sum"] + 0.4345 * 2) / (g["count"] + 2)).astype("float32")  # smoothed; NaN = no match
        out[f"om_{n}_n"] = ka.map(g["count"]).fillna(0).astype("float32")
    out["n_zero_ratings"] = (al[rat] == 0).sum(axis=1)
    from sklearn.ensemble import HistGradientBoostingRegressor
    feats = rat + [AGE, FD]
    out["inflight_service_hat"] = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(og[feats], og["Inflight service"]).predict(al[feats]).astype("float32")
    out.to_parquet(cache)
    return out.iloc[: len(tr)].reset_index(drop=True), out.iloc[len(tr):].reset_index(drop=True)


def teachers(tr, te, og, og_y, cache="data/feat_teachers.parquet"):
    """Extra teachers (trained on the original only), as logits: a05 found lgb_1500 / xgb_d8 / cat_d8 rank our labels 0.9547-0.9549."""
    import os
    if os.path.exists(cache):
        out = pd.read_parquet(cache)
        if os.environ.get("TEACH") == "2":  # + the RealMLP teacher (models/teacher_mlp.py)
            out = pd.concat([out, pd.read_parquet("data/feat_teacher_mlp.parquet")], axis=1)
        return out.iloc[: len(tr)].reset_index(drop=True), out.iloc[len(tr):].reset_index(drop=True)
    import lightgbm as lgb, xgboost as xgb
    from catboost import CatBoostClassifier
    raw = [c for c in tr.columns if c != "id"]; al = pd.concat([tr, te], ignore_index=True)[raw]; O = og[raw]
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).astype("float32")
    out = pd.DataFrame(index=al.index)
    out["t_lgb1500"] = lg(lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, random_state=0).fit(O, og_y).predict_proba(al)[:, 1])
    out["t_xgb8"] = lg(xgb.XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.8, enable_categorical=True, tree_method="hist", n_jobs=8).fit(O, og_y).predict_proba(al)[:, 1])
    Os, As = O.copy(), al.copy()
    for c in CAT: Os[c], As[c] = Os[c].astype(str), As[c].astype(str)
    out["t_cat8"] = lg(CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=8, verbose=0, thread_count=8, random_seed=0).fit(Os, og_y, cat_features=CAT).predict_proba(As)[:, 1])
    out.to_parquet(cache)
    return out.iloc[: len(tr)].reset_index(drop=True), out.iloc[len(tr):].reset_index(drop=True)


def ham_core(Q, O, oy, og_grp=None, q_grp=None, k=1, chunk=2000):
    """Per query rating pattern: Hamming distance to the nearest original pattern, and the original satisfied-rate / count
    among originals at that distance, within <=1 and <=2. With groups (0..k-1), also the rate at <=1 among same-group originals.
    d<=2 by enumerating every original's 1- and 2-cell variants and looking them up among the query patterns; brute force only for d>=3."""
    from itertools import combinations
    og_grp = np.zeros(len(O), int) if og_grp is None else og_grp; q_grp = np.zeros(len(Q), int) if q_grp is None else q_grp
    V, m = int(max(O.max(), Q.max())) + 1, Q.shape[1]; pw = V ** np.arange(m, dtype=np.int64)
    oc, oinv = np.unique(O.astype(np.int64) @ pw, return_inverse=True); oinv = oinv.ravel()
    C = np.zeros((len(oc), k)); S = C.copy(); np.add.at(C, (oinv, og_grp), 1); np.add.at(S, (oinv, og_grp), oy)
    up, qinv = np.unique(Q.astype(np.int64) @ pw, return_inverse=True); qinv = qinv.ravel()
    D = np.array([(oc // pw[j]) % V for j in range(m)]).T  # unique original patterns, digits
    N = np.zeros((3, len(up), k)); R = N.copy()  # count / satisfied at exact distance 0,1,2, per group

    def hit(code, d):
        ix = np.searchsorted(up, code).clip(0, len(up) - 1); ok = up[ix] == code
        np.add.at(N[d], ix[ok], C[ok]); np.add.at(R[d], ix[ok], S[ok])
    hit(oc, 0)
    for j in range(m):
        for v in range(1, V): hit(oc + ((D[:, j] + v) % V - D[:, j]) * pw[j], 1)
    for a, b in combinations(range(m), 2):
        for u in range(1, V):
            for v in range(1, V): hit(oc + ((D[:, a] + u) % V - D[:, a]) * pw[a] + ((D[:, b] + v) % V - D[:, b]) * pw[b], 2)
    n, r = N.sum(2).T, R.sum(2).T; has = n > 0
    dmin = np.where(has.any(1), has.argmax(1), 3).astype(np.float32); nmin = n[np.arange(len(up)), dmin.clip(0, 2).astype(int)]
    rmin = r[np.arange(len(up)), dmin.clip(0, 2).astype(int)]
    far = np.where(dmin == 3)[0]; c, s = C.sum(1), S.sum(1); U = D.astype(np.uint8)
    QD = np.array([(up[far] // pw[j]) % V for j in range(m)]).T.astype(np.uint8)
    for i in range(0, len(far), chunk):
        q = QD[i:i + chunk]; d = np.zeros((len(q), len(U)), np.uint8)
        for j in range(m): d += q[:, j, None] != U[None, :, j]
        mn = d.min(1); w = (d == mn[:, None]).astype(np.float32); f = far[i:i + chunk]
        dmin[f], nmin[f], rmin[f] = mn, w @ c, w @ s
    g = (np.arange(len(Q)), q_grp)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"ham_min": dmin[qinv], "ham_rate_min": (rmin / nmin)[qinv], "ham_n_min": nmin[qinv],
               "ham_rate_1": (r[:, :2].sum(1) / n[:, :2].sum(1))[qinv], "ham_n_1": n[:, :2].sum(1)[qinv],
               "ham_rate_2": (r.sum(1) / n.sum(1))[qinv], "ham_n_2": n.sum(1)[qinv]}
        if k > 1:
            n1, r1 = N[:2].sum(0)[qinv][g], R[:2].sum(0)[qinv][g]
            out["ham_rate_1_grp"], out["ham_n_1_grp"] = r1 / n1, n1
    return {a: b.astype(np.float32) for a, b in out.items()}

def ham(tr, te, cache="data/feat_ham.parquet"):
    """Near-copy features: train/test rating vectors are original rows with a few cells changed (d0 22%, d1 44%, d2 25%)."""
    import os
    if not os.path.exists(cache):
        og = pd.concat([pd.read_csv("data/orig/train.csv"), pd.read_csv("data/orig/test.csv")])
        al = pd.concat([tr, te], ignore_index=True)
        rat = [c for c in al.columns if al[c].dtype.kind == "i" and c not in ("id", AGE, FD, DD)]
        key = lambda d: (d["Class"].astype(str) + "|" + d["Type of Travel"].astype(str))
        cats = {v: i for i, v in enumerate(sorted(set(key(og))))}
        f = ham_core(al[rat].values, og[rat].values, (og.satisfaction == "satisfied").values.astype(float),
                     key(og).map(cats).values, key(al).map(cats).values, k=len(cats))
        pd.DataFrame(f).to_parquet(cache)
    out = pd.read_parquet(cache)
    return out.iloc[: len(tr)].reset_index(drop=True), out.iloc[len(tr):].reset_index(drop=True)


def aux(tr, te, cache="data/feat_aux.parquet", k=2, threads=4):
    """Label-free auxiliary tasks (goodpjw2008 Part 4): predict each rating / categorical from the other 20 columns,
    cross-fitted over train+test. Keeps the expected rating, the probability of the actual value, and the row's pseudo-log-likelihood."""
    import os
    if os.path.exists(cache):
        out = pd.read_parquet(cache); return out.iloc[: len(tr)].reset_index(drop=True), out.iloc[len(tr):].reset_index(drop=True)
    import lightgbm as lgb
    raw = [c for c in tr.columns if c != "id"]; al = pd.concat([tr, te], ignore_index=True)[raw]
    rat = [c for c in raw if al[c].dtype.kind == "i" and c not in (AGE, FD, DD)]
    out = pd.DataFrame(index=al.index); logp = np.zeros(len(al))
    parts = np.random.RandomState(0).randint(0, k, len(al))
    for c in rat + ["Class", "Type of Travel", "Customer Type"]:
        vals, yc = np.unique(al[c].astype(str) if c in CAT else al[c], return_inverse=True)
        X = al.drop(columns=[c]); P = np.zeros((len(al), len(vals)))
        for f in range(k):
            m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.1, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=threads, verbose=-1, random_state=0)
            m.fit(X[parts != f], yc[parts != f]); P[parts == f] = m.predict_proba(X[parts == f])
        if c in rat: out["aux_ev_" + c] = (P * vals.astype(float)).sum(1).astype("float32")
        p = np.clip(P[np.arange(len(al)), yc], 1e-6, 1); out["aux_p_" + c] = p.astype("float32"); logp += np.log(p)
        print("aux", c, flush=True)
    out["aux_sum_logp"] = logp.astype("float32")
    out.to_parquet(cache)
    return out.iloc[: len(tr)].reset_index(drop=True), out.iloc[len(tr):].reset_index(drop=True)
