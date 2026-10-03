"""v6: weak decorrelated members on cached features. Usage: K=20 python v6_simple.py knn|lr [k=150]
knn: standardized teacher logits + ratings + key features; lr: one-hot ratings + teacher logits. Test = per-fold average."""
import os, sys, time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
sys.path.insert(0, "models"); from features import folds


def model(kind, X, onehot, k):
    if kind == "knn": return make_pipeline(StandardScaler(), KNeighborsClassifier(k, n_jobs=4))
    num = [c for c in X.columns if c not in onehot]
    ct = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), onehot), ("num", StandardScaler(), num)])
    return make_pipeline(ct, LogisticRegression(C=1.0, max_iter=2000))


def run(kind, X, XT, y, onehot, K=20, k=150):
    oof, pt = np.zeros(len(y)), np.zeros(len(XT))
    for i, (a, b) in enumerate(folds(y, K)):
        m = model(kind, X, onehot, k).fit(X.iloc[a], y[a])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]; pt += m.predict_proba(XT)[:, 1] / K
        print(f"fold {i} {roc_auc_score(y[b], oof[b]):.6f}", flush=True)
    return oof, pt


if __name__ == "__main__":
    from features import load, extra, teachers, CAT, AGE, FD, DD, AD
    kind = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 150; K = int(os.environ.get("K", 5)); t0 = time.time()
    tr, te, og, y, og_y = load(); T, TT = teachers(tr, te, og, og_y); E, ET = extra(tr, te)
    rat = [c for c in tr.columns if tr[c].dtype.kind == "i" and c not in ("id", AGE, FD, DD)]
    def frame(d, t, e):
        f = pd.concat([t, d[rat].reset_index(drop=True)], axis=1)
        if kind == "knn":
            for c in CAT: f[c] = d[c].cat.codes.values
            f[AGE], f["lfd"], f["ldel"] = d[AGE].values, np.log1p(d[FD].values), np.log1p(d[[DD, AD]].fillna(0).max(axis=1).values)
            f["om"] = e["om_ratings"].fillna(0.4345).values
        else:
            for c in CAT: f[c] = d[c].astype(str).values
        return f
    X, XT = frame(tr, T, E), frame(te, TT, ET)
    name = f"v6_{kind}" + (f"{k}" if kind == "knn" else "") + (f"_k{K}" if K != 5 else "")
    oof, pt = run(kind, X, XT, y, rat + CAT, K, k)
    print(f"{name} OOF {roc_auc_score(y, oof):.6f} ({time.time()-t0:.0f}s)")
    np.save(f"oof/{name}.npy", oof); np.save(f"oof/{name}_test.npy", pt)
