"""Load every public OOF/test member aligned to our ids, as named columns. Their split is ours (StratifiedKFold 5, shuffle, 42).
check_meta proves it per member: the published fold AUCs must match ours to 1e-6.
CLI: python blend/public_lib.py --check <dir>   (exit 1 on any mismatch)"""
import glob, json, os, sys
import numpy as np, pandas as pd

PAIRS = [("oof_members.csv", "test_members.csv"), ("oof_xgb.csv", "test_xgb.csv"), ("oof_preds.csv", "submission.csv")]
SKIP = {fo for fo, _ in PAIRS}  # already loaded by the PAIRS loop


def _align(path, ids):
    df = pd.read_csv(path)
    if "id" not in df:
        raise ValueError(f"{path}: no id column")
    return df.set_index("id").reindex(ids)


def _add(oof, tst, tag, fo, o, t, sfx=""):
    for c in o.columns:
        tc = c if c in t else "satisfaction"
        if c == "fold" or tc not in t or o[c].isna().any() or t[tc].isna().any(): continue
        key = f"{tag}:{fo[:-4]}:{c}{sfx}"
        if key in oof: raise ValueError(f"duplicate public column {key}")  # two members must never share a key silently
        oof[key] = o[c].to_numpy(float); tst[key] = t[tc].to_numpy(float)


def load_public(train_ids, test_ids):
    oof, tst = {}, {}
    for d in sorted(glob.glob("public_out/*/")):
        # keys are <family>:<file>:<col>@<rest of folder>: the family prefix is what the #6 --drop regexes match,
        # the @suffix keeps two folders of one family from sharing columns
        name = os.path.basename(d.rstrip("/"))
        tag, _, rest = name.partition("_")
        if tag == "own" or ":" in name: raise ValueError(f"{d}: tag 'own' is reserved and ':' is the key separator")  # run_blend's own: keys
        sfx = f"@{rest}" if rest else ""
        for fo, ft in PAIRS:
            if not (os.path.exists(d + fo) and os.path.exists(d + ft)): continue
            o, t = pd.read_csv(d + fo), pd.read_csv(d + ft)
            if len(o) != len(train_ids): continue
            o, t = o.set_index("id").reindex(train_ids), t.set_index("id").reindex(test_ids)
            _add(oof, tst, tag, fo, o, t, sfx)
        # every other oof_*.csv that has a matching test_*.csv (the TFM member dataset and later pulls)
        for p in sorted(glob.glob(d + "oof_*.csv")):
            fo = os.path.basename(p)
            if fo in SKIP or not os.path.exists(d + "test_" + fo[4:]): continue
            _add(oof, tst, tag, fo, _align(d + fo, train_ids), _align(d + "test_" + fo[4:], test_ids), sfx)
    return oof, tst


def _fold_aucs(d, m, train_ids, y, fold):
    from sklearn.metrics import roc_auc_score
    pred = _align(d + f"oof_{m}.csv", train_ids)["pred"].to_numpy(float)
    missing = int(np.isnan(pred).sum())
    if missing:
        return None, missing
    return [float(roc_auc_score(y[fold == k], pred[fold == k])) for k in range(5)], 0


def check_meta(d, train_ids, y, fold):
    """One message per member fold whose AUC on our folds differs from meta.json by more than 1e-6."""
    meta = json.load(open(d + "meta.json"))
    out = []
    for m, info in meta["members"].items():
        aucs, missing = _fold_aucs(d, m, train_ids, y, fold)
        if aucs is None:
            out.append(f"{m}: {missing} rows missing from our ids")
            continue
        for k, a in enumerate(aucs):
            b = info["fold_auc"][k]
            if abs(a - b) > 1e-6:
                out.append(f"{m} fold {k}: ours {a:.7f} theirs {b:.7f}")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--check":
        sys.exit("usage: python blend/public_lib.py --check <dir>")
    sys.path.insert(0, "models")
    from features import folds
    d = sys.argv[2].rstrip("/") + "/"
    tr = pd.read_csv("data/train.csv", usecols=["id", "satisfaction"])
    y = tr.satisfaction.astype(str).eq("True").astype(int).values
    fold = np.zeros(len(y), int)
    for i, (_, b) in enumerate(folds(y)): fold[b] = i
    meta = json.load(open(d + "meta.json"))
    for m, info in meta["members"].items():
        aucs, missing = _fold_aucs(d, m, tr.id.values, y, fold)
        if aucs is None:
            print(f"  {m}: {missing} rows missing"); continue
        diff = max(abs(a - b) for a, b in zip(aucs, info["fold_auc"]))
        print(f"  {m}: folds {' '.join(f'{a:.6f}' for a in aucs)}  max abs diff {diff:.2e}")
    bad = check_meta(d, tr.id.values, y, fold)
    print("\n".join(bad) if bad else "OK")
    sys.exit(1 if bad else 0)
