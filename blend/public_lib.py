"""Load every public OOF/test member aligned to our ids, as named columns. Their split is ours (StratifiedKFold 5, shuffle, 42)."""
import glob, os, pandas as pd

PAIRS = [("oof_members.csv", "test_members.csv"), ("oof_xgb.csv", "test_xgb.csv"), ("oof_preds.csv", "submission.csv")]


def load_public(train_ids, test_ids):
    oof, tst = {}, {}
    for d in sorted(glob.glob("public_out/*/")):
        tag = os.path.basename(d.rstrip("/")).split("_")[0]
        for fo, ft in PAIRS:
            if not (os.path.exists(d + fo) and os.path.exists(d + ft)): continue
            o, t = pd.read_csv(d + fo), pd.read_csv(d + ft)
            if len(o) != len(train_ids): continue
            o, t = o.set_index("id").reindex(train_ids), t.set_index("id").reindex(test_ids)
            for c in o.columns:
                tc = c if c in t else "satisfaction"
                if c == "fold" or tc not in t or o[c].isna().any() or t[tc].isna().any(): continue
                oof[f"{tag}:{fo[:-4]}:{c}"] = o[c].to_numpy(float); tst[f"{tag}:{fo[:-4]}:{c}"] = t[tc].to_numpy(float)
    return oof, tst
