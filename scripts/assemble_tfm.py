"""Assemble per-fold TFM runs (models/v9_tfm.py) into the usual oof/ shape: oof/<name>.npy, oof/<name>_test.npy, logs/<name>.json.
Usage: python scripts/assemble_tfm.py <member> <src_dir> <name>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from features import folds


def assemble(member, src, name, data="data", root="."):
    src, data, root = Path(src), Path(data), Path(root)
    (root / "oof").mkdir(parents=True, exist_ok=True); (root / "logs").mkdir(parents=True, exist_ok=True)
    y = pd.read_csv(data / "train.csv", usecols=["satisfaction"]).satisfaction.astype(str).eq("True").astype(int).values
    fl = folds(y)
    oof = np.full(len(y), np.nan); cnt = np.zeros(len(y), int)
    tests, ctxs, aucs = [], [], []
    for k in range(5):
        z = np.load(src / f"{member}_f{k}.npz")  # a missing fold raises FileNotFoundError
        val = z["val_idx"]
        assert np.array_equal(val, fl[k][1]), f"fold {k}: validation rows differ from features.folds(y)"
        oof[val] = z["val_pred"]; cnt[val] += 1
        tests.append(z["test_pred"])
        ctxs.append(json.load(open(src / f"{member}_f{k}.json"))["ctx"])
        aucs.append(float(roc_auc_score(y[val], z["val_pred"])))
    assert (cnt == 1).all(), "rows not filled exactly once"
    np.save(root / "oof" / f"{name}.npy", oof)
    np.save(root / "oof" / f"{name}_test.npy", np.mean(tests, axis=0))
    auc = float(roc_auc_score(y, oof))
    json.dump({"name": name, "oof": auc, "folds": aucs, "ctx": ctxs}, open(root / "logs" / f"{name}.json", "w"), indent=1)
    return auc


if __name__ == "__main__":
    member, src, name = sys.argv[1:4]
    auc = assemble(member, src, name)
    print(name, f"{auc:.6f}", [round(a, 6) for a in json.load(open(f"logs/{name}.json"))["folds"]])
