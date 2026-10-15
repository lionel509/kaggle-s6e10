"""v9: TabICL / TabPFN member on the Mac (MPS or CPU, no GPU). One fold per run, resumable by fold.
Usage: MEMBER=tabicl_raw CTX=100000 FOLDS=0,1 python models/v9_tfm.py
  MEMBER: tabicl_raw | tabpfn_catfd10 | dummy (tests only). CTX: context rows (0 = all fold-train rows).
  Writes OUT/<MEMBER>_f<k>.json then OUT/<MEMBER>_f<k>.npz; a fold whose .npz exists is skipped (resume).
  Context rows are always fold-train rows, so no OOF row sees itself. Assemble with scripts/assemble_tfm.py."""
import os
T = os.environ.get("THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", T); os.environ.setdefault("MKL_NUM_THREADS", T)
import json, resource, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "models"); from features import folds

try:
    import torch
    torch.set_num_threads(int(T))
except ImportError:
    torch = None
OOM = getattr(torch, "OutOfMemoryError", RuntimeError) if torch is not None else RuntimeError

P = dict(MEMBER="tabicl_raw", FOLDS="0", CTX=100000, PRED_BATCH=8192, SEED=0)
MEMBER = os.environ.get("MEMBER", P["MEMBER"])
FOLD_LIST = [int(k) for k in os.environ.get("FOLDS", P["FOLDS"]).split(",")]
CTX, PRED_BATCH, SEED = (int(os.environ.get(k, P[k])) for k in ("CTX", "PRED_BATCH", "SEED"))
DATA = Path(os.environ.get("DATA", "data")); OUT = Path(os.environ.get("OUT", f"tfm_out/{MEMBER}")); OUT.mkdir(parents=True, exist_ok=True)
if os.environ.get("DEVICE"):
    DEVICE = os.environ["DEVICE"]
else:
    DEVICE = "mps" if torch is not None and torch.backends.mps.is_available() else "cpu"

# features: train.csv columns minus id and label, in file order; train and test on one concatenated frame
tr = pd.read_csv(DATA / "train.csv"); te = pd.read_csv(DATA / "test.csv")
y = tr.satisfaction.astype(str).eq("True").astype(int).values
FEATS = [c for c in tr.columns if c not in ("id", "satisfaction")]
full = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)
for c in full.columns:
    if not pd.api.types.is_numeric_dtype(full[c]):  # text columns (object or pandas string dtype)
        full[c] = full[c].astype("category").cat.codes
full["Arrival Delay in Minutes"] = full["Arrival Delay in Minutes"].fillna(-1)
COLS = list(full.columns)
X = full.to_numpy(np.float32); n_tr = len(tr)
FOLD = folds(y)
CAT_NAMES = ["Flight Distance", "Gender", "Customer Type", "Type of Travel", "Class"]


def _proba(clf, A):
    return np.concatenate([clf.predict_proba(A[i:i + PRED_BATCH])[:, 1] for i in range(0, len(A), PRED_BATCH)])


def _tabicl(Xc, yc, Xq, dev):
    from tabicl import TabICLClassifier
    clf = TabICLClassifier(n_estimators=1, random_state=SEED, device=dev, kv_cache=True, offload_mode="auto").fit(Xc, yc)
    return [_proba(clf, A) for A in Xq], {}


def _tabpfn(Xc, yc, Xq, dev):
    os.environ.setdefault("TABPFN_MPS_MEMORY_FRACTION", "0.6"); os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")
    from tabpfn import TabPFNClassifier
    info = {}
    try:
        from tabpfn import ModelVersion
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5); info["version"] = "3.5"
    except ImportError:
        clf = TabPFNClassifier(); info["version"] = "default"
    clf.set_params(device=dev, ignore_pretraining_limits=True)
    fd, n = COLS.index("Flight Distance"), len(COLS)
    cat = [fd, n] + [COLS.index(c) for c in CAT_NAMES[1:]]  # fd10 is the appended column
    with_fd10 = lambda A: np.c_[A, np.floor(A[:, fd] / 10)].astype(np.float32)
    try:
        clf.set_params(categorical_features_indices=cat); info["cat_idx"] = cat
    except ValueError:
        info["cat_idx"] = None
    clf.fit(with_fd10(Xc), yc)
    return [_proba(clf, with_fd10(A)) for A in Xq], info


def _dummy(Xc, yc, Xq, dev):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)).fit(Xc, yc)
    return [_proba(clf, A) for A in Xq], {}


MEMBERS = {"tabicl_raw": _tabicl, "tabpfn_catfd10": _tabpfn, "dummy": _dummy}


def _is_oom(e):
    return isinstance(e, OOM) or (isinstance(e, RuntimeError) and "out of memory" in str(e))


def run_fold(k, ctx):
    tr_idx, val_idx = FOLD[k]
    if ctx == 0 or ctx >= len(tr_idx):
        ctx_idx = tr_idx
    else:
        ctx_idx = np.sort(np.random.default_rng(SEED + k).choice(tr_idx, ctx, replace=False))
    if os.environ.get("_LEAK_TEST") == "1":  # guard self-test: put one validation row into the context
        ctx_idx = np.append(ctx_idx, val_idx[0])
    assert not np.intersect1d(ctx_idx, val_idx).size, "context overlaps validation"
    t0 = time.time()
    (vp, tp), info = MEMBERS[MEMBER](X[ctx_idx], y[ctx_idx], [X[val_idx], X[n_tr:]], DEVICE)
    t1 = time.time()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** (30 if sys.platform == "darwin" else 20)
    auc = float(roc_auc_score(y[val_idx], vp))
    J = {"member": MEMBER, "fold": k, "ctx": int(len(ctx_idx)), "device": DEVICE, **info,
         "fit_s": round(t1 - t0, 1), "pred_s": round(time.time() - t1, 1), "peak_mem_gb": round(peak, 2), "auc": auc}
    J.setdefault("version", None); J.setdefault("cat_idx", None)
    json.dump(J, open(OUT / f"{MEMBER}_f{k}.json", "w"), indent=1)
    tmp = OUT / f"{MEMBER}_f{k}.tmp.npz"  # write then rename, so a killed job never leaves a half .npz that resume would skip
    np.savez(tmp, val_idx=val_idx.astype(np.int64), val_pred=vp.astype(np.float64), test_pred=tp.astype(np.float64))
    os.replace(tmp, OUT / f"{MEMBER}_f{k}.npz")
    print(f"fold {k}: ctx {J['ctx']} | auc {auc:.6f} | fit {J['fit_s']}s pred {J['pred_s']}s | peak {J['peak_mem_gb']} GB", flush=True)


for k in FOLD_LIST:
    if (OUT / f"{MEMBER}_f{k}.npz").exists():
        print(f"SKIP fold {k}", flush=True); continue
    try:
        run_fold(k, CTX)
    except Exception as e:
        if not _is_oom(e):
            raise
        half = CTX // 2 if CTX else len(FOLD[k][0]) // 2
        print(f"OOM fold {k} ctx {CTX} -> {half}", flush=True)
        run_fold(k, half)  # one retry at half the context; a second failure propagates
