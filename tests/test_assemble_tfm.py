"""models/v9_tfm.py per-fold runs + scripts/assemble_tfm.py. Uses the dummy member, so no tabicl or tabpfn import.
Invariant: no OOF prediction comes from a model whose context held that row; fold k's rows are exactly features.folds(y)[k][1]."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models")); sys.path.insert(0, str(REPO / "scripts"))
from features import folds  # noqa: E402

SWITCH_KEYS = ("MEMBER", "FOLDS", "CTX", "DATA", "OUT", "DEVICE", "_LEAK_TEST", "SEED", "PRED_BATCH")


def _data(root, seed, n):
    rng = np.random.default_rng(seed)
    d = root / "data"
    d.mkdir(parents=True)
    y = rng.integers(0, 2, n); y[:2] = [0, 1]
    tr = pd.DataFrame({
        "id": np.arange(n),
        "Gender": rng.choice(["Male", "Female"], n),
        "Customer Type": rng.choice(["Loyal Customer", "disloyal Customer"], n),
        "Type of Travel": rng.choice(["Business travel", "Personal Travel"], n),
        "Class": rng.choice(["Eco", "Business", "Eco Plus"], n),
        "Age": rng.integers(7, 80, n),
        "Flight Distance": rng.integers(50, 4000, n),
        "Arrival Delay in Minutes": np.where(rng.random(n) < 0.05, np.nan, rng.integers(0, 60, n)).astype(float),
        "satisfaction": np.where(y == 1, "True", "False"),
    })
    m = n // 3
    te = tr.drop(columns=["satisfaction"]).iloc[:m].copy(); te["id"] = np.arange(n, n + m)
    tr.to_csv(d / "train.csv", index=False); te.to_csv(d / "test.csv", index=False)
    return d, y


def _run(data, out, folds_="0,1,2,3,4", ctx=0, leak=False):
    env = {k: v for k, v in os.environ.items() if k not in SWITCH_KEYS}
    env |= {"MEMBER": "dummy", "FOLDS": folds_, "CTX": str(ctx), "DATA": str(data), "OUT": str(out), "DEVICE": "cpu"}
    if leak:
        env["_LEAK_TEST"] = "1"
    return subprocess.run([sys.executable, str(REPO / "models" / "v9_tfm.py")], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("seed,n,ctx", [(0, 1000, 0), (1, 1500, 50), (2, 2400, 400), (3, 3000, 3000)])
def test_assemble_matches_folds_and_test_mean(tmp_path, seed, n, ctx):
    from assemble_tfm import assemble
    data, y = _data(tmp_path, seed, n)
    out = tmp_path / "out"
    p = _run(data, out, ctx=ctx)
    assert p.returncode == 0, p.stderr[-2000:]
    root = tmp_path / "root"; (root / "oof").mkdir(parents=True); (root / "logs").mkdir()
    auc = assemble("dummy", str(out), "t", data=str(data), root=str(root))
    oof = np.load(root / "oof" / "t.npy")
    tst = np.load(root / "oof" / "t_test.npy")
    assert np.isfinite(oof).all()
    fl = folds(y)
    for k, (_, val) in enumerate(fl):
        z = np.load(out / f"dummy_f{k}.npz")
        assert np.array_equal(z["val_idx"], val)
        assert roc_auc_score(y[val], oof[val]) == pytest.approx(roc_auc_score(y[val], z["val_pred"]))
    J = json.load(open(root / "logs" / "t.json"))
    assert len(J["folds"]) == 5 and len(J["ctx"]) == 5
    assert auc == pytest.approx(J["oof"])
    mean_test = np.mean([np.load(out / f"dummy_f{k}.npz")["test_pred"] for k in range(5)], axis=0)
    assert np.allclose(tst, mean_test)


def test_swapped_validation_rows_raise(tmp_path):
    from assemble_tfm import assemble
    data, _ = _data(tmp_path, 5, 1200)
    out = tmp_path / "out"
    assert _run(data, out).returncode == 0
    z = dict(np.load(out / "dummy_f0.npz"))
    z["val_idx"] = z["val_idx"].copy(); z["val_idx"][[0, 1]] = z["val_idx"][[1, 0]]
    np.savez(out / "dummy_f0.npz", **z)
    root = tmp_path / "root"; (root / "oof").mkdir(parents=True); (root / "logs").mkdir()
    with pytest.raises(AssertionError):
        assemble("dummy", str(out), "t", data=str(data), root=str(root))


def test_deleted_fold_file_raises(tmp_path):
    from assemble_tfm import assemble
    data, _ = _data(tmp_path, 6, 1200)
    out = tmp_path / "out"
    assert _run(data, out).returncode == 0
    (out / "dummy_f3.npz").unlink()
    root = tmp_path / "root"; (root / "oof").mkdir(parents=True); (root / "logs").mkdir()
    with pytest.raises(FileNotFoundError):
        assemble("dummy", str(out), "t", data=str(data), root=str(root))


def test_leak_guard_stops_fold(tmp_path):
    data, _ = _data(tmp_path, 7, 1200)
    p = _run(data, tmp_path / "out_leak", folds_="0", ctx=100, leak=True)
    assert p.returncode != 0
    assert "context overlaps validation" in p.stderr


def test_rerun_skips_finished_folds(tmp_path):
    data, _ = _data(tmp_path, 8, 1200)
    out = tmp_path / "out"
    assert _run(data, out, folds_="0").returncode == 0
    p = _run(data, out, folds_="0")
    assert p.returncode == 0, p.stderr[-2000:]
    assert "SKIP fold 0" in p.stdout


FAKE_TABICL = '''
import os, sys
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
MARK = os.environ["FAKE_MARK"]
class TabICLClassifier:
    def __init__(self, **kw): pass
    def fit(self, X, y):
        if not os.path.exists(MARK):  # first call: simulated OOM at the full context
            open(MARK, "w").write(str(len(X)))
            raise RuntimeError("out of memory (simulated)")
        open(MARK + ".retry", "w").write(f"{len(X)} {sys.exc_info()[1] is None}")  # retry: rows, and no live exception
        self.m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)).fit(X, y)
        return self
    def predict_proba(self, A): return self.m.predict_proba(A)
'''


def _oom_run(tmp_path, ctx):
    data, _ = _data(tmp_path, 4, 1000)  # 800 fold-train rows
    fake = tmp_path / "fake"; fake.mkdir()
    (fake / "tabicl.py").write_text(FAKE_TABICL)
    env = {k: v for k, v in os.environ.items() if k not in SWITCH_KEYS}
    env |= {"MEMBER": "tabicl_raw", "FOLDS": "0", "CTX": str(ctx), "DATA": str(data), "OUT": str(tmp_path / "out"),
            "DEVICE": "cpu", "FAKE_MARK": str(tmp_path / "mark"), "PYTHONPATH": str(fake)}
    return subprocess.run([sys.executable, str(REPO / "models" / "v9_tfm.py")], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=600)


def test_oom_retry_is_smaller_and_runs_after_the_handler(tmp_path):
    p = _oom_run(tmp_path, ctx=100000)
    assert p.returncode == 0, p.stderr[-1500:]
    J = json.load(open(tmp_path / "out" / "tabicl_raw_f0.json"))
    assert 1 <= J["ctx"] < 800, f"OOM retry used ctx={J['ctx']}, not smaller than the 800 that OOM'd"
    assert (tmp_path / "mark.retry").read_text().endswith("True"), "retry ran inside the except block"


def test_oom_retry_at_ctx1_fails_loudly(tmp_path):
    p = _oom_run(tmp_path, ctx=1)
    assert p.returncode != 0, "OOM at CTX=1 was retried on all fold-train rows"
    assert not (tmp_path / "out" / "tabicl_raw_f0.npz").exists()
