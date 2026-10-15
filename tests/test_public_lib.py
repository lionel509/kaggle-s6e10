"""blend/public_lib.py: every public oof_*.csv loads once, aligned by id; check_meta proves the split is ours."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "blend")); sys.path.insert(0, str(REPO / "models"))

N, NT = 200, 60
IDS, TIDS = np.arange(N), np.arange(100, 100 + NT)


def _dir(root):
    d = root / "public_out" / "x_tag"
    d.mkdir(parents=True)
    return d


def test_every_oof_file_loads_once(tmp_path, monkeypatch):
    import public_lib
    monkeypatch.chdir(tmp_path)
    d = _dir(tmp_path)
    rng = np.random.default_rng(0)
    pd.DataFrame({"id": IDS, "pred": rng.random(N)}).to_csv(d / "oof_xgb.csv", index=False)
    pd.DataFrame({"id": TIDS, "pred": rng.random(NT)}).to_csv(d / "test_xgb.csv", index=False)
    pd.DataFrame({"id": IDS, "a": rng.random(N), "b": rng.random(N), "fold": np.arange(N) % 5}).to_csv(d / "oof_members.csv", index=False)
    pd.DataFrame({"id": TIDS, "a": rng.random(NT), "b": rng.random(NT)}).to_csv(d / "test_members.csv", index=False)
    pd.DataFrame({"id": IDS, "pred": rng.random(N)}).to_csv(d / "oof_foo.csv", index=False)
    pd.DataFrame({"id": TIDS, "pred": rng.random(NT)}).to_csv(d / "test_foo.csv", index=False)
    oof, tst = public_lib.load_public(IDS, TIDS)
    expected = {"x_tag:oof_xgb:pred", "x_tag:oof_members:a", "x_tag:oof_members:b", "x_tag:oof_foo:pred"}
    assert set(oof) == expected
    assert set(tst) == expected
    assert len(oof) == 4


def test_member_without_id_column_raises(tmp_path, monkeypatch):
    import public_lib
    monkeypatch.chdir(tmp_path)
    d = _dir(tmp_path)
    pd.DataFrame({"pred": np.arange(N, dtype=float)}).to_csv(d / "oof_bar.csv", index=False)
    pd.DataFrame({"pred": np.arange(NT, dtype=float)}).to_csv(d / "test_bar.csv", index=False)
    with pytest.raises(ValueError, match="no id column"):
        public_lib.load_public(IDS, TIDS)


def _meta_dir(root, written, pred, y, fold):
    """meta.json holds the AUCs of `pred`; the csv holds `written` (the same rows unless the test breaks alignment)."""
    from sklearn.metrics import roc_auc_score
    d = root / "pub" / "tabpfnmember_x"
    d.mkdir(parents=True)
    pd.DataFrame({"id": IDS, "pred": written}).to_csv(d / "oof_tabpfn_m.csv", index=False)
    aucs = [float(roc_auc_score(y[fold == k], pred[fold == k])) for k in range(5)]
    meta = {"folds": "test", "members": {"tabpfn_m": {"oof_auc": float(np.mean(aucs)), "fold_auc": aucs}}}
    (d / "meta.json").write_text(json.dumps(meta))
    return str(d) + "/"


def _split(seed):
    from features import folds
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, N)
    y[:2] = [0, 1]
    fold = np.zeros(N, int)
    for i, (a, b) in enumerate(folds(y)):
        fold[b] = i
    return y, fold


def test_check_meta_passes_on_matching_meta(tmp_path, monkeypatch):
    import public_lib
    monkeypatch.chdir(tmp_path)
    y, fold = _split(1)
    pred = np.random.default_rng(2).random(N)
    d = _meta_dir(tmp_path, pred, pred, y, fold)
    assert public_lib.check_meta(d, IDS, y, fold) == []


def test_check_meta_flags_rolled_predictions(tmp_path, monkeypatch):
    import public_lib
    monkeypatch.chdir(tmp_path)
    y, fold = _split(3)
    pred = np.random.default_rng(4).random(N)
    d = _meta_dir(tmp_path, np.roll(pred, 1), pred, y, fold)  # meta was computed on the unrolled rows
    problems = public_lib.check_meta(d, IDS, y, fold)
    assert len(problems) == 5


def _member(root, name, stem, va, ta, oof_ids=None):
    d = root / "public_out" / name
    d.mkdir(parents=True)
    pd.DataFrame({"id": IDS if oof_ids is None else oof_ids, "a": va}).to_csv(d / f"oof_{stem}.csv", index=False)
    pd.DataFrame({"id": TIDS, "a": ta}).to_csv(d / f"test_{stem}.csv", index=False)


def test_same_tag_members_do_not_overwrite(tmp_path, monkeypatch):
    import public_lib
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    va, vb = rng.random(N), rng.random(N)
    _member(tmp_path, "tabpfnmember_alice", "foo", va, rng.random(NT))
    _member(tmp_path, "tabpfnmember_bob", "foo", vb, rng.random(NT))
    oof, _ = public_lib.load_public(IDS, TIDS)
    vals = list(oof.values())
    assert any(np.allclose(v, va) for v in vals), "alice's member column was overwritten or lost"
    assert any(np.allclose(v, vb) for v in vals), "bob's member column is missing"


def test_public_tag_own_is_reserved(tmp_path, monkeypatch):
    import public_lib
    monkeypatch.chdir(tmp_path)
    _member(tmp_path, "own_goodpjw2008", "foo", np.random.default_rng(1).random(N), np.random.default_rng(2).random(NT))
    with pytest.raises(ValueError, match="reserved"):
        public_lib.load_public(IDS, TIDS)


def test_add_refuses_a_duplicate_key(tmp_path):
    import public_lib
    o = pd.DataFrame({"a": np.arange(N, dtype=float)}, index=IDS)
    t = pd.DataFrame({"a": np.arange(NT, dtype=float)}, index=TIDS)
    oof, tst = {}, {}
    public_lib._add(oof, tst, "x_one", "oof_foo.csv", o, t)
    with pytest.raises(ValueError, match="duplicate"):
        public_lib._add(oof, tst, "x_one", "oof_foo.csv", o, t)
