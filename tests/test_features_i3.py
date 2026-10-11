"""C and D: block keys (BLKTE) and the 5th teacher (TEACH=3). Synthetic frames only, no data/."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models"))
import features  # noqa: E402

RATINGS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
           "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
           "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]
STD = {"te_fd", "te_fd_cls", "te_fd_tt", "te_age_cls_tt_ct", "te_fd10"}


def _frame(n=6, seed=0, index=None):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({c: r.integers(0, 6, n) for c in RATINGS})
    df["Flight Distance"] = r.integers(100, 3000, n); df["Age"] = r.integers(10, 80, n)
    df["Class"] = r.choice(["Eco", "Business"], n); df["Type of Travel"] = r.choice(["Personal", "Business"], n)
    df["Customer Type"] = r.choice(["Loyal", "disloyal"], n)
    if index is not None:
        df.index = index
    return df


def test_blocks_constants():
    assert len(features.RATINGS) == 13
    assert features.BLOCKS["blk_all"] == features.RATINGS
    assert set(features.BLOCKS["blk_conv"]) <= set(features.RATINGS)
    assert set(features.BLOCKS["blk_comf"]) <= set(features.RATINGS)


def test_te_keys_unchanged_without_blkte(monkeypatch):
    monkeypatch.delenv("BLKTE", raising=False)
    assert set(features.te_keys(_frame())) == STD


def test_te_keys_blkte(monkeypatch):
    monkeypatch.setenv("BLKTE", "1")
    df = _frame(index=np.arange(10, 16))
    keys = features.te_keys(df)
    assert set(keys) == STD | {"te_blk_conv", "te_blk_comf", "te_blk_all"}
    for k in ("te_blk_conv", "te_blk_comf", "te_blk_all"):
        assert keys[k].index.equals(df.index)
    a = df.iloc[[0]].copy(); b = a.copy(); b["Online boarding"] = (int(a["Online boarding"].iloc[0]) + 1) % 6
    ka, kb = features.te_keys(a), features.te_keys(b)
    assert ka["te_blk_conv"].iloc[0] == kb["te_blk_conv"].iloc[0]
    assert ka["te_blk_comf"].iloc[0] == kb["te_blk_comf"].iloc[0]
    assert ka["te_blk_all"].iloc[0] != kb["te_blk_all"].iloc[0]


def test_teachers_teach3(tmp_path, monkeypatch):
    n_tr, n_te = 8, 5; n = n_tr + n_te
    r = np.random.default_rng(1)
    cache = tmp_path / "teachers.parquet"; cat2 = tmp_path / "cat2.parquet"
    pd.DataFrame({c: r.normal(size=n).astype("float32") for c in ("t_lgb1500", "t_xgb8", "t_cat8", "logits")}).to_parquet(cache)
    pd.DataFrame({"t_catx": r.normal(size=n).astype("float32")}).to_parquet(cat2)
    tr = pd.DataFrame({"id": range(n_tr), "x": 0.0}); te = pd.DataFrame({"id": range(n_te), "x": 0.0})
    monkeypatch.setenv("TEACH", "3")
    a, b = features.teachers(tr, te, pd.DataFrame(), np.zeros(0), cache=str(cache), cat2=str(cat2))
    assert "t_catx" in a.columns and len(a) == n_tr
    assert "t_catx" in b.columns and len(b) == n_te
    np.testing.assert_allclose(b["t_catx"].values, pd.read_parquet(cat2)["t_catx"].values[n_tr:])
    monkeypatch.setenv("TEACH", "1")
    a1, _ = features.teachers(tr, te, pd.DataFrame(), np.zeros(0), cache=str(cache), cat2=str(cat2))
    assert "t_catx" not in a1.columns
