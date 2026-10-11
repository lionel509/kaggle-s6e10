"""Breaker tests for PR #4 (head e05cc72, branch 3-one-shot-queue). Each test states the behaviour the plan promises; a failure is a break."""
import os, sys
from pathlib import Path

import numpy as np, pandas as pd, pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models"))

RAT = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
       "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
       "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]


def _frame(n, seed):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"id": np.arange(n), "Age": r.integers(7, 80, n), "Flight Distance": r.integers(31, 4000, n)})
    for c in RAT:
        d[c] = r.integers(0, 6, n)
    d["Departure Delay in Minutes"] = r.integers(0, 50, n)
    d["Arrival Delay in Minutes"] = r.integers(0, 50, n).astype(float)
    d["Gender"] = r.choice(["Male", "Female"], n)
    d["Customer Type"] = r.choice(["Loyal Customer", "disloyal Customer"], n)
    d["Type of Travel"] = r.choice(["Business travel", "Personal Travel"], n)
    d["Class"] = r.choice(["Eco", "Eco Plus", "Business"], n)
    for c in CATS:
        d[c] = d[c].astype("category")
    return d


def _labels(n, seed):
    return np.random.default_rng(seed).integers(0, 2, n)


# 1. Frame: TEACH=3 with no teachers cache silently drops the CatBoost teacher column.
def test_teach3_without_cache_keeps_catx(tmp_path, monkeypatch):
    import features
    tr, te, og = _frame(120, 0), _frame(40, 1), _frame(200, 2)
    cat2 = tmp_path / "cat2.parquet"
    pd.DataFrame({"t_catx": np.linspace(-1, 1, 160).astype("float32")}).to_parquet(cat2)
    monkeypatch.setenv("TEACH", "3")
    trX, teX = features.teachers(tr, te, og, _labels(200, 3), cache=str(tmp_path / "missing.parquet"), cat2=str(cat2))
    assert "t_catx" in trX.columns and "t_catx" in teX.columns, list(trX.columns)


# 2. Frame: SMOKE runs with TEACH=3 cannot find the smoke teacher that teacher_cat.py writes under smoke/.
def test_smoke_teach3_finds_smoke_teacher(tmp_path, monkeypatch):
    import features
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMOKE", "50")
    monkeypatch.setenv("TEACH", "3")
    (tmp_path / "data").mkdir(); (tmp_path / "smoke").mkdir()
    tr, te, og = _frame(120, 0), _frame(40, 1), _frame(200, 2)
    pd.DataFrame({"t_lgb1500": np.zeros(160, "float32")}).to_parquet(tmp_path / "data" / "feat_teachers.parquet")
    pd.DataFrame({"t_catx": np.zeros(160, "float32")}).to_parquet(tmp_path / "smoke" / "feat_teacher_cat2.parquet")
    trX, _ = features.teachers(tr, te, og, _labels(200, 3))
    assert "t_catx" in trX.columns


# 3. Frame: v8_tabm.py calls teachers() unconditionally and its name is fixed to *_aux_teach,
#    but teachers() reads TEACH from the environment, so an exported TEACH changes F's frame under the same name.
def test_teachers_frame_independent_of_teach_env(tmp_path, monkeypatch):
    import features
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    tr, te, og = _frame(120, 0), _frame(40, 1), _frame(200, 2)
    pd.DataFrame({"t_lgb1500": np.zeros(160, "float32")}).to_parquet("data/feat_teachers.parquet")
    pd.DataFrame({"t_catx": np.ones(160, "float32")}).to_parquet("data/feat_teacher_cat2.parquet")
    pd.DataFrame({"t_mlp": np.ones(160, "float32")}).to_parquet("data/feat_teacher_mlp.parquet")
    monkeypatch.delenv("TEACH", raising=False)
    base, _ = features.teachers(tr, te, og, _labels(200, 3))
    monkeypatch.setenv("TEACH", "3")
    with_env, _ = features.teachers(tr, te, og, _labels(200, 3))
    assert list(base.columns) == list(with_env.columns), (list(base.columns), list(with_env.columns))


# 4. Soft targets (B): the plan says changing y on a student's validation rows cannot change its soft targets.
#    fold_ids(y) is built from all labels, so flipping one validation label reshuffles the teacher pairs.
def _write_teachers(s, y_fold, y_lab, d):
    for fam in ("xgb", "mlp"):
        for S in s.PAIRS + s.SINGLES:
            a, b = s.rows_for(y_fold, S)
            np.save(s.teacher_path(fam, S, d), np.full(len(b), y_lab[a].mean(), dtype="float32"))


def test_soft_target_unchanged_by_flipping_val_label_with_fixed_teachers(tmp_path):
    import soft as s
    y = (np.random.default_rng(0).random(2000) < 0.43).astype(int)
    _write_teachers(s, y, y, str(tmp_path))  # teacher files fixed, built from the true labels
    for a, b, j in s.nested_folds(y, 5):
        y2 = y.copy(); y2[b] = 1 - y2[b]
        np.testing.assert_array_equal(s.soft_target(y, a, j, d=str(tmp_path)), s.soft_target(y2, a, j, d=str(tmp_path)))


