"""Breaker test for PR #4 (models/features.py teachers, TEACH=3). A failure is a break."""
import sys
from pathlib import Path

import numpy as np, pandas as pd

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


# Frame: TEACH=3 with no teachers cache silently drops the CatBoost teacher column.
def test_teach3_without_cache_keeps_catx(tmp_path, monkeypatch):
    import features
    tr, te, og = _frame(120, 0), _frame(40, 1), _frame(200, 2)
    cat2 = tmp_path / "cat2.parquet"
    pd.DataFrame({"t_catx": np.linspace(-1, 1, 160).astype("float32")}).to_parquet(cat2)
    monkeypatch.setenv("TEACH", "3")
    trX, teX = features.teachers(tr, te, og, _labels(200, 3), cache=str(tmp_path / "missing.parquet"), cat2=str(cat2))
    assert "t_catx" in trX.columns and "t_catx" in teX.columns, list(trX.columns)
