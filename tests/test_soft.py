"""B: leak-free soft targets (models/soft.py). The invariant: changing y on a student's validation rows can't change its soft targets."""
import sys
from pathlib import Path
import numpy as np, pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models"))
from features import folds  # noqa: E402


def _soft():
    import soft
    return soft


def _labels(seed, n=2000):
    r = np.random.default_rng(seed)
    return (r.random(n) < 0.43).astype(int)


def _fake_teachers(s, y_fold, y_lab, d):
    """Writes every teacher file; the prediction is the mean label of its fit rows (labels from y_lab, folds from y_fold)."""
    for fam in ("xgb", "mlp"):
        for S in s.PAIRS + s.SINGLES:
            a, b = s.rows_for(y_fold, S)
            np.save(s.teacher_path(fam, S, d), np.full(len(b), y_lab[a].mean(), dtype="float32"))


def test_nested_folds_k5_matches_folds():
    s = _soft(); y = _labels(0)
    got = s.nested_folds(y, 5)
    want = [(a, b, i) for i, (a, b) in enumerate(folds(y, 5))]
    assert len(got) == len(want) == 5
    for (a, b, j), (a2, b2, j2) in zip(got, want):
        assert j == j2 and np.array_equal(a, a2) and np.array_equal(b, b2)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nested_folds_k20_partition(seed):
    s = _soft(); y = _labels(seed); n = len(y)
    splits = s.nested_folds(y, 20)
    assert len(splits) == 20
    f5 = s.fold_ids(y)
    seen = np.concatenate([b for _, b, _ in splits])
    assert np.array_equal(np.sort(seen), np.arange(n))  # every row validates exactly once
    for a, b, j in splits:
        assert set(f5[b]) == {j}  # each validation set lies inside one k5 fold
        assert np.array_equal(a, np.setdiff1d(np.arange(n), b))


def test_soft_target_alignment(tmp_path):
    s = _soft(); y = _labels(3)
    d = str(tmp_path)
    for S in s.PAIRS + s.SINGLES:  # teacher "predicts" the row index itself
        _, b = s.rows_for(y, S)
        np.save(s.teacher_path("xgb", S, d), b.astype("float64"))
    for K in (5, 20):
        for a, b, j in s.nested_folds(y, K):
            out = s.soft_target(y, a, j, fams=("xgb",), d=d)
            np.testing.assert_array_equal(out, a.astype("float64"))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("K", [5, 20])
def test_soft_target_ignores_val_labels(tmp_path, seed, K):
    s = _soft(); y = _labels(seed)
    d0, d1 = tmp_path / "d0", tmp_path / "d1"
    d0.mkdir(); d1.mkdir()
    _fake_teachers(s, y, y, str(d0))
    # teachers are built from y2 (folds still from y), so only the outer validation labels differ
    for a, b, j in s.nested_folds(y, K):
        y2 = y.copy(); y2[b] = 1 - y2[b]
        _fake_teachers(s, y, y2, str(d1))
        np.testing.assert_array_equal(s.soft_target(y, a, j, d=str(d0)), s.soft_target(y, a, j, d=str(d1)))


def test_detector_catches_plain_oof():
    """Sanity for the invariant: a plain OOF target (row r from the teacher of its own fold) does depend on y[b]."""
    s = _soft(); y = _labels(0); f5 = s.fold_ids(y)
    a, b, j = s.nested_folds(y, 5)[0]
    y2 = y.copy(); y2[b] = 1 - y2[b]
    leaky = lambda lab: np.array([lab[f5 != f5[r]].mean() for r in a])
    assert not np.array_equal(leaky(y), leaky(y2))


def test_missing_teacher_raises(tmp_path):
    s = _soft(); y = _labels(0); a, b, j = s.nested_folds(y, 5)[0]
    with pytest.raises(FileNotFoundError):
        s.soft_target(y, a, j, d=str(tmp_path))
