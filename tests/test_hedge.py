"""blend/hedge.py: swaps refit hedge files into the blend's test columns; never touches OOF pool files."""
import sys
from pathlib import Path
import numpy as np, pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "blend")); sys.path.insert(0, str(REPO / "models"))


def _hedge():
    import hedge
    return hedge


def _setup(root):
    (root / "oof").mkdir(parents=True); (root / "subs" / "hedge").mkdir(parents=True)
    a, b, h = np.arange(5.0), np.arange(5.0) * 2, np.arange(5.0) * 10 + 1
    np.save(root / "oof" / "a_test.npy", a); np.save(root / "oof" / "b_test.npy", b)
    np.save(root / "subs" / "hedge" / "a_hedge_test.npy", h)
    return a, b, h


def test_hedge_file_replaces_test_column(tmp_path):
    a, b, h = _setup(tmp_path)
    T, w, swapped = _hedge().hedge_matrix({"own:a": 0.6, "own:b": 0.4}, root=str(tmp_path))
    assert T.shape == (5, 2)
    np.testing.assert_allclose(T[:, 0], h)
    np.testing.assert_allclose(T[:, 1], b)
    np.testing.assert_allclose(np.asarray(w), [0.6, 0.4])
    assert [s.removeprefix("own:") for s in swapped] == ["a"]


def test_without_hedge_files_test_columns_are_unchanged(tmp_path):
    a, b, h = _setup(tmp_path)
    (tmp_path / "subs" / "hedge" / "a_hedge_test.npy").unlink()
    T, w, swapped = _hedge().hedge_matrix({"own:a": 0.6, "own:b": 0.4}, root=str(tmp_path))
    np.testing.assert_allclose(T[:, 0], a)
    np.testing.assert_allclose(T[:, 1], b)
    assert len(swapped) == 0


def test_public_members_are_refused(tmp_path):
    _setup(tmp_path)
    with pytest.raises(ValueError):
        _hedge().hedge_matrix({"pub:x": 1.0}, root=str(tmp_path))
