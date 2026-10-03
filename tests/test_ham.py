"""Pins models/features.ham_core: nearest-original Hamming distance and the original label rate among neighbours."""
import sys, numpy as np
sys.path.insert(0, "models")
from features import ham_core


def test_ham_min_and_rates():
    O = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 4], [5, 5, 5]]); oy = np.array([1, 0, 1, 1])
    Q = np.array([[1, 2, 3], [1, 2, 5], [5, 5, 1]])
    f = ham_core(Q, O, oy)
    assert f["ham_min"].tolist() == [0, 1, 1]
    assert np.allclose(f["ham_rate_min"], [0.5, 2 / 3, 1.0]) and f["ham_n_min"].tolist() == [2, 3, 1]
    assert np.allclose(f["ham_rate_1"], [2 / 3, 2 / 3, 1.0]) and f["ham_n_1"].tolist() == [3, 3, 1]
    assert f["ham_n_2"].tolist() == [3, 4, 1]  # [1,2,5] is 2 from [5,5,5]
