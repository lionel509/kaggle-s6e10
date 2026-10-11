"""B: leak-free soft targets for the students (#3 B).

A student trained on outer rows `a` (validated on `b`, which lies inside one k5 fold j) needs a soft
target for every row in `a`. Using the blend OOF would leak: a row's OOF came from a model that saw
fold j's validation labels. Instead each row r in k5 fold k gets its target from a teacher trained
without folds {j, k}, so changing y[b] can never change the target. Imports numpy, sklearn and features.folds only.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from features import folds

PAIRS = ["01", "02", "03", "04", "12", "13", "14", "23", "24", "34"]
SINGLES = ["0", "1", "2", "3", "4"]


def fold_ids(y, k=5):
    """f[r] = index of the k fold whose validation set holds row r."""
    f = np.empty(len(y), dtype=int)
    for i, (_, b) in enumerate(folds(y, k)):
        f[b] = i
    return f


def nested_folds(y, K):
    """K outer splits (a, b, j); each validation set b lies inside one k5 fold j."""
    assert K % 5 == 0, "K must be a multiple of 5"
    n = len(y)
    if K == 5:
        return [(a, b, i) for i, (a, b) in enumerate(folds(y, 5))]
    f5 = fold_ids(y)
    out = []
    for j in range(5):
        Fj = np.where(f5 == j)[0]
        for _, v in StratifiedKFold(K // 5, shuffle=True, random_state=42).split(Fj, y[Fj]):
            b = Fj[v]
            a = np.setdiff1d(np.arange(n), b)
            out.append((a, b, j))
    return out


def rows_for(y, S):
    """S = string of fold digits. Returns (fit rows, predicted rows) of the teacher S, both ascending."""
    m = np.isin(fold_ids(y), [int(c) for c in S])
    return np.where(~m)[0], np.where(m)[0]


def teacher_path(fam, S, d="data/soft"):
    return f"{d}/{fam}_{''.join(sorted(S))}.npy"


def soft_target(y, rows, j, fams=("xgb", "mlp"), d="data/soft"):
    """Equal-weight mean over families of the leave-folds-out teacher predictions for `rows` (outer fold j)."""
    f5 = fold_ids(y)
    out = np.zeros(len(rows), dtype=np.float64)
    for fam in fams:
        full = np.full(len(y), np.nan)
        for k in np.unique(f5[rows]):
            S = "".join(sorted({str(j), str(k)}))
            _, b = rows_for(y, S)
            p = np.load(teacher_path(fam, S, d))
            assert len(p) == len(b)
            idx = np.where(f5 == k)[0]
            full[idx] = p[np.searchsorted(b, idx)]
        out += full[rows] / len(fams)
    assert not np.isnan(out).any()
    return out
