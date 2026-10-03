"""Pins blend/stack.py nesting: a row's stacked prediction comes only from a model fit without its own fold."""
import sys, numpy as np
sys.path.insert(0, "blend")
from stack import nested_oof


class Spy:  # "predicts" the set of folds it was trained on
    def fit(self, X, y): self.seen = frozenset(X[:, 0].astype(int)); return self
    def predict_proba(self, X): return np.c_[np.zeros(len(X)), [hash(self.seen) for _ in X]]


def test_no_row_predicted_by_model_fit_on_its_own_fold():
    fold = np.arange(40) % 4; X = fold[:, None].astype(float); y = np.arange(40) % 2
    p = nested_oof(X, y, fold, Spy)
    for f in range(4):
        assert (p[fold == f] == hash(frozenset(set(range(4)) - {f}))).all()
