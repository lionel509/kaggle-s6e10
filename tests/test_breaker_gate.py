"""Breaker probes for scripts/gate.py: malformed inputs must come back as MISSING (exit 2), not a crash."""
import json, subprocess, sys
from pathlib import Path
import pytest

GATE = Path(__file__).resolve().parents[1] / "scripts" / "gate.py"
FOLDS = [0.961414, 0.960357, 0.96174, 0.961437, 0.961399]


def _run(root, *args):
    return subprocess.run([sys.executable, str(GATE), *args, str(root)], capture_output=True, text=True)


def _recs(root):
    p = root / "logs" / "queue_i3_gates.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


@pytest.fixture
def root(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/base.json").write_text(json.dumps({"oof": 0.961266, "folds": FOLDS}))
    return tmp_path


def test_breaker_fold_json_without_oof_is_missing(root):
    (root / "logs/cand.json").write_text(json.dumps({"folds": [x + 1e-3 for x in FOLDS]}))
    r = _run(root, "cand", "base")
    assert r.returncode == 2, f"rc={r.returncode}, stderr tail: {r.stderr.strip().splitlines()[-1:]}"
    assert _recs(root)[-1]["missing"] is True


def test_breaker_fold_json_null_oof_is_missing(root):
    (root / "logs/cand.json").write_text(json.dumps({"oof": None, "folds": [x + 1e-3 for x in FOLDS]}))
    r = _run(root, "cand", "base")
    assert r.returncode == 2, f"rc={r.returncode}, stderr tail: {r.stderr.strip().splitlines()[-1:]}"


def test_breaker_blend_json_not_an_object_is_missing(root):
    (root / "logs/q3_f.json").write_text("[]")
    (root / "logs/q3_base.json").write_text(json.dumps({"oof": 0.96, "split_half": [0.96, 0.96]}))
    r = _run(root, "--blend", str(root / "logs/q3_f.json"), str(root / "logs/q3_base.json"))
    assert r.returncode == 2, f"rc={r.returncode}, stderr tail: {r.stderr.strip().splitlines()[-1:]}"
