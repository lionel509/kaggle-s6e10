"""scripts/gate.py: the per-idea promotion gate. Run as the CLI the queue uses: gate.py <cand> <base> <root>."""
import json, subprocess, sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "gate.py"
BASE_FOLDS = [0.961414, 0.960357, 0.96174, 0.961437, 0.961399]  # real k5 baseline folds
BASE_OOF = 0.961266


def _write(root, name, oof, folds):
    (root / "logs").mkdir(exist_ok=True)
    json.dump({"name": name, "oof": oof, "folds": folds}, open(root / "logs" / f"{name}.json", "w"))


def _gate(root, cand="cand", base="base"):
    assert GATE.exists(), "scripts/gate.py not written yet"
    return subprocess.run([sys.executable, str(GATE), cand, base, str(root)], capture_output=True, text=True)


def _records(root):
    p = root / "logs" / "queue_i3_gates.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


@pytest.fixture
def root(tmp_path):
    _write(tmp_path, "base", BASE_OOF, BASE_FOLDS)
    return tmp_path


def test_exactly_threshold_with_four_up_promotes(root):
    folds = [f + 1e-5 for f in BASE_FOLDS[:4]] + [BASE_FOLDS[4] - 1e-5]
    _write(root, "cand", BASE_OOF + 0.00003, folds)
    assert _gate(root).returncode == 0
    assert _records(root)[-1]["pass"] is True


def test_just_below_threshold_fails(root):
    folds = [f + 1e-5 for f in BASE_FOLDS]
    _write(root, "cand", BASE_OOF + 0.000029, folds)
    assert _gate(root).returncode == 1
    assert _records(root)[-1]["pass"] is False


def test_three_up_fails_even_with_big_gain(root):
    folds = [f + 1e-5 for f in BASE_FOLDS[:3]] + [BASE_FOLDS[3] - 1e-5, BASE_FOLDS[4] - 1e-5]
    _write(root, "cand", BASE_OOF + 0.0001, folds)
    assert _gate(root).returncode == 1


def test_tie_does_not_count_as_up(root):
    folds = [f + 1e-5 for f in BASE_FOLDS[:3]] + [BASE_FOLDS[3], BASE_FOLDS[4] - 1e-5]
    _write(root, "cand", BASE_OOF + 0.0001, folds)
    assert _gate(root).returncode == 1


def test_missing_cand_returns_2_and_logs_missing(root):
    r = _gate(root, cand="nope")
    assert r.returncode == 2
    rec = _records(root)[-1]
    assert rec["missing"] is True and rec["cand"] == "nope"


def test_pass_appends_one_record(root):
    folds = [f + 1e-5 for f in BASE_FOLDS]
    _write(root, "cand", BASE_OOF + 0.0001, folds)
    before = len(_records(root))
    assert _gate(root).returncode == 0
    recs = _records(root)
    assert len(recs) == before + 1
    rec = recs[-1]
    assert rec["up"] == 5 and rec["pass"] is True and rec["missing"] is False
    assert abs(rec["delta"] - 0.0001) < 1e-9


def _write_blend(root, name, oof, halves):
    (root / "logs").mkdir(exist_ok=True)
    json.dump({"oof": oof, "weights": {}, "split_half": halves}, open(root / "logs" / f"{name}.json", "w"))


def _gate_blend(root, cand, base):
    assert GATE.exists(), "scripts/gate.py not written yet"
    args = [sys.executable, str(GATE), "--blend", str(root / "logs" / f"{cand}.json"),
            str(root / "logs" / f"{base}.json"), str(root)]
    return subprocess.run(args, capture_output=True, text=True)


def test_blend_higher_oof_but_one_worse_half_fails(tmp_path):
    _write_blend(tmp_path, "bcand", 0.961800, [0.961300, 0.962300])
    _write_blend(tmp_path, "bbase", 0.961700, [0.961400, 0.962200])
    assert _gate_blend(tmp_path, "bcand", "bbase").returncode == 1
    rec = _records(tmp_path)[-1]
    assert rec["up"] == 1 and rec["pass"] is False and rec["missing"] is False


def test_blend_gain_in_both_halves_passes(tmp_path):
    _write_blend(tmp_path, "bcand", 0.961800, [0.961500, 0.962300])
    _write_blend(tmp_path, "bbase", 0.961700, [0.961400, 0.962200])
    assert _gate_blend(tmp_path, "bcand", "bbase").returncode == 0
    rec = _records(tmp_path)[-1]
    assert rec["up"] == 2 and rec["pass"] is True
