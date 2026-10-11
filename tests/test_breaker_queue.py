"""Breaker probes for scripts/queue_i3.sh. Runs the real queue in a temp repo with a fake .venv python.
gate.py runs for real; models/* are fake trainings that log START/END to a spy file; blend/* exit BLEND_RC."""
import os, shutil, signal, stat, subprocess, time
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
REAL_PY = "/Users/lionelweng/Documents/Citadel/Active/Kaggle S6E10/.venv/bin/python"
STUB = """#!/bin/sh
now() { perl -MTime::HiRes=time -e 'printf "%.3f", time'; }
case "$1" in
  scripts/gate.py) exec "$REAL_PY" "$@";;
  models/*)
    echo "$(now) START K=${K:-5} TEACH=${TEACH:-unset} $*" >> "$SPY"; sleep "${TRAIN_SECS:-0}"
    echo "$(now) END K=${K:-5} $*" >> "$SPY"; exit 0;;
  *) echo "$(now) NONTRAIN $*" >> "$SPY"; exit "${BLEND_RC:-0}";;
esac
"""
PGREP_SHIM = """#!/bin/sh
case "$*" in *overnight*) exit 1;; esac
for p in $(/usr/bin/pgrep -f "$2"); do [ "$(lsof -a -p $p -d cwd -Fn 2>/dev/null | sed -n s/^n//p)" = "$(pwd -P)" ] && exit 0; done
exit 1
"""


def _exec(p, text):
    p.write_text(text); p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def rig(tmp_path):
    R = tmp_path / "repo"
    for d in ("scripts", "blend", "logs", "subs", ".venv/bin", "shim"):
        (R / d).mkdir(parents=True, exist_ok=True)
    for f in ("queue_i3.sh", "gate.py", "report_i3.py"):
        shutil.copy(REPO / "scripts" / f, R / "scripts" / f)
    _exec(R / ".venv/bin/python", STUB)
    _exec(R / "shim/sleep", '#!/bin/sh\n[ "$1" = 300 ] && set -- 1\nexec /bin/sleep "$1"\n')  # the queue's 300s poll becomes 1s
    # sees only processes whose cwd is this temp repo: no real Applied Pipeline render or real A may hold the queue
    _exec(R / "shim/pgrep", PGREP_SHIM)
    spy = tmp_path / "spy.txt"; spy.write_text("")
    env = {**os.environ, "SPY": str(spy), "REAL_PY": REAL_PY, "PATH": f"{R/'shim'}:{os.environ['PATH']}"}
    return R, env, spy


def _run(R, env, secs, n=1):
    procs = [subprocess.Popen(["zsh", "scripts/queue_i3.sh"], cwd=R, env=env, start_new_session=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(n)]
    time.sleep(secs)
    for p in procs:
        try: os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError): pass  # macOS: EPERM once the group leader has exited
        p.wait()


def _starts(spy):
    return [l for l in spy.read_text().splitlines() if " START " in l]


def test_breaker_queue_waits_while_catboost_a_still_runs(rig):
    """A's CatBoost k20 (models/v4_catboost.py) may still be running when the RealMLP k20 JSON lands.
    The queue has no pgrep guard for it and nothing writes logs/run_A.done, so it should not start yet."""
    R, env, spy = rig
    (R / "logs/v3b_realmlp_e6_n8_s42_k20_aux_teach.json").write_text("{}")
    cat = subprocess.Popen(["perl", "-e", "sleep 60", "models/v4_catboost.py"], cwd=R)
    try:
        _run(R, env, secs=5)
    finally:
        cat.kill(); cat.wait()
    assert _starts(spy) == [], f"queue trained while A's catboost was alive: {_starts(spy)[:2]}"


def test_queue_stop_before_A_runs_nothing(rig):
    """A stop file written while A is still running must end the queue before any blend, hedge, report or training."""
    R, env, spy = rig
    (R / "logs/queue_i3.stop").write_text("")
    _run(R, env, secs=5)
    assert spy.read_text() == "", f"queue ran after a stop file before A finished: {spy.read_text()[:300]}"


def test_breaker_second_queue_instance_overlaps_training(rig):
    """Relaunching the queue (it is documented as re-runnable) while the first is alive has no lock:
    two fake trainings should never be in flight at once, but they overlap."""
    R, env, spy = rig
    (R / "logs/run_A.done").write_text("")
    env = {**env, "TRAIN_SECS": "1.5"}
    _run(R, env, secs=6, n=2)
    depth = peak = 0
    for l in sorted(spy.read_text().splitlines(), key=lambda l: (float(l.split()[0]), l.split()[1] == "START")):
        if " START " in l: depth += 1; peak = max(peak, depth)
        elif " END " in l: depth -= 1
    assert peak <= 1, f"{peak} trainings in flight at once"


def test_breaker_stale_blend_json_promotes_k20_tabm(rig):
    """A failed q3_f/q3_base blend leaves the previous run's JSON in logs/. The F gate reads it and
    promotes TabM to K=20 on stale numbers."""
    R, env, spy = rig
    (R / "logs/run_A.done").write_text("")
    (R / "logs/v8_tabm_s42_aux_teach.json").write_text("{}")  # F k5 already done
    (R / "logs/q3_base.json").write_text('{"oof": 0.96, "split_half": [0.96, 0.96], "weights": {}}')
    (R / "logs/q3_f.json").write_text('{"oof": 0.97, "split_half": [0.97, 0.97], "weights": {}}')
    env = {**env, "BLEND_RC": "1"}  # this run's blends crash, so the JSON is not rewritten
    _run(R, env, secs=10)
    k20 = [l for l in _starts(spy) if "K=20" in l and "models/v8_tabm.py" in l]
    assert k20 == [], f"K=20 TabM started on a gate read from stale blend JSON: {k20}"


def test_breaker_exported_env_does_not_leak_into_queue_steps(rig):
    """An exported TEACH=3 or K=20 in the launching shell reaches every step: the k5 runs turn into K=20 runs,
    and TabM (v8, named *_aux_teach) gets TEACH=3 under a fixed name."""
    R, env, spy = rig
    (R / "logs/run_A.done").write_text("")
    env = {**env, "TEACH": "3", "K": "20"}
    _run(R, env, secs=10)
    v5 = [l for l in _starts(spy) if "models/v5_xgb.py" in l]
    v8 = [l for l in _starts(spy) if "models/v8_tabm.py" in l]
    assert v5 and v8, f"queue did not reach the steps under test: {_starts(spy)[:3]}"
    assert " K=5 " in v5[0], f"first k5 step ran with the exported K: {v5[0]}"
    bad = [l for l in v8 if "TEACH=unset" not in l and "TEACH=1" not in l]
    assert bad == [], f"TabM ran with an exported TEACH: {bad[:2]}"
