"""Round-2 breaker probes for scripts/queue_i3.sh at cc166ac. The real queue runs in a temp repo with a fake
.venv python (models/* log START/END to a spy file, anything else logs NONTRAIN). gate.py runs for real.
sleep 300 becomes 1 s through a shim. Nothing touches the real logs/, data/ or training."""
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
    echo "$(now) START K=${K:-5} $*" >> "$SPY"; sleep "${TRAIN_SECS:-0}"
    echo "$(now) END K=${K:-5} $*" >> "$SPY"; exit 0;;
  *) echo "$(now) NONTRAIN $*" >> "$SPY"; exit 0;;
esac
"""
FAST_SLEEP = '#!/bin/sh\n[ "$1" = 300 ] && set -- 1\nexec /bin/sleep "$1"\n'


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
    _exec(R / "shim/sleep", FAST_SLEEP)
    _exec(R / "shim/pgrep", "#!/bin/sh\nexit 1\n")
    spy = tmp_path / "spy.txt"; spy.write_text("")
    env = {**os.environ, "SPY": str(spy), "REAL_PY": REAL_PY, "PATH": f"{R/'shim'}:{os.environ['PATH']}"}
    return R, env, spy


def _launch(R, env):
    return subprocess.Popen(["zsh", "scripts/queue_i3.sh"], cwd=R, env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _kill_group(p):
    try: os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError): pass
    p.wait()


def _starts(spy):
    return [l for l in spy.read_text().splitlines() if " START " in l]


def test_r2_stop_touched_during_poll_then_A_done_still_blends(rig):
    """Stop file is written while A runs; A finishes within the same 300 s poll. Plan: 'stop file before A
    finishes exits with no blend, hedge or report'. The until-loop checks run_A.done first, so the stop file is never seen."""
    R, env, spy = rig
    _exec(R / "shim/sleep", '#!/bin/sh\n[ "$1" = 300 ] && { touch logs/queue_i3.stop logs/run_A.done; set -- 0; }\nexec /bin/sleep "$1"\n')
    p = _launch(R, env)
    try:
        p.wait(timeout=30)
    finally:
        _kill_group(p)
    nontrain = [l for l in spy.read_text().splitlines() if "NONTRAIN" in l]
    assert nontrain == [], f"blend/report ran after a stop file written before A finished: {nontrain[:3]}"


def test_r2_crashed_A_without_done_waits_silently_forever(rig):
    """A dies without writing run_A.done. The queue sits in the wait loop: no heartbeat, no exit, nothing logged.
    Expected: the wait is visible in logs/queue_i3.log (or the queue exits) within 6 s."""
    R, env, spy = rig
    p = _launch(R, env)
    time.sleep(6)
    alive = p.poll() is None
    _kill_group(p)
    lines = (R / "logs/queue_i3.log").read_text().splitlines()
    assert (not alive) or len(lines) > 1, f"queue alive and silent after 6 s: log={lines}"


def test_r2_stale_run_A_done_starts_while_A_is_relaunched(rig):
    """run_A.done left over from an earlier A run, A relaunched and still running (models/v4_catboost.py alive).
    The queue's only guard is that file, so it starts training while A is still running."""
    R, env, spy = rig
    (R / "logs/run_A.done").write_text("")  # stale: from the earlier A pass
    cat = subprocess.Popen(["perl", "-e", "sleep 60", "models/v4_catboost.py"], cwd=R)
    p = _launch(R, env)
    try:
        time.sleep(5)
    finally:
        _kill_group(p); cat.kill(); cat.wait()
    assert _starts(spy) == [], f"queue trained while A (v4_catboost) was alive: {_starts(spy)[:2]}"


def test_r2_killed_queue_shell_lets_second_queue_train_over_orphan(rig):
    """Kill only the queue shell (SIGKILL, not its group). Its training child lives on, but the flock is
    released with the shell (the child does not inherit it). A relaunched queue takes the lock and starts
    a second training while the first is still running."""
    R, env, spy = rig
    (R / "logs/run_A.done").write_text("")
    env = {**env, "TRAIN_SECS": "8"}
    p1 = _launch(R, env)
    try:
        t0 = time.time()
        while not _starts(spy) and time.time() - t0 < 10: time.sleep(0.2)
        assert _starts(spy), "first queue never started a training"
        os.kill(p1.pid, signal.SIGKILL); p1.wait()
        p2 = _launch(R, env)
        time.sleep(4)
        _kill_group(p2)
    finally:
        _kill_group(p1)  # reaps the orphaned training child too
    lines = spy.read_text().splitlines()
    first_end = next((i for i, l in enumerate(lines) if " END " in l), len(lines))
    starts_before_end = [l for l in lines[:first_end] if " START " in l]
    assert len(starts_before_end) <= 1, f"{len(starts_before_end)} trainings started before the first one ended: {starts_before_end}"


GATE = REPO / "scripts" / "gate.py"


@pytest.mark.parametrize("bad_oof", ["Infinity", "1.5"])
def test_r2_blend_gate_promotes_impossible_auc(tmp_path, bad_oof):
    """An AUC outside [0, 1] (corrupt or hand-edited blend JSON) is not a measurement. With both halves up
    the gate PROMOTEs it. Expected: MISSING (exit 2), like the other malformed inputs."""
    cand, base = tmp_path / "c.json", tmp_path / "b.json"
    cand.write_text(f'{{"oof": {bad_oof}, "split_half": [0.97, 0.97]}}')
    base.write_text('{"oof": 0.96, "split_half": [0.96, 0.96]}')
    r = subprocess.run([REAL_PY, str(GATE), "--blend", str(cand), str(base), str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 2, f"rc={r.returncode}: {r.stdout.strip()}"
