"""Guard probes for scripts/queue_i3.sh (issue #3 rework). The queue runs in a temp repo with a fake .venv python:
models/* log START/END to a spy file, anything else logs NONTRAIN; gate.py runs for real.
The queue decides from logs/run_A.done, logs/run_A.pid and logs/queue_i3.lock.d only. No process list is read.

Safety: every test signals only a PID or process group that this test started, or a group read from its own temp
logs/queue_i3.lock.d/step. Nothing here calls pgrep or ps to pick a target. Real training runs on this Mac."""
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
  *) echo "$(now) NONTRAIN $*" >> "$SPY"; exit "${BLEND_RC:-0}";;
esac
"""
# the queue's 300 s poll becomes 0.2 s; every other sleep is untouched
FAST_SLEEP = '#!/bin/sh\n[ "$1" = 300 ] && set -- 0.2\nexec /bin/sleep "$1"\n'
# pass-through: no argv matching anywhere, so a stray editor or grep never changes the queue's decision
PGREP_SHIM = '#!/bin/sh\ncase "$*" in *overnight*) exit 1;; esac\nexec /usr/bin/pgrep "$@"\n'
PIDFILES = ("missing", "", "abc", "0", "dead", "live")
CASES = [(done, pf) for done in (True, False) for pf in PIDFILES]
FOREIGN = ["less models/v4_catboost.py", "vim models/v5_xgb.py", "grep -n python models/v4_catboost.py",
           "python -m pydoc models/v4_catboost.py", "less logs/run_A.sh"]


def _exec(p, text):
    p.write_text(text); p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _make_rig(root):
    R = root / "repo"
    for d in ("scripts", "blend", "logs", "subs", "models", ".venv/bin", "shim", "data/soft"):
        (R / d).mkdir(parents=True, exist_ok=True)
    for f in ("queue_i3.sh", "gate.py", "report_i3.py"):
        shutil.copy(REPO / "scripts" / f, R / "scripts" / f)
    _exec(R / ".venv/bin/python", STUB)
    _exec(R / "shim/sleep", FAST_SLEEP)
    _exec(R / "shim/pgrep", PGREP_SHIM)
    spy = root / "spy.txt"; spy.write_text("")
    env = {**os.environ, "SPY": str(spy), "REAL_PY": REAL_PY, "PATH": f"{R/'shim'}:{os.environ['PATH']}"}
    return R, env, spy


@pytest.fixture
def rig(tmp_path):
    R, env, spy = _make_rig(tmp_path)
    (R / "logs/run_A.done").write_text("")
    return R, env, spy


def _launch(R, env):
    return subprocess.Popen(["zsh", "scripts/queue_i3.sh"], cwd=R, env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _live():
    """A plain live process in its own session (pgid == pid)."""
    return subprocess.Popen(["/bin/sleep", "60"], start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _foreign(R, argv):
    """A live process whose argv is exactly argv (exec -a), cwd = temp repo, in its own session."""
    return subprocess.Popen(["zsh", "-c", f'exec -a "{argv}" /bin/sleep 60'], cwd=R, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _dead_pid():
    p = subprocess.Popen(["/usr/bin/true"]); p.wait()
    return p.pid


def _kill_pgid(g):
    if g is None: return
    try: os.killpg(g, signal.SIGKILL)
    except (ProcessLookupError, PermissionError): pass


def _kill_proc(p):
    _kill_pgid(p.pid)
    p.wait()


def _group_alive(g):
    try:
        os.killpg(g, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _step_group(R):
    try: return int((R / "logs/queue_i3.lock.d/step").read_text().strip())
    except (FileNotFoundError, ValueError): return None


def _starts(spy):
    return [l for l in spy.read_text().splitlines() if " START " in l]


def _until(pred, secs):
    t0 = time.time()
    while time.time() - t0 < secs:
        if pred(): return True
        time.sleep(0.1)
    return pred()


def _log(R):
    return (R / "logs/queue_i3.log").read_text()


def test_guard_trains_iff_A_is_finished(tmp_path):
    """Generated check of the invariant: a queue starts training iff logs/run_A.done exists and the PID in
    logs/run_A.pid is not a live process. 12 cases (done x pidfile) run at once, each in its own rig.
    Foreign processes whose argv names models/*.py or run_A.sh must not hold the queue."""
    rigs = []
    for i, (done, pf) in enumerate(CASES):
        R, env, spy = _make_rig(tmp_path / f"c{i}")
        if done: (R / "logs/run_A.done").write_text("")
        rigs.append((done, pf, R, env, spy))
    live, dead = _live(), _dead_pid()
    foreign, queues = [], []
    try:
        foreign = [_foreign(rigs[0][2], a) for a in FOREIGN]
        assert all(f.poll() is None for f in foreign), "precondition: foreign stub not alive"
        for done, pf, R, env, spy in rigs:
            if pf != "missing":
                (R / "logs/run_A.pid").write_text({"dead": str(dead), "live": str(live.pid)}.get(pf, pf))
            queues.append(_launch(R, env))
        time.sleep(4)
        for (done, pf, R, env, spy), q in zip(rigs, queues):
            expect = done and pf != "live"
            starts = _starts(spy)
            assert bool(starts) == expect, f"done={done} pidfile={pf!r}: expected START={expect}, got {starts[:1]}"
            if not expect:
                assert "waiting for A" in _log(R), f"done={done} pidfile={pf!r}: queue did not log the wait"
    finally:
        for q in queues: _kill_proc(q)
        for _, _, R, _, _ in rigs: _kill_pgid(_step_group(R))
        for s in [live, *foreign]: _kill_proc(s)


def test_guard_lock_states(tmp_path):
    """Six lock states, each in its own rig with run_A.done present and no run_A.pid. START iff the lock is absent,
    stale (dead PID and no live step group) or unreadable. A live queue PID or a live step group holds it: exit 1."""
    live, orphan, dead = _live(), _live(), _dead_pid()  # orphan stands in for an orphaned step group
    cases = [
        ("a absent", None, True),
        ("b live queue", {"pid": live.pid}, False),
        ("c dead pid", {"pid": dead}, True),
        ("d dead pid + live step", {"pid": dead, "step": orphan.pid}, False),
        ("e empty dir", {}, True),
        ("f pid abc", {"pid": "abc"}, True),
    ]
    rigs = []
    try:
        for i, (name, lock, expect) in enumerate(cases):
            R, env, spy = _make_rig(tmp_path / f"l{i}")
            (R / "logs/run_A.done").write_text("")
            if lock is not None:
                d = R / "logs/queue_i3.lock.d"; d.mkdir(parents=True)
                for k, v in lock.items(): (d / k).write_text(str(v))
            rigs.append((name, expect, R, spy, _launch(R, env), lock))
        time.sleep(5)
        for name, expect, R, spy, q, lock in rigs:
            starts = _starts(spy)
            if expect:
                assert starts, f"{name}: no START within 5 s (log: {_log(R)[-200:]})"
            else:
                assert starts == [], f"{name}: trained while the lock was held: {starts[:1]}"
                assert q.poll() == 1, f"{name}: queue did not exit by itself (rc {q.poll()})"
                assert "another queue holds" in _log(R), f"{name}: no 'another queue holds' line"
            if name.startswith("b "):
                assert (R / "logs/queue_i3.lock.d/pid").read_text().strip() == str(live.pid), f"{name}: lock pid changed"
            if name.startswith("d "):
                assert (R / "logs/queue_i3.lock.d/step").read_text().strip() == str(orphan.pid), f"{name}: step changed"
            if name.startswith("c "):
                assert "stale lock" in _log(R), f"{name}: no 'stale lock' line"
    finally:
        for _, _, R, _, q, _ in rigs:
            _kill_proc(q)
            _kill_pgid(_step_group(R))
        for s in (live, orphan): _kill_proc(s)


@pytest.mark.parametrize("sig,name,rc", [(signal.SIGTERM, "TERM", 143), (signal.SIGINT, "INT", 130)])
def test_guard_signal_kills_step_group(tmp_path, sig, name, rc):
    """A TERM or INT to the queue PID kills that step's process group, releases the lock and exits with 128+signal.
    Two bystanders in their own sessions must survive."""
    R, env, spy = _make_rig(tmp_path)
    (R / "logs/run_A.done").write_text("")
    env = {**env, "TRAIN_SECS": "30"}
    bystanders = [_live(), _foreign(R, "python models/v4_catboost.py")]
    p, g = None, None
    try:
        assert all(b.poll() is None for b in bystanders), "precondition: bystander not alive"
        p = _launch(R, env)
        assert _until(lambda: bool(_starts(spy)) and _step_group(R) is not None, 10), "no step started"
        g = _step_group(R)
        assert _group_alive(g), "precondition: step group not alive"
        os.kill(p.pid, sig)  # the queue PID only, as a plain `kill <pid>` would
        assert p.wait(timeout=10) == rc, f"queue exit code after SIG{name}"
        assert _until(lambda: not _group_alive(g), 5), "step group survived the signal"
        assert not (R / "logs/queue_i3.lock.d").exists(), "lock not released"
        assert f"signal {name}" in _log(R), "no 'signal' line in the log"
        assert " END " not in spy.read_text(), "the interrupted step reached its END line"
        assert all(b.poll() is None for b in bystanders), "a bystander was killed"
    finally:
        if p: _kill_proc(p)
        _kill_pgid(g)
        for b in bystanders: _kill_proc(b)


def test_guard_normal_exit_releases_lock(tmp_path):
    """A queue that finishes normally ends with 'queue done' and leaves no lock behind."""
    R, env, spy = _make_rig(tmp_path)
    (R / "logs/run_A.done").write_text("")
    p = _launch(R, env)
    try:
        assert p.wait(timeout=120) == 0
    finally:
        _kill_proc(p)
    assert _log(R).splitlines()[-1].endswith("queue done"), _log(R)[-300:]
    assert not (R / "logs/queue_i3.lock.d").exists(), "lock left behind after a normal exit"


def test_guard_stop_while_waiting_for_live_A(tmp_path):
    """A stop file while run_A.pid names a live process ends the queue before any training, blend or report."""
    R, env, spy = _make_rig(tmp_path)
    (R / "logs/run_A.done").write_text("")
    a = _live()
    (R / "logs/run_A.pid").write_text(str(a.pid))
    (R / "logs/queue_i3.stop").write_text("")
    p = None
    try:
        p = _launch(R, env)
        assert p.wait(timeout=10) == 0, "queue did not exit on the stop file"
        assert spy.read_text() == "", f"queue ran after the stop file: {spy.read_text()[:300]}"
        assert "stop file before A finished: exit" in _log(R), "no stop line in the log"
        assert not (R / "logs/queue_i3.lock.d").exists(), "lock left behind"
    finally:
        if p: _kill_proc(p)
        _kill_proc(a)
