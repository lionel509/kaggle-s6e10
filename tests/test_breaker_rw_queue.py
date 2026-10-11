"""Breaker probes for scripts/queue_i3.sh at e455ee5 (PR #4, rework of #3 guard).

Each test runs the real queue in a temp repo. The python in .venv is a stub: models/* log START/END to a spy file and
sleep TRAIN_SECS; other calls log NONTRAIN. scripts/gate.py runs for real. The pgrep shim is the real pgrep, filtered
to PIDs listed in allow.txt, so the real overnight renders on this Mac never count.

Safety: every signal goes to a process this file started, or to a process group read from the temp repo's own
logs/queue_i3.lock.d/step. Nothing calls pgrep or ps to pick a target. Real training runs on this Mac.
"""
import os, shutil, signal, stat, subprocess, time
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
REAL_PY = "/Users/lionelweng/Documents/Citadel/Active/Kaggle S6E10/.venv/bin/python"
PAIRS = ["01", "02", "03", "04", "12", "13", "14", "23", "24", "34"]

STUB = r"""#!/bin/sh
now() { perl -MTime::HiRes=time -e 'printf "%.3f", time'; }
case "$1" in
  scripts/gate.py) exec "$REAL_PY" "$@";;
  models/*)
    [ -n "$STUBBORN" ] && trap '' TERM
    if [ -n "$GRANDKID" ]; then sh -c 'trap "" TERM; exec sleep 60' & echo $! > "$KIDPID"; fi
    echo "$(now) START $*" >> "$SPY"
    sleep "${TRAIN_SECS:-0}"
    echo "$(now) END $*" >> "$SPY"
    exit "${STEP_RC:-0}";;
  *) echo "$(now) NONTRAIN $*" >> "$SPY"; exit 0;;
esac
"""
# the queue's 300 s poll becomes WAIT_SECS (default 0.2); every other sleep is real
SLEEP_SHIM = '#!/bin/sh\n[ "$1" = 300 ] && set -- "${WAIT_SECS:-0.2}"\nexec /bin/sleep "$1"\n'
# real pgrep, but only PIDs listed in $ALLOW count, so the real overnight renders on this Mac are invisible
PGREP_SHIM = ('#!/bin/sh\nhit=1\nfor p in $(/usr/bin/pgrep "$@"); do\n'
              '  grep -qx "$p" "$ALLOW" 2>/dev/null && { echo "$p"; hit=0; }\ndone\nexit $hit\n')
# real mkdir; with STALL_LOCK set, the first successful mkdir of a lock.d path stalls before returning
MKDIR_SHIM = ('#!/bin/sh\n/bin/mkdir "$@"; rc=$?\n'
              'if [ $rc = 0 ] && [ -n "$STALL_LOCK" ] && [ ! -e "$STALL_MARK" ]; then\n'
              '  case "$*" in *lock.d) : > "$STALL_MARK"; sleep "$STALL_LOCK";; esac\nfi\nexit $rc\n')


def _exec(p, text):
    p.write_text(text)
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _rig(root, **extra):
    R = root / "repo"
    for d in ("scripts", "blend", "logs", "subs", "models", ".venv/bin", "shim", "data/soft"):
        (R / d).mkdir(parents=True, exist_ok=True)
    for f in ("queue_i3.sh", "gate.py", "report_i3.py"):
        shutil.copy(REPO / "scripts" / f, R / "scripts" / f)
    _exec(R / ".venv/bin/python", STUB)
    _exec(R / "shim/sleep", SLEEP_SHIM)
    _exec(R / "shim/pgrep", PGREP_SHIM)
    _exec(R / "shim/mkdir", MKDIR_SHIM)
    spy = root / "spy.txt"; spy.write_text("")
    allow = root / "allow.txt"; allow.write_text("")
    env = {**os.environ, "SPY": str(spy), "REAL_PY": REAL_PY, "ALLOW": str(allow),
           "KIDPID": str(root / "kid.pid"), "PATH": f"{R/'shim'}:{os.environ['PATH']}", **extra}
    return R, env, spy, allow


def _launch(R, env):
    return subprocess.Popen(["zsh", "scripts/queue_i3.sh"], cwd=R, env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _live():
    return subprocess.Popen(["/bin/sleep", "60"], start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _foreign(R, argv):
    return subprocess.Popen(["zsh", "-c", f'exec -a "{argv}" /bin/sleep 60'], cwd=R, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _dead_pid():
    p = subprocess.Popen(["/usr/bin/true"]); p.wait()
    return p.pid


def _kill_proc(p):
    if p is None:
        return
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _kill_group(g):
    if g:
        try:
            os.killpg(g, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _group_alive(g):
    try:
        os.killpg(g, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _starts(spy):
    return [l for l in spy.read_text().splitlines() if " START " in l]


def _max_concurrent(spy):
    cur = best = 0
    for l in spy.read_text().splitlines():
        if " START " in l:
            cur += 1; best = max(best, cur)
        elif " END " in l:
            cur -= 1
    return best


def _log(R):
    p = R / "logs/queue_i3.log"
    return p.read_text() if p.exists() else ""


def _until(pred, secs):
    end = time.monotonic() + secs
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.1)
    return bool(pred())


def _group(R):
    try:
        return int((R / "logs/queue_i3.lock.d/step").read_text().strip())
    except (OSError, ValueError):
        return None


def _wait_exit(p, secs):
    try:
        p.wait(timeout=secs)
        return True
    except subprocess.TimeoutExpired:
        return False


# 1. A foreign editor or grep whose argv names scripts/overnight.py stalls the queue in wait_render.
def test_foreign_editor_argv_does_not_stall_queue(tmp_path):
    """Control (no foreign process) and foreign (`less scripts/overnight.py` open). The goal in the rework plan is
    that open editors and greps can no longer stall the queue; the foreign case must reach teach mlp as well."""
    results = {}
    for case in ("control", "foreign"):
        R, env, spy, allow = _rig(tmp_path / case)
        (R / "logs/run_A.done").write_text("")
        for S in PAIRS:
            (R / f"data/soft/xgb_{S}.npy").write_bytes(b"")  # teach xgb is already done, so the queue reaches teach mlp
        f = None
        if case == "foreign":
            f = _foreign(R, "less scripts/overnight.py")
            allow.write_text(f"{f.pid}\n")
        q = _launch(R, env)
        try:
            results[case] = _until(lambda: any("v3b_realmlp" in l for l in _starts(spy)), 4)
        finally:
            _kill_proc(q); _kill_proc(f)
    assert results["control"], "harness: control never reached teach mlp"
    assert results["foreign"], "a foreign `less scripts/overnight.py` stalled the queue in wait_render (pgrep -f reads argv)"


# 2. TERM while waiting for A: the trap is deferred until the foreground sleep 300 returns.
def test_term_during_wait_for_A_exits_promptly(tmp_path):
    R, env, spy, _ = _rig(tmp_path, WAIT_SECS="6")  # run_A.done absent: queue sits in wait_A
    q = _launch(R, env)
    try:
        assert _until(lambda: "waiting for A" in _log(R), 3), "harness: queue never reached wait_A"
        t0 = time.monotonic()
        os.kill(q.pid, signal.SIGTERM)
        exited = _wait_exit(q, 2.5)
        dt = time.monotonic() - t0
    finally:
        _kill_proc(q)
    assert exited, f"queue still alive {dt:.1f}s after TERM while waiting for A (header says 'Stop now')"


# 3. TERM during a step: a grandchild that ignores TERM survives the group kill, and the lock is released anyway.
def test_term_leaves_no_grandchild_in_step_group(tmp_path):
    R, env, spy, _ = _rig(tmp_path, GRANDKID="1", TRAIN_SECS="60")
    (R / "logs/run_A.done").write_text("")
    q = _launch(R, env)
    g = None
    try:
        assert _until(lambda: _starts(spy), 4), "harness: no step started"
        g = _group(R)
        assert g, "harness: no step group recorded"
        assert _until(lambda: (R.parent / "kid.pid").exists() and (R.parent / "kid.pid").read_text().strip(), 3)
        os.kill(q.pid, signal.SIGTERM)
        exited = _wait_exit(q, 6)
        survivors = _group_alive(g)
    finally:
        _kill_group(g); _kill_proc(q)
    assert exited and not survivors, (
        f"after TERM: queue exited={exited}, step group {g} still has a live member "
        f"(a child that ignores TERM; on_signal never escalates to KILL, and the lock is released anyway)")


# 4. TERM during a step whose python ignores TERM: on_signal waits on the group leader forever.
def test_term_with_stubborn_step_does_not_hang(tmp_path):
    R, env, spy, _ = _rig(tmp_path, STUBBORN="1", TRAIN_SECS="60")
    (R / "logs/run_A.done").write_text("")
    q = _launch(R, env)
    g = None
    try:
        assert _until(lambda: _starts(spy), 4), "harness: no step started"
        g = _group(R)
        os.kill(q.pid, signal.SIGTERM)
        exited = _wait_exit(q, 6)
    finally:
        _kill_group(g); _kill_proc(q)
    assert exited, "queue hung after TERM: on_signal does `wait $g` on a step that ignores TERM, with no KILL escalation"


# 5. A takeover dir left by a takeover that was killed mid-way wedges the lock forever.
def test_stale_takeover_dir_is_reclaimed(tmp_path):
    R, env, spy, _ = _rig(tmp_path)
    (R / "logs/run_A.done").write_text("")
    lock = R / "logs/queue_i3.lock.d"; lock.mkdir(parents=True)
    (lock / "pid").write_text(f"{_dead_pid()}\n")                 # dead holder, no step file: stale
    (R / "logs/queue_i3.lock.d.takeover").mkdir()                 # left behind by a killed takeover
    q = _launch(R, env)
    try:
        ok = _until(lambda: _starts(spy), 4)
    finally:
        _kill_proc(q)
    assert ok, f"stale lock with a leftover takeover dir is never reclaimed; log tail: {_log(R)[-200:]!r}"


# 6. A live PID the queue cannot signal reads as dead: kill -0 fails with EPERM and is treated as "gone".
def test_live_unsignalable_pid_is_not_done(tmp_path):
    R, env, spy, _ = _rig(tmp_path)
    (R / "logs/run_A.done").write_text("")
    (R / "logs/run_A.pid").write_text("1\n")  # launchd: always alive, not ours, so kill -0 gives EPERM (no signal sent)
    q = _launch(R, env)
    try:
        started = _until(lambda: _starts(spy), 3)
    finally:
        _kill_proc(q)
    assert not started, "run_A.pid=1 (alive, not signalable) was read as finished: kill -0 EPERM counted as dead"


# 7. A holder that stalls between mkdir and writing its pid lets a second queue take over; both train.
def test_stalled_holder_cannot_race_a_second_trainer(tmp_path):
    R, env, spy, _ = _rig(tmp_path, TRAIN_SECS="4", STALL_LOCK="3", STALL_MARK=str(tmp_path / "stalled"))
    (R / "logs/run_A.done").write_text("")
    q1 = _launch(R, env)
    time.sleep(0.5)
    q2 = _launch(R, env)
    try:
        _until(lambda: _max_concurrent(spy) >= 2, 8)
        overlap = _max_concurrent(spy)
    finally:
        _kill_proc(q1); _kill_proc(q2)
    assert overlap <= 1, f"two queues trained at once (max concurrent steps {overlap}); log: {_log(R)[-300:]!r}"


# 8. A lock whose recorded PID was reused by an unrelated live process wedges the queue until someone deletes it.
def test_reused_pid_in_lock_does_not_wedge_queue(tmp_path):
    R, env, spy, _ = _rig(tmp_path)
    (R / "logs/run_A.done").write_text("")
    lock = R / "logs/queue_i3.lock.d"; lock.mkdir(parents=True)
    other = _live()  # an unrelated live process standing in for a PID reused after a crash or reboot
    (lock / "pid").write_text(f"{other.pid}\n")
    q = _launch(R, env)
    try:
        ok = _until(lambda: _starts(spy), 4)
    finally:
        _kill_proc(q); _kill_proc(other)
    assert ok, f"lock holder PID reused by an unrelated process: queue refuses forever; log: {_log(R)[-200:]!r}"
