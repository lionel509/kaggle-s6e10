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


# Breaker 3. TERM during a step: a grandchild that ignores TERM survives the group kill, and the lock is released anyway.
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
        exited = _wait_exit(q, 15)
        survivors = _group_alive(g)
    finally:
        _kill_group(g); _kill_proc(q)
    assert exited and not survivors, (
        f"after TERM: queue exited={exited}, step group {g} still has a live member "
        f"(a child that ignores TERM; on_signal never escalates to KILL, and the lock is released anyway)")


# Breaker 4. TERM during a step whose python ignores TERM: on_signal waits on the group leader forever.
def test_term_with_stubborn_step_does_not_hang(tmp_path):
    R, env, spy, _ = _rig(tmp_path, STUBBORN="1", TRAIN_SECS="60")
    (R / "logs/run_A.done").write_text("")
    q = _launch(R, env)
    g = None
    try:
        assert _until(lambda: _starts(spy), 4), "harness: no step started"
        g = _group(R)
        os.kill(q.pid, signal.SIGTERM)
        exited = _wait_exit(q, 15)
    finally:
        _kill_group(g); _kill_proc(q)
    assert exited, "queue hung after TERM: on_signal does `wait $g` on a step that ignores TERM, with no KILL escalation"


