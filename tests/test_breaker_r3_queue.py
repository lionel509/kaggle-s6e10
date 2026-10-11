"""Round-3 breaker probes for scripts/queue_i3.sh at 0b5d6ba. Only the failing probes are kept.
The queue runs in a temp repo with a fake .venv python (models/* log START/END to a spy file).
pgrep is shimmed to see only processes whose cwd is the temp repo, so the real A is never seen.
Stub training processes are sleeps renamed with exec -a. Nothing signals a process this test did not start."""
import os, shutil, signal, stat, subprocess, time
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
REAL_PY = "/Users/lionelweng/Documents/Citadel/Active/Kaggle S6E10/.venv/bin/python"
GATE = REPO / "scripts" / "gate.py"
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
# the queue's 300 s poll and the 60 s busy poll both become 1 s
FAST_SLEEP = '#!/bin/sh\n[ "$1" = 300 -o "$1" = 60 ] && set -- 1\nexec /bin/sleep "$1"\n'
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
    for d in ("scripts", "blend", "logs", "subs", "models", ".venv/bin", "shim", "data/soft"):
        (R / d).mkdir(parents=True, exist_ok=True)
    for f in ("queue_i3.sh", "gate.py", "report_i3.py"):
        shutil.copy(REPO / "scripts" / f, R / "scripts" / f)
    _exec(R / ".venv/bin/python", STUB)
    _exec(R / "shim/sleep", FAST_SLEEP)
    _exec(R / "shim/pgrep", PGREP_SHIM)
    (R / "logs/run_A.done").write_text("")  # stale done-file, as in the review's scenario
    spy = tmp_path / "spy.txt"; spy.write_text("")
    env = {**os.environ, "SPY": str(spy), "REAL_PY": REAL_PY, "PATH": f"{R/'shim'}:{os.environ['PATH']}"}
    return R, env, spy


def _launch(R, env):
    return subprocess.Popen(["zsh", "scripts/queue_i3.sh"], cwd=R, env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _fake(R, argv0, secs=60):
    """A live process whose argv is exactly argv0 (exec -a), cwd = temp repo, in its own session."""
    return subprocess.Popen(["zsh", "-c", f'exec -a "{argv0}" /bin/sleep {secs}'], cwd=R, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _kill_group(p):
    try: os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError): pass
    p.wait()


def _starts(spy):
    return [l for l in spy.read_text().splitlines() if " START " in l]


def _alive(R, pattern):
    r = subprocess.run(["/usr/bin/pgrep", "-f", pattern], capture_output=True, text=True)
    return r.returncode == 0


def test_r3_open_editor_on_a_model_file_blocks_queue_forever(rig):
    """A non-training process whose argv merely names a models/*.py file (an editor or pager with the file open)
    matches the BUSY pattern, so wait_busy loops for as long as that window stays open. With a stale run_A.done
    and nothing training, the queue should start its first teacher. Expected: a START within 12 s."""
    R, env, spy = rig
    ed = _fake(R, "less models/v4_catboost.py")
    p = None
    try:
        time.sleep(0.5)
        assert _alive(R, "less models/v4_catboost.py"), "precondition: stub editor not alive"
        p = _launch(R, env)
        time.sleep(12)
    finally:
        if p: _kill_group(p)
        _kill_group(ed)
    assert _starts(spy), "queue waited on an open editor and started nothing in 12 s (log: wait_busy loop)"
