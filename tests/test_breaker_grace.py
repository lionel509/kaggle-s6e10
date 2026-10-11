"""Breaker probes for scripts/queue_i3.sh on_signal grace + KILL (PR #4, da5b9d4). Failing probe only; the setsid probe (design limit, not from da5b9d4) is not adopted.

Each test copies the queue into a temp repo, puts a stub .venv/bin/python in front of the step, and starts the
queue detached. The stub appends its PIDs to <root>/pids and sleeps. The test sends TERM to the queue's own PID
(read from its lock dir). Only processes these tests started get signalled; liveness is probed with kill -0 / killpg(0).
Nothing calls pkill, pgrep or ps to choose a target.
"""
import os, shutil, signal, subprocess, time
from pathlib import Path
import pytest

QUEUE = Path(__file__).resolve().parents[1] / "scripts" / "queue_i3.sh"

MODES = {
    # step ignores TERM; its sleep children inherit the ignore
    "stubborn": "trap '' TERM\nwhile :; do sleep 1; done\n",
}
# Prelude for the survivor probe only: `kill -0 -- -<group>` reports "alive" after KILL, which forces the
# "survived KILL" branch. The real KILL still goes to the real group through builtin kill.
SURVIVE_PRELUDE = 'kill() { if [[ -n $FAKE_SURVIVE && $1 == -0 && $2 == -- ]]; then return 0; fi; builtin kill "$@"; }\n'


def make_root(tmp_path, mode, survive=False):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / ".venv" / "bin").mkdir(parents=True)
    src = QUEUE.read_text()
    if survive:
        src = src.replace("#!/bin/zsh\n", "#!/bin/zsh\n" + SURVIVE_PRELUDE, 1)
    (root / "scripts" / "queue_i3.sh").write_text(src)
    (root / "logs" / "run_A.done").touch()
    (root / "logs" / "run_A.pid").write_text("0\n")  # 0 counts as gone: A is finished, no process is touched
    py = root / ".venv" / "bin" / "python"
    py.write_text("#!/bin/sh\nD=" + str(root) + "\necho $$ >> \"$D/pids\"\n" + MODES[mode])
    py.chmod(0o755)
    return root


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_for(pred, secs):
    end = time.time() + secs
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.2)
    return False


def start_queue(root):
    subprocess.run(["sh", "-c", f"cd '{root}' && (nohup zsh scripts/queue_i3.sh >/dev/null 2>&1 &)"],
                   check=True, timeout=10)
    lockpid = root / "logs" / "queue_i3.lock.d" / "pid"
    assert wait_for(lambda: lockpid.exists() and lockpid.read_text().strip().isdigit(), 15), "queue never took the lock"
    qpid = int(lockpid.read_text())
    stepfile = root / "logs" / "queue_i3.lock.d" / "step"
    assert wait_for(lambda: (root / "pids").exists() and stepfile.exists(), 20), "step never started"
    return qpid


def pids_of(root):
    f = root / "pids"
    return [int(x) for x in f.read_text().split()] if f.exists() else []


def stop(qpid, limit=40):
    t0 = time.time()
    try:
        os.kill(qpid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    gone = wait_for(lambda: not alive(qpid), limit)
    return gone, time.time() - t0


@pytest.fixture
def cleanup():
    started = []
    yield started
    for p in started:  # only PIDs the stubs recorded; SIGKILL only for leftovers of a failing run
        if alive(p):
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_lock_not_released_while_group_survives_kill(tmp_path, cleanup):
    # Probe: the step group still reports alive after KILL (simulated, see SURVIVE_PRELUDE). The script's own
    # comment says: "survived KILL, lock kept (a relaunch refuses until it is gone)".
    root = make_root(tmp_path, "stubborn", survive=True)
    os.environ["FAKE_SURVIVE"] = "1"
    try:
        q = start_queue(root)
        cleanup.extend(pids_of(root))
        gone, dt = stop(q)
    finally:
        os.environ.pop("FAKE_SURVIVE", None)
    assert gone, "queue did not exit"
    log = (root / "logs" / "queue_i3.log").read_text()
    assert "survived KILL, lock kept" in log, log[-500:]
    assert (root / "logs" / "queue_i3.lock.d").exists(), \
        "lock released although the step group survived (EXIT trap release_lock)"
    # (b) half: a relaunch while the survivor is there must refuse and start no step
    n_before = len(pids_of(root))
    relaunch = subprocess.run(["zsh", "scripts/queue_i3.sh"], cwd=root, capture_output=True, timeout=30,
                              env={**os.environ, "FAKE_SURVIVE": "1"})
    assert relaunch.returncode == 1, (relaunch.returncode, relaunch.stderr[-300:])
    assert len(pids_of(root)) == n_before, "relaunch started a step beside the surviving group"


