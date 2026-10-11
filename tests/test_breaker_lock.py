"""Breaker probes for release_lock in scripts/queue_i3.sh at 878f969 (PR #4, commit 878f969). Failing probes only.

Each test copies the queue into a temp repo with a stub .venv/bin/python and a stub pgrep (a real overnight render
on this Mac would otherwise block wait_render), launches it in its own session, and signals only the queue it
started. The stub records PIDs in <root>/pids; the first PID is the step's process group. A group is only killed
when every member's cwd is inside this test's temp root (PIDs can be reused).
"""
import os, signal, subprocess, time
from pathlib import Path

QUEUE = Path(__file__).resolve().parents[1] / "scripts" / "queue_i3.sh"
LOCK = "logs/queue_i3.lock.d"

# Normal exit: the first step's leader exits 0 and leaves a child (sleep 600) in its process group.
# step() removes the step file, the queue ends normally, and the group is still alive.
LINGER = 'if [ ! -e "$D/linger" ]; then touch "$D/linger"; sleep 600 & fi\nexit 0\n'


def make_root(tmp_path, stub_body):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / "scripts" / "queue_i3.sh").write_text(QUEUE.read_text())
    (root / "logs" / "run_A.done").touch()
    (root / "logs" / "run_A.pid").write_text("0\n")  # 0 counts as gone
    py = root / ".venv" / "bin" / "python"
    py.write_text(f"#!/bin/sh\nD={root}\necho $$ >> \"$D/pids\"\n" + stub_body)
    py.chmod(0o755)
    (root / "bin").mkdir()
    pg = root / "bin" / "pgrep"
    pg.write_text("#!/bin/sh\nexit 1\n")
    pg.chmod(0o755)
    return root


def launch(root):
    env = dict(os.environ, PATH=f"{root}/bin:" + os.environ["PATH"])
    return subprocess.Popen(["zsh", str(root / "scripts" / "queue_i3.sh")], cwd=str(root), env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def first_group(root):
    return int((root / "pids").read_text().split()[0])


def group_is_ours(pgid, root):
    out = subprocess.run(["ps", "-axo", "pid=,pgid="], capture_output=True, text=True).stdout
    pids = [int(l.split()[0]) for l in out.splitlines() if len(l.split()) == 2 and int(l.split()[1]) == pgid]
    if not pids:
        return False
    for p in pids:
        cwd = subprocess.run(["lsof", "-a", "-p", str(p), "-d", "cwd", "-Fn"], capture_output=True, text=True).stdout
        if str(root) not in cwd:
            return False
    return True


def cleanup(root, q):
    if q.poll() is None:
        q.kill()
        q.wait(timeout=10)
    try:
        pgid = first_group(root)
    except (FileNotFoundError, ValueError, IndexError):
        return
    if group_is_ours(pgid, root):
        os.killpg(pgid, signal.SIGKILL)


def test_breaker_lock_kept_when_normal_exit_leaves_group_member(tmp_path):
    # Expected per the commit's rule (lock_live): a live step group keeps the lock. Actual: released.
    root = make_root(tmp_path, LINGER)
    q = launch(root)
    try:
        q.wait(timeout=50)
        assert group_alive(first_group(root)), "precondition: a child should still be in the step group"
        assert (root / LOCK).exists(), "lock released while the step group is still alive"
    finally:
        cleanup(root, q)


def test_breaker_relaunch_refused_after_normal_exit_with_group_member(tmp_path):
    # Consequence of the same break: a second queue takes the lock and runs steps beside the live group.
    root = make_root(tmp_path, LINGER)
    q1 = launch(root)
    q2 = None
    try:
        q1.wait(timeout=50)
        assert group_alive(first_group(root)), "precondition: a child should still be in the step group"
        q2 = launch(root)
        q2.wait(timeout=50)
        log = (root / "logs" / "queue_i3.log").read_text()
        assert "another queue holds" in log, "relaunch started beside a live step group"
    finally:
        cleanup(root, q1)
        if q2 is not None:
            cleanup(root, q2)
