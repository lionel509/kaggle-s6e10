"""Breaker probes for sweep_group / step() in scripts/queue_i3.sh (PR #4 at 1dd50aa).

Each test copies queue_i3.sh's function block (lines 1-166) into a temp root, appends a
test body that calls the real step(), and runs it under zsh. Stub step commands are
shell scripts in the temp root; each stub waits until the member it starts is ready
before the step leader exits, so the sweep meets a settled group, not a half-started one.
The only override is wait_render (a no-op): a live scripts/overnight.py exists on this
machine and pgrep would wait on it. Every process a test starts is killed by the test
via its recorded PID, never by name.
"""
import os
import pathlib
import shutil
import signal
import subprocess
import tempfile
import time

import pytest

QUEUE = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "queue_i3.sh"
DEFS_LINES = 166  # last line of on_signal in queue_i3.sh at 1dd50aa
WORK = os.path.expanduser("~/worktrees/kaggle-s6e10")

WAIT = 'i=0; while [ ! -s "$ROOT/%s" ] && [ $i -lt 100 ]; do sleep 0.05; i=$((i+1)); done\n'

STUBS = {
    "clean.sh": "exit 0\n",
    # member ignores TERM (only KILL ends it)
    "ignore_term.sh": (
        "sh -c 'trap \"\" TERM; echo $$ > \"$ROOT/member.pid\"; exec sleep 300' &\n"
        + WAIT % "member.pid" + "exit 0\n"),
    # member takes ~9 s to exit after TERM (inside the 10 s grace)
    "slow_term.sh": (
        "sh -c 'trap \"sleep 8; exit 0\" TERM; echo $$ > \"$ROOT/member.pid\"; while :; do sleep 1; done' &\n"
        + WAIT % "member.pid" + "exit 0\n"),
    # member that starts a new group child on every TERM
    "respawn.sh": (
        "sh -c 'trap \"sh -c \\\"sleep 300\\\" &\" TERM; echo $$ > \"$ROOT/member.pid\"; while :; do sleep 1; done' &\n"
        + WAIT % "member.pid" + "exit 0\n"),
    # member that stopped itself; TERM stays pending until KILL
    "stopped.sh": (
        "sh -c 'echo $$ > \"$ROOT/member.pid\"; kill -STOP $$; exec sleep 300' &\n"
        + WAIT % "member.pid" + "exit 0\n"),
    # member calls setsid (leaves the step group) and runs 20 s like a training
    "escape.sh": (
        "/usr/bin/perl -MPOSIX -e 'POSIX::setsid(); open my $f, \">\", $ARGV[0] or die; print $f $$; close $f; sleep 20' \"$ROOT/esc.pid\" &\n"
        + WAIT % "esc.pid" + "exit 0\n"),
    # records whether the escaped member is alive when the next step starts
    "check_esc.sh": (
        "p=$(cat \"$ROOT/esc.pid\" 2>/dev/null)\n"
        "if [ -n \"$p\" ] && kill -0 \"$p\" 2>/dev/null; then echo ALIVE > \"$ROOT/s2.seen\"; else echo DEAD > \"$ROOT/s2.seen\"; fi\n"
        "exit 0\n"),
    # a process in its own session, outside the step group, that must survive the sweep
    "control.sh": (
        "/usr/bin/perl -MPOSIX -e 'POSIX::setsid(); open my $f, \">\", $ARGV[0] or die; print $f $$; close $f; sleep 60' \"$ROOT/control.pid\" &\n"
        + WAIT % "control.pid" + "exit 0\n"),
}

BODY_HEAD = """
take_lock || { print -r -- "lock busy"; exit 9; }
trap release_lock EXIT
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT
wait_render() { :; }
"""


def _root(tmp):
    root = tmp / "root"
    (root / "scripts").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "logs" / "run_A.done").write_text("")  # a_done true, no pid file
    (root / "stubs").mkdir()
    for name, body in STUBS.items():
        (root / "stubs" / name).write_text("#!/bin/sh\nROOT=%s\nexport ROOT\n" % root + body)
    return root


def _harness(root, steps):
    head = "".join(QUEUE.read_text().splitlines(keepends=True)[:DEFS_LINES])
    path = root / "scripts" / "harness.zsh"
    path.write_text(head + "ROOT=%s\n" % root + BODY_HEAD + steps)
    return path


def _pid_of(root, name):
    f = root / name
    return int(f.read_text().strip()) if f.exists() and f.read_text().strip() else None


def _alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _kill_ours(pids):
    for p in pids:
        if _alive(p):
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass


@pytest.fixture
def tmp():
    d = pathlib.Path(tempfile.mkdtemp(prefix="sweep-", dir=WORK))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _run(harness, timeout=55):
    return subprocess.run(["/bin/zsh", str(harness)], cwd=str(harness.parent.parent),
                          capture_output=True, text=True, timeout=timeout)


def _cleanup(root, *names):
    _kill_ours([_pid_of(root, n) for n in ("member.pid", "esc.pid", "control.pid") if n in names or not names])


def test_escaped_member_next_step_must_not_start(tmp):
    """A member that left the step group (setsid) is still training when the next step starts."""
    root = _root(tmp)
    h = _harness(root, 'step s1 /bin/sh "$ROOT/stubs/escape.sh"\nsleep 1\n'
                       'step s2 /bin/sh "$ROOT/stubs/check_esc.sh"\n')
    try:
        _run(h)
        seen = (root / "s2.seen").read_text().strip()
        assert seen == "DEAD", "next step started while the escaped member was still alive: %s" % seen
    finally:
        _cleanup(root)
