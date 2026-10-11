"""Round-3b breaker probes for the BUSY pattern in scripts/queue_i3.sh at 2267620.
The pattern is read from the script and matched with the real pgrep -f. Each probe starts its own
stub (sleep under exec -a, so argv is the chosen command line), checks whether pgrep sees it, and
kills only that Popen child. Only the failing probes are kept."""
import re, subprocess, time, pathlib, pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (REPO / "scripts" / "queue_i3.sh").read_text()
BUSY = re.search(r"^BUSY='(.*)'$", SCRIPT, re.M).group(1)


def stub(argv):
    p = subprocess.Popen(["/bin/bash", "-c", 'exec -a "$0" /bin/sleep 300', argv])
    deadline = time.time() + 5
    while time.time() < deadline:
        cmd = subprocess.run(["ps", "-o", "command=", "-p", str(p.pid)], capture_output=True, text=True).stdout.strip()
        if cmd.startswith(argv):
            break
        time.sleep(0.05)
    return p


def busy_pids():
    out = subprocess.run(["pgrep", "-f", BUSY], capture_output=True, text=True, timeout=20).stdout
    return {int(x) for x in out.split()}


def check(argv, should_block):
    p = stub(argv)
    try:
        blocked = p.pid in busy_pids()
    finally:
        p.kill(); p.wait()
    assert blocked == should_block, f"{argv!r}: BUSY {'matched' if blocked else 'missed'}, expected {'match' if should_block else 'no match'}"


# False positives: a viewer or a non-training python that holds the queue
@pytest.mark.parametrize("argv", [
    "grep -n python models/v4_catboost.py",
    "python -m pydoc models/v4_catboost.py",
    "less logs/run_A.sh",
])
def test_r3b_editor_or_viewer_does_not_block(argv):
    check(argv, should_block=False)


# Holes: training that the pattern misses
@pytest.mark.parametrize("argv", [
    "python -m models.v4_catboost",
    "python v4_catboost.py",
    "trainer models/v4_catboost.py",
])
def test_r3b_training_hole(argv):
    check(argv, should_block=True)
