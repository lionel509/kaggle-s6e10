"""Shared fixtures. Smoke rigs run the real scripts on a 3000-row slice, CPU, inside a throwaway dir."""
import os, subprocess, sys
from pathlib import Path
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[1]
SWITCHES = ("SMOKE", "LEAVE", "SOFT", "BLKTE", "TEACH", "REFIT", "EXTRA", "AUX", "CROSS", "BLOCK", "K", "DEVICE", "THREADS")


class Rig:
    def __init__(self, root, data):
        self.root, self.data = root, data

    def run(self, script, *args, **env):
        src = (REPO / "models" / script).read_text()  # missing file fails here, before any run
        assert "SMOKE" in src, f"{script} has no SMOKE switch yet"
        base = {k: v for k, v in os.environ.items() if k not in SWITCHES}
        base |= {"SMOKE": "3000", "THREADS": "2", "DEVICE": "cpu"}
        base |= {k: str(v) for k, v in env.items()}
        p = subprocess.run([sys.executable, f"models/{script}", *map(str, args)], cwd=self.root, env=base,
                           capture_output=True, text=True, timeout=900)
        assert p.returncode == 0, p.stderr[-2000:]
        return p


@pytest.fixture
def smoke_rig(tmp_path):
    if os.environ.get("RUN_SMOKE") != "1":
        pytest.skip("set RUN_SMOKE=1 (CPU smoke, about 10 min)")
    data = Path(os.environ.get("S6E10_DATA", REPO / "data"))
    if not (data / "feat_lf.parquet").exists():
        pytest.skip("S6E10_DATA must point at a data/ with the feature caches")
    root = tmp_path
    (root / "data").mkdir()
    for p in data.iterdir():  # per-file symlinks: scripts read data/, never write it
        if p.name != "feat_teacher_cat2.parquet":
            (root / "data" / p.name).symlink_to(p)
    (root / "models").symlink_to(REPO / "models")
    # fake 5th teacher, so TEACH=3 can be tested without teacher_cat.py
    import numpy as np, pandas as pd
    n = pq.ParquetFile(data / "feat_lf.parquet").metadata.num_rows
    pd.DataFrame({"t_catx": np.zeros(n, dtype="float32")}).to_parquet(root / "data" / "feat_teacher_cat2.parquet")
    return Rig(root, data)
