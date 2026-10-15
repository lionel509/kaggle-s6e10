"""scripts/finals.py: own-only by construction. Neither pick ever holds a non-own: member; SAFE is undominated on both split halves."""
import json, sys
from pathlib import Path
import numpy as np, pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "blend"))
try:
    import finals  # noqa: E402
except ImportError:  # the module is not written yet; each test fails on its own
    finals = None

PREFIXES = ["own:", "tabpfnmember:oof_", "goodpjw2008:oof_members:", "twinstacks:"]


def _blend(root, name, oof, halves, weights, csv=True):
    (root / "logs").mkdir(parents=True, exist_ok=True); (root / "subs").mkdir(parents=True, exist_ok=True)
    json.dump({"oof": oof, "split_half": list(halves), "weights": weights}, open(root / "logs" / f"{name}.json", "w"))
    if csv:
        (root / "subs" / f"{name}.csv").write_text("id,satisfaction\n")


def _random_pool(root, rng):
    n = int(rng.integers(2, 13))
    names = [f"b{int(rng.integers(20, 40))}"] + list(rng.choice([f"b{i}" for i in range(20)] + [f"m_b{i}" for i in range(20)], n - 1, replace=False))
    for j, name in enumerate(names):  # names[0] is always a non-m own-only blend, so a pick always exists
        keys = [f"{PREFIXES[int(rng.integers(0, 4))]}x{int(rng.integers(0, 9))}" for _ in range(int(rng.integers(1, 4)))]
        if j == 0:
            keys = ["own:x0"]
        weights = {k: float(rng.random()) for k in keys}
        halves = rng.uniform(0.955, 0.965, 2)
        _blend(root, name, float(rng.uniform(0.955, 0.965)), halves, weights)


def test_random_pools_never_pick_public_or_m_members(tmp_path, monkeypatch):
    for seed in range(300):
        root = tmp_path / f"s{seed}"
        root.mkdir()
        rng = np.random.default_rng(seed)
        _random_pool(root, rng)
        pool = finals.load_pool(root=str(root))
        assert all(not n.startswith("m_") for n in pool)
        safe = finals.choose(pool)
        assert finals.is_own(pool[safe])
        s = pool[safe]["split_half"]
        for n, b in pool.items():
            if n == safe or not finals.is_own(b):
                continue
            assert not (b["split_half"][0] > s[0] and b["split_half"][1] > s[1]), (seed, safe, n)


def test_fixed_case_picks_b26(tmp_path, monkeypatch):
    _blend(tmp_path, "b26", 0.961857, [0.96148, 0.962176], {"own:a": 0.6, "own:b": 0.4})
    _blend(tmp_path, "b24", 0.961732, [0.961376, 0.962039], {"own:a": 1.0})
    _blend(tmp_path, "b3", 0.9625, [0.9620, 0.9630], {"tabpfnmember:oof_x:pred": 1.0})
    pool = finals.load_pool(root=str(tmp_path))
    assert finals.choose(pool) == "b26"


def test_main_states_follow_hedge_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "test.csv").write_text("id,x\n" + "".join(f"{i},0\n" for i in range(5)))
    (tmp_path / "oof").mkdir(); (tmp_path / "subs" / "hedge").mkdir(parents=True)
    np.save(tmp_path / "oof" / "a_test.npy", np.arange(5.0)); np.save(tmp_path / "oof" / "b_test.npy", np.arange(5.0) * 2)
    _blend(tmp_path, "b26", 0.961857, [0.96148, 0.962176], {"own:a": 0.6, "own:b": 0.4})
    _blend(tmp_path, "b3", 0.9625, [0.9620, 0.9630], {"tabpfnmember:oof_x:pred": 1.0})

    finals.main(["finals"])
    J = json.load(open(tmp_path / "logs" / "finals.json"))
    assert J["safe"] == "b26"
    assert J["aggressive_kind"] == "family_hedge_pending" and J["aggressive_csv"] is None

    (tmp_path / "subs" / "b26_fam.csv").write_text("id,satisfaction\n")
    finals.main(["finals"])
    J = json.load(open(tmp_path / "logs" / "finals.json"))
    assert J["aggressive_kind"] == "family_hedge"

    np.save(tmp_path / "subs" / "hedge" / "a_hedge_test.npy", np.arange(5.0) * 10 + 1)
    finals.main(["finals"])
    J = json.load(open(tmp_path / "logs" / "finals.json"))
    assert J["aggressive_kind"] == "refit_hedge"
    assert (tmp_path / "subs" / "b26_hedge.csv").exists()


@pytest.mark.parametrize("oof,halves", [
    (float("nan"), [0.96, 0.96]), (0.96, [float("nan"), 0.96]),
    (float("inf"), [0.96, 0.96]), (0.96, [0.96, float("-inf")]),
])
def test_nonfinite_record_never_picked(tmp_path, oof, halves):
    _blend(tmp_path, "a_bad", oof, halves, {"own:x": 1.0})  # sorts first, so it would win a NaN tie
    _blend(tmp_path, "b26", 0.961857, [0.96148, 0.962176], {"own:a": 0.6, "own:b": 0.4})
    pool = finals.load_pool(root=str(tmp_path))
    assert "a_bad" not in pool
    assert finals.choose(pool) == "b26"


def test_r3_subs_csv_with_nan_is_never_picked(tmp_path):
    _blend(tmp_path, "b26", 0.961857, [0.96148, 0.962176], {"own:a": 0.6, "own:b": 0.4})
    _blend(tmp_path, "b30", 0.9630, [0.9620, 0.9640], {"own:a": 1.0})
    (tmp_path / "subs" / "b30.csv").write_text("id,satisfaction\n1,nan\n2,0.7\n")
    pool = finals.load_pool(root=str(tmp_path))
    assert "b30" not in pool
    assert finals.choose(pool) == "b26"


def test_r3_oversized_int_record_is_skipped_not_fatal(tmp_path):
    _blend(tmp_path, "b26", 0.961857, [0.96148, 0.962176], {"own:a": 0.6, "own:b": 0.4})
    (tmp_path / "logs" / "b99.json").write_text('{"oof": 1' + "0" * 400 + ', "split_half": [1, 1], "weights": {"own:a": 1.0}}')
    (tmp_path / "subs" / "b99.csv").write_text("id,satisfaction\n")
    pool = finals.load_pool(root=str(tmp_path))
    assert "b99" not in pool
    assert finals.choose(pool) == "b26"
