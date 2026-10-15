"""Final picks, own-only by construction. Public and m_* blends are never read as picks.
SAFE: the highest-OOF own-only blend that no other own-only blend beats strictly on both split halves.
AGGR: SAFE with its own-member test columns refit-swapped (hedge files), else its family hedge (subs/<SAFE>_fam.csv).
Usage (from the repo root): python scripts/finals.py   -> prints the table, writes logs/finals.json"""
import csv, datetime, json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blend"))
import hedge  # noqa: E402


def _num(v):
    # an AUC in [0, 1]; NaN fails the comparison, and a huge int compares exactly (math.isfinite would overflow)
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0


def _subs_finite(path):
    """Every satisfaction value in the submission is finite; a NaN row would upload as an invalid file."""
    try:
        with open(path, newline="") as f:
            vals = [float(r["satisfaction"]) for r in csv.DictReader(f)]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return all(math.isfinite(v) for v in vals)


def load_pool(root="."):
    """Every logs/<name>.json that is a valid blend record with its subs/<name>.csv; m_* measurement files are skipped."""
    root = Path(root); pool = {}
    for p in sorted((root / "logs").glob("*.json")):
        if p.stem.startswith("m_"): continue
        try:
            b = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not (isinstance(b, dict) and _num(b.get("oof")) and isinstance(b.get("split_half"), list)
                and len(b["split_half"]) == 2 and all(_num(v) for v in b["split_half"])
                and isinstance(b.get("weights"), dict)):
            continue
        if not _subs_finite(root / "subs" / f"{p.stem}.csv"): continue
        pool[p.stem] = b
    return pool


def is_own(b):
    return bool(b["weights"]) and all(k.startswith("own:") for k in b["weights"])


def choose(pool):
    own = {n: b for n, b in pool.items() if is_own(b)}
    if not own:
        raise ValueError("no own-only blend")
    def beats(a, b):
        return all(x > y for x, y in zip(a["split_half"], b["split_half"]))
    undominated = [n for n in own if not any(beats(own[m], own[n]) for m in own if m != n)]
    return min(undominated, key=lambda n: (-own[n]["oof"], n))


def aggressive(pool, safe, root="."):
    _, _, swapped = hedge.hedge_matrix(pool[safe]["weights"], root)
    if swapped:
        hedge.main(["", safe, f"{safe}_hedge"])
        return "refit_hedge", f"subs/{safe}_hedge.csv"
    if (Path(root) / "subs" / f"{safe}_fam.csv").exists():
        return "family_hedge", f"subs/{safe}_fam.csv"
    print(f"RUN: .venv/bin/python blend/run_blend.py {safe}_fam --own-only --drop 'v4_cat'")
    return "family_hedge_pending", None


def main(argv):
    pool = load_pool()
    safe = choose(pool)
    kind, csv = aggressive(pool, safe)
    aggr = {"refit_hedge": f"{safe}_hedge", "family_hedge": f"{safe}_fam"}.get(kind)
    print("name | oof | half_a | half_b | own | pick")
    for n, b in sorted(pool.items(), key=lambda kv: -kv[1]["oof"]):
        h = b["split_half"]
        print(f"{n} | {b['oof']:.6f} | {h[0]:.6f} | {h[1]:.6f} | {'yes' if is_own(b) else 'no'} | {'SAFE' if n == safe else ''}")
    if aggr:
        print(f"{aggr} | | | | yes | AGGR")
    Path("logs").mkdir(exist_ok=True)
    Path("logs/finals.json").write_text(json.dumps({
        "safe": safe, "safe_csv": f"subs/{safe}.csv", "aggressive": aggr, "aggressive_kind": kind,
        "aggressive_csv": csv, "date": datetime.date.today().isoformat()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
