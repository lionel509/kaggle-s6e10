"""scripts/gate.py: promotion gates for the i3 queue.
Fold gate:  gate.py <cand> <base> [root]                     reads logs/<n>.json (5 folds each)
Blend gate: gate.py --blend <cand.json> <base.json> [root]   reads split_half AUCs from blend JSON
Exit 0 = PROMOTE, 1 = DROP, 2 = MISSING. Appends one line to <root>/logs/queue_i3_gates.jsonl."""
import json, sys
from pathlib import Path


def gate(c, b, min_d=3e-5, min_up=4):
    d = c["oof"] - b["oof"]
    up = sum(x > z for x, z in zip(c["folds"], b["folds"]))
    return d >= min_d - 1e-12 and up >= min_up, d, up


def blend_gate(c, b, min_d=3e-5):
    d = c["oof"] - b["oof"]
    up = sum(x > z for x, z in zip(c["split_half"], b["split_half"]))
    return d >= min_d - 1e-12 and up == 2, d, up


def _read(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _append(root, rec):
    (root / "logs").mkdir(parents=True, exist_ok=True)
    with open(root / "logs" / "queue_i3_gates.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def main(argv):
    args = argv[1:]
    blend = bool(args) and args[0] == "--blend"
    args = args[1:] if blend else args
    cand, base = args[0], args[1]
    root = Path(args[2]) if len(args) > 2 else Path(".")
    if blend:
        c, b = _read(cand), _read(base)
        names = (Path(cand).stem, Path(base).stem)
        ok_in = c is not None and b is not None and len(c.get("split_half", [])) == 2 and len(b.get("split_half", [])) == 2
        n, unit = 2, "halves"
    else:
        c, b = _read(root / "logs" / f"{cand}.json"), _read(root / "logs" / f"{base}.json")
        names = (cand, base)
        ok_in = c is not None and b is not None and len(c.get("folds", [])) == 5 and len(b.get("folds", [])) == 5
        n, unit = 5, "folds"
    if not ok_in:
        print(f"GATE MISSING {names[0]} vs {names[1]}")
        _append(root, {"cand": names[0], "base": names[1], "missing": True})
        return 2
    ok, d, up = blend_gate(c, b) if blend else gate(c, b)
    print(f"GATE {'PROMOTE' if ok else 'DROP'} {names[0]} vs {names[1]}: {d:+.6f} OOF, {up}/{n} {unit} up")
    _append(root, {"cand": names[0], "base": names[1], "cand_oof": c["oof"], "base_oof": b["oof"],
                   "delta": d, "up": up, "pass": bool(ok), "missing": False})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
