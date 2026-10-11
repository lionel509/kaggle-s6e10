"""scripts/report_i3.py: writes logs/queue_i3_report.md for the i3 queue. Never raises on missing inputs."""
import json, re, sys
from pathlib import Path

INPUTS = ["v4_cat_s42_extra_aux_teach_cross_block", "v5_xgb_d8_s42_k20_extra_aux_teach", "b24_block"]
A_NAMES = ["v4_cat_s42_k20_extra_aux_teach_cross_block", "v3b_realmlp_e6_n8_s42_k20_aux_teach"]
FAIL_RE = re.compile(r"FAILED|SKIP|dropped|preflight")


def _json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _lines(p):
    try:
        return Path(p).read_text().splitlines()
    except OSError:
        return []


def _halves(lines):
    out = []
    for l in lines:
        m = re.search(r"split-half held-out ([0-9.]+)", l)
        if m:
            out.append(float(m.group(1)))
    return out


def _fmt(x, spec=".6f"):
    return "MISSING" if x is None else format(x, spec)


def _oof(root, n):
    j = _json(Path(root) / "logs" / f"{n}.json")
    return j.get("oof") if isinstance(j, dict) else None


def build(root):
    root = Path(root); L = root / "logs"
    out = ["# i3 queue report", "", "## Inputs (Oct 7)"]
    out += [f"- {n}: OOF {_fmt(_oof(root, n))}" for n in INPUTS]

    out += ["", "## A (resume + RealMLP)"]
    for n in A_NAMES:
        j = _json(L / f"{n}.json")
        folds = len(j.get("folds", [])) if isinstance(j, dict) else "MISSING"
        out.append(f"- {n}: OOF {_fmt(_oof(root, n))}, folds {folds}")

    # Gates: last JSONL record per cand. Blend gates (cand q3_*) count halves, fold gates count 5 folds.
    recs = {}
    for line in _lines(L / "queue_i3_gates.jsonl"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        recs[r.get("cand")] = r
    out += ["", "## Gates", "", "| cand | base | cand OOF | base OOF | Δ | up | verdict | k20 OOF |",
            "|---|---|---|---|---|---|---|---|"]
    if not recs:
        out.append("| MISSING | | | | | | MISSING | |")
    for cand, r in recs.items():
        if r.get("missing"):
            out.append(f"| {cand} | {r.get('base')} | MISSING | MISSING | — | — | MISSING | — |")
            continue
        n = 2 if str(cand).startswith("q3_") else 5
        k20 = re.sub(r"(_s\d+)", r"\1_k20", str(cand), count=1)
        k20oof = _oof(root, k20) if k20 != cand else None
        out.append(f"| {cand} | {r.get('base')} | {_fmt(r.get('cand_oof'))} | {_fmt(r.get('base_oof'))} | "
                   f"{_fmt(r.get('delta'), '+.6f')} | {r.get('up')}/{n} | "
                   f"{'PROMOTE' if r.get('pass') else 'DROP'} | {'—' if k20oof is None else _fmt(k20oof)} |")

    out += ["", "## Blends"]
    bj = {n: _json(L / f"{n}.json") for n in ("q3_base", "q3_all")}
    bh = {n: _halves(_lines(L / f"{n}.log")) for n in ("q3_base", "q3_all")}
    for n in ("q3_base", "q3_all"):
        j = bj[n] if isinstance(bj[n], dict) else {}
        top = sorted(j.get("weights", {}).items(), key=lambda kv: -kv[1])[:8]
        out.append(f"- {n}: OOF {_fmt(j.get('oof'))}, split-half {[round(v, 6) for v in bh[n]] or 'MISSING'}")
        out.append("  top weights: " + (", ".join(f"{k} {v:.3f}" for k, v in top) or "MISSING"))
    hb, ha = bh["q3_base"], bh["q3_all"]
    ob = bj["q3_base"].get("oof") if isinstance(bj["q3_base"], dict) else None
    oa = bj["q3_all"].get("oof") if isinstance(bj["q3_all"], dict) else None
    if len(hb) == 2 and len(ha) == 2 and ob is not None and oa is not None:
        both = ha[0] > hb[0] and ha[1] > hb[1]
        out.append(f"- q3_all − q3_base: ΔOOF {oa - ob:+.6f}; halves {ha[0]:.6f} vs {hb[0]:.6f}, "
                   f"{ha[1]:.6f} vs {hb[1]:.6f}; both up: {'yes' if both else 'no'}")
    else:
        out.append("- q3_all − q3_base: MISSING")
    b24 = _oof(root, "b24_block")
    out.append(f"- q3_all − b24: ΔOOF {_fmt(None if oa is None or b24 is None else oa - b24, '+.6f')}")

    out += ["", "## Hedge (E)"]
    sw = sorted(p.name.removesuffix("_hedge_test.npy") for p in (root / "subs" / "hedge").glob("*_hedge_test.npy"))
    out.append("- swapped members: " + (", ".join(sw) if sw else "none"))
    out.append("- subs/q3_all_hedge.csv: " + ("present" if (root / "subs" / "q3_all_hedge.csv").exists() else "MISSING"))

    ql = _lines(L / "queue_i3.log")
    out += ["", "## Failures", "(rc=142 means the timeout fired)"]
    fl = [f"- {l}" for l in ql if FAIL_RE.search(l)]
    out += fl if fl else ["- none"]

    out += ["", "## Queue", f"- first: {ql[0] if ql else 'MISSING'}", f"- last: {ql[-1] if ql else 'MISSING'}"]
    out += ["", "## Not done", "- no submission; main session submits", ""]
    L.mkdir(parents=True, exist_ok=True)
    (L / "queue_i3_report.md").write_text("\n".join(out))


def main(root="."):
    build(root)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
