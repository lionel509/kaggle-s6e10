"""Hedge final: rank-average our stack csv with a public blend csv.

Usage:
  python3 scripts/hedge_blend.py --public public/<file>.csv [--ours subs/b27_lr_tfm_pub124.csv]
      prints rank correlations for w in {0.3, 0.5} (no files written)
  python3 scripts/hedge_blend.py --public public/<file>.csv --write 0.5
      also writes subs/hedge_<w>.csv (ids checked against data/sample_submission.csv)

Blend: (1-w) * rank(ours) + w * rank(public), then re-ranked to (rank-0.5)/N like the other subs.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = (0.3, 0.5)


def unit_rank(x):
    return (rankdata(x) - 0.5) / len(x)


def load_aligned(path, ids):
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "satisfaction"]:
        raise SystemExit(f"{path}: unexpected columns {list(df.columns)}")
    if len(df) != len(ids) or not np.array_equal(df.id.values, ids):
        raise SystemExit(f"{path}: ids do not match data/sample_submission.csv order")
    return df.satisfaction.values.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="subs/b27_lr_tfm_pub124.csv")
    ap.add_argument("--public", required=True)
    ap.add_argument("--write", type=float, choices=WEIGHTS, default=None)
    args = ap.parse_args()

    sample = pd.read_csv(REPO / "data/sample_submission.csv", usecols=["id"])
    ids = sample.id.values
    ours = load_aligned(REPO / args.ours, ids)
    pub = load_aligned(Path(args.public) if Path(args.public).is_absolute()
                       else REPO / args.public, ids)
    r_ours, r_pub = unit_rank(ours), unit_rank(pub)
    print(f"rows {len(ids)}  ours={args.ours}  public={args.public}")
    print(f"spearman ours vs public (raw): {spearmanr(ours, pub)[0]:.6f}")

    blends = {}
    for w in WEIGHTS:
        b = unit_rank((1 - w) * r_ours + w * r_pub)
        blends[w] = b
        print(f"w={w}: spearman vs ours {spearmanr(b, ours)[0]:.6f}  "
              f"vs public {spearmanr(b, pub)[0]:.6f}")

    if args.write is not None:
        out = REPO / "subs" / f"hedge_{args.write}.csv"
        pd.DataFrame({"id": ids, "satisfaction": blends[args.write]}).to_csv(out, index=False)
        chk = pd.read_csv(out)
        assert len(chk) == len(ids) and np.array_equal(chk.id.values, ids)
        assert chk.satisfaction.notna().all()
        print(f"wrote {out}  rows={len(chk)}")


if __name__ == "__main__":
    main()
