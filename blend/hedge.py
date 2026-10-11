"""blend/hedge.py: swaps refit hedge files into a blend's test columns and writes subs/<out>.csv.
own:<n> uses subs/hedge/<n>_hedge_test.npy when it exists, else oof/<n>_test.npy.
Usage: python blend/hedge.py <blend_name> <out>. Hedge files live outside oof/, so run_blend never reads them."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend import blend as blend_fn


def hedge_matrix(weights, root="."):
    root = Path(root)
    names = list(weights)
    for n in names:
        if not n.startswith("own:"):
            raise ValueError(f"hedge only swaps own: members, got {n}")
    cols, swapped = [], []
    for n in names:
        s = n.removeprefix("own:")
        hedge = root / "subs" / "hedge" / f"{s}_hedge_test.npy"
        if hedge.exists():
            cols.append(np.load(hedge)); swapped.append(n)
        else:
            cols.append(np.load(root / "oof" / f"{s}_test.npy"))
    T = np.column_stack(cols)
    w = np.array([weights[n] for n in names], float)
    return T, w, swapped


def main(argv):
    blend_name, out = argv[1], argv[2]
    weights = json.load(open(f"logs/{blend_name}.json"))["weights"]
    T, w, swapped = hedge_matrix(weights)
    ids = pd.read_csv("data/test.csv", usecols=["id"]).id
    pd.DataFrame({"id": ids, "satisfaction": norm.cdf(blend_fn(T, w))}).to_csv(f"subs/{out}.csv", index=False)
    print(f"HEDGE {out}: swapped {len(swapped)}/{len(weights)}: {[s.removeprefix('own:') for s in swapped]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
