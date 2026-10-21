#!/bin/zsh
# Pull goodpjw2008's CC0 TabPFN/TabICL member dataset (on our exact split) and check it against our folds.
# train_tabicl.py inside the download is read for porting only, never executed.
set -euo pipefail
cd "$(dirname "$0")/.."
D=public_out/tabpfnmember_goodpjw2008
mkdir -p "$D"
kaggle datasets download goodpjw2008/s6e10-tabpfn-member -p "$D" --unzip -q
.venv/bin/python blend/public_lib.py --check "$D"
