#!/bin/zsh
# 20-fold versions of b24's three k5 winners (issue #1 next-step): CatBoost +aux+block, XGB +aux, RealMLP +aux.
# Wave 1: cat (6t) + xgb (4t) in parallel on CPU. Wave 2: realmlp alone on MPS —
# torch + xgb/lgb OpenMP runtimes segfault (exit 139, v3 note), so never beside a GBDT.
# Never submits. Writes logs/k20_winners.done with the three OOF lines when finished.
cd "${0:A:h}/.."
K=20 EXTRA=1 AUX=1 TEACH=1 CROSS=1 BLOCK=1 THREADS=6 uv run python models/v4_catboost.py 42 >> logs/v4_cat_aux_block_k20.log 2>&1 &
K=20 EXTRA=1 AUX=1 TEACH=1 THREADS=4 uv run python models/v5_xgb.py 42 >> logs/v5_xgb_aux_k20.log 2>&1 &
wait
TEACH=1 AUX=1 K=20 THREADS=4 uv run python models/v3b_realmlp.py 6 8 >> logs/v3b_aux_k20.log 2>&1
echo "$(date) cat: $(grep '^OOF' logs/v4_cat_aux_block_k20.log) | xgb: $(grep '^OOF' logs/v5_xgb_aux_k20.log) | realmlp: $(grep '^OOF' logs/v3b_aux_k20.log)" > logs/k20_winners.done
