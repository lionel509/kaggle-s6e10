#!/bin/zsh
# Item A of #3: resume k20 CatBoost aux+block+cross from ckpt, then k20 RealMLP aux+teach. Sequential.
cd "${0:A:h}/.."
K=20 EXTRA=1 TEACH=1 AUX=1 CROSS=1 BLOCK=1 THREADS=8 uv run python models/v4_catboost.py >> logs/v4_cat_aux_block_k20.log 2>&1
K=20 TEACH=1 AUX=1 uv run python models/v3b_realmlp.py 6 8 >> logs/v3b_aux_teach_k20.log 2>&1
echo "A done $(date)" >> logs/run_A.done
