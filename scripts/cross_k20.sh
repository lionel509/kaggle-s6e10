#!/bin/zsh
# 20-fold cross CatBoost (resumes from oof/ckpt), then b22 own-only blend via after_cross_k20.sh. Never submits.
cd "${0:A:h}/.."
K=20 EXTRA=1 TEACH=1 CROSS=1 THREADS=${THREADS:-6} uv run python models/v4_catboost.py 42 >> logs/v4_cat_cross_k20.log 2>&1
zsh scripts/after_cross_k20.sh
