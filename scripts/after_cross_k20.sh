#!/bin/zsh
# Wait for the 20-fold cross CatBoost, then own-only blend b22. Never submits.
cd "${0:A:h}/.."
while pgrep -f "models/v4_catboost.py" >/dev/null; do sleep 60; done
grep -q "^OOF" logs/v4_cat_cross_k20.log || { echo "$(date) k20 cross died: $(tail -1 logs/v4_cat_cross_k20.log)" > logs/b22.done; exit 1; }
uv run python blend/run_blend.py b22_own --own-only > logs/b22_own.log 2>&1
echo "$(date) $(grep '^OOF' logs/v4_cat_cross_k20.log); $(grep -E 'BLEND OOF|split-half' logs/b22_own.log | tr '\n' ' ') (b21 halves 0.961338 0.961975)" > logs/b22.done
