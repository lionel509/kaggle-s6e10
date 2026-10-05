#!/bin/zsh
# LightGBM + extra + teachers: k5 gate (must beat v1 0.960971 by +0.00003), then k20, then own-only blend. Never submits.
cd "${0:A:h}/.."; L=logs/overnight_lgb.log
echo "$(date) start k5" >> $L
EXTRA=1 TEACH=1 THREADS=8 uv run python models/v1_lgb.py > logs/v1_lgb_extra_teach.log 2>&1 || { echo "$(date) k5 failed" >> $L; exit 1; }
o=$(grep -o 'OOF [0-9.]*' logs/v1_lgb_extra_teach.log | cut -d' ' -f2); echo "$(date) k5 OOF $o" >> $L
(( o > 0.961001 )) || { echo "$(date) below bar, stop" > logs/overnight_lgb.done; exit 0; }
K=20 EXTRA=1 TEACH=1 THREADS=8 uv run python models/v1_lgb.py > logs/v1_lgb_k20_extra_teach.log 2>&1 || { echo "$(date) k20 failed" >> $L; exit 1; }
uv run python blend/run_blend.py b20_own --own-only > logs/b20_own.log 2>&1
echo "$(date) k5 $o; $(grep -o 'OOF [0-9.]*' logs/v1_lgb_k20_extra_teach.log) ; $(grep -E 'BLEND OOF|split-half' logs/b20_own.log | tr '\n' ' ') (b19 halves 0.961306 0.961947; not submitted)" > logs/overnight_lgb.done
