#!/bin/zsh
# Wait for the Applied Pipeline render (scripts/overnight.py) to exit, then run 20-fold RealMLP + own-only blend. Never submits.
cd "${0:A:h}/.."; L=logs/overnight_realmlp.log
while pgrep -f scripts/overnight.py >/dev/null; do
  [[ $(date +%H%M) > 0615 && $(date +%H%M) < 1200 ]] && { echo "$(date) render still running at 06:15, not starting" >> $L; exit 0; }
  sleep 60
done
echo "$(date) render gone, starting RealMLP k20" >> $L
K=20 TEACH=1 uv run python models/v3b_realmlp.py > logs/v3b_teach_k20.log 2>&1 || { echo "$(date) realmlp failed" >> $L; exit 1; }
uv run python blend/run_blend.py b17_own --own-only > logs/b17_own.log 2>&1
echo "$(date) $(grep -E 'BLEND OOF|split-half' logs/b17_own.log | tr '\n' ' ') (b15 halves 0.961264 0.961925; not submitted)" > logs/overnight_realmlp.done
