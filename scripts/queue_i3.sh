#!/bin/zsh
# queue_i3.sh: one detached queue for the i3 ideas, run strictly sequentially after A.
#   B soft labels (leak-free teachers), C 5th teacher (CatBoost cats), D block target-encoding,
#   E refit of hedge files, F TabM (promoted only by the blend-gain gate).
# Never submits to Kaggle. Re-runnable: runs whose logs/<name>.json exists are skipped.
# Kill switch: touch logs/queue_i3.stop (skips the remaining training; blends and report still run).
# Launch once merged: nohup zsh scripts/queue_i3.sh >/dev/null 2>&1 & disown
cd "${0:A:h}/.."
L=logs/queue_i3.log
PY=.venv/bin/python        # never uv run: a signal must reach python itself
TABM_K5_SECS=14400 TABM_K20_SECS=43200
A_CAT=v4_cat_s42_k20_extra_aux_teach_cross_block
A_MLP=v3b_realmlp_e6_n8_s42_k20_aux_teach
XB=v5_xgb_d8_s42_extra_aux_teach        # same-frame k5 baselines
MB=v3b_realmlp_e6_n8_s42_aux_teach
XF=(EXTRA=1 AUX=1 TEACH=1)
MF=(AUX=1 TEACH=1)
PAIRS=(01 02 03 04 12 13 14 23 24 34)
SINGLES=(0 1 2 3 4)
DROP=()
bx=0 bm=0

qlog() { print -r -- "$(date '+%F %T') $*" >> $L; }
have() { [[ -f logs/$1.json ]]; }
wait_render() { while pgrep -f scripts/overnight.py >/dev/null; do sleep 60; done; }
# run_A.done is not enough: a relaunched A or an orphan from a killed queue can still be training. Machine-wide on purpose (RAM is).
BUSY='logs/run_A\.sh|[Pp]ython[0-9.]* .*models/[a-z0-9_]+\.py'
wait_busy() { local n=0; while pgrep -f "$BUSY" >/dev/null; do (( n++ )) || qlog "waiting: training process alive"; sleep 60; done; }

# step <stem> [VAR=v ...] cmd...   logs to logs/<stem>.log
step() {
  wait_busy
  local stem=$1; shift
  if [[ -e logs/queue_i3.stop ]]; then qlog "SKIP $stem (stop file)"; return 1; fi
  qlog "start $stem"
  env "$@" >> logs/$stem.log 2>&1
  local rc=$?
  if (( rc == 0 )); then qlog "ok $stem"; return 0; fi
  qlog "FAILED $stem rc=$rc: $(tail -n 1 logs/$stem.log)"
  return 1
}
# tstep <secs> <stem> ...: python is killed by alarm after <secs> (rc 142)
tstep() {
  local secs=$1; shift
  step "$1" perl -e 'alarm shift; exec @ARGV or die' $secs env "${@:2}"
}
# gate <cand> <base>: fold gate, DROP+=cand on any non-zero exit
gate() {
  $PY scripts/gate.py $1 $2 >> $L 2>&1
  local rc=$?
  (( rc == 0 )) || DROP+=($1)
  return $rc
}
# bgate <cand> <base>: blend gate on logs/<n>.json; the caller decides what to drop
bgate() { $PY scripts/gate.py --blend logs/$1.json logs/$2.json >> $L 2>&1; }
# run <name> <stem> ...: runs unless logs/<name>.json exists
run() { local n=$1; shift; have $n || step "$@"; }
# refit <name> <stem> ...: full-data refit for hedge, only if the k20 run exists and no hedge file yet
refit() {
  local n=$1 stem=$2; shift 2
  have $n || return 0
  [[ -f subs/hedge/${n}_hedge_test.npy ]] && return 0
  step $stem "$@"
}
# teach <fam> <S>...: leave-S teacher predictions in data/soft/<fam>_<S>.npy
teach() {
  local fam=$1 S; shift
  for S in "$@"; do
    [[ -f data/soft/${fam}_${S}.npy ]] && continue
    if [[ $fam == xgb ]]; then
      step b_teach_xgb $XF LEAVE=$S THREADS=8 $PY models/v5_xgb.py || return 1
    else
      wait_render
      step b_teach_mlp $MF LEAVE=$S $PY models/v3b_realmlp.py 6 8 42 || return 1
    fi
  done
  return 0
}

# Single instance: flock on logs/queue_i3.lock, released when the process dies.
: >> logs/queue_i3.lock
zmodload zsh/system
zsystem flock -t 0 logs/queue_i3.lock 2>/dev/null || { print -r -- "$(date '+%F %T') another queue holds logs/queue_i3.lock: exit" >> $L; exit 1; }

# Nothing the launching shell exports reaches a step: each step sets what it needs.
unset K SMOKE LEAVE SOFT BLKTE REFIT TEACH EXTRA AUX HAM CROSS BLOCK DEVICE

# (0) Wait for A: logs/run_A.done only, polled every 5 min. A stop file during the wait ends the queue with nothing run.
qlog "queue start"
until [[ -f logs/run_A.done ]]; do
  if [[ -e logs/queue_i3.stop ]]; then qlog "stop file before A finished: exit"; exit 0; fi
  sleep 300
done
wait_busy
qlog "A finished: cat $(have $A_CAT && echo ok || echo MISSING), realmlp $(have $A_MLP && echo ok || echo MISSING)"

# (B) soft labels. Teachers first (pairs, then singles for K=20), then the k5 students.
if teach xgb $PAIRS && teach mlp $PAIRS; then
  run ${XB}_soft b_soft_xgb $XF SOFT=1 THREADS=8 $PY models/v5_xgb.py
  gate ${XB}_soft $XB && bx=1
  wait_render
  run ${MB}_soft b_soft_mlp $MF SOFT=1 $PY models/v3b_realmlp.py 6 8 42
  gate ${MB}_soft $MB && bm=1
  if (( bx || bm )); then
    if teach xgb $SINGLES && teach mlp $SINGLES; then
      (( bx )) && run v5_xgb_d8_s42_k20_extra_aux_teach_soft b_k20_xgb_soft K=20 $XF SOFT=1 THREADS=8 $PY models/v5_xgb.py
      if (( bm )); then
        wait_render
        run v3b_realmlp_e6_n8_s42_k20_aux_teach_soft b_k20_mlp_soft K=20 $MF SOFT=1 $PY models/v3b_realmlp.py 6 8 42
      fi
    else
      qlog "B singles incomplete: no K=20 B"
    fi
  fi
else
  qlog "B teachers incomplete: B dropped"; DROP+=(_soft)
fi

# (C) 5th teacher (CatBoost on cats, separate process), then the k5 student and its K=20 if it passes.
if [[ ! -f data/feat_teacher_cat2.parquet ]]; then step c_teacher THREADS=8 $PY models/teacher_cat.py; fi
if [[ -f data/feat_teacher_cat2.parquet ]]; then
  run ${XB}3 c_teach3 EXTRA=1 AUX=1 TEACH=3 THREADS=8 $PY models/v5_xgb.py
  if gate ${XB}3 $XB; then
    run v5_xgb_d8_s42_k20_extra_aux_teach3 c_k20 K=20 EXTRA=1 AUX=1 TEACH=3 THREADS=8 $PY models/v5_xgb.py
  fi
else
  qlog "C teacher missing: C dropped"; DROP+=(_teach3)
fi

# (D) block target-encoding, XGB then RealMLP, each with K=20 on a pass.
run ${XB}_blkte d_xgb $XF BLKTE=1 THREADS=8 $PY models/v5_xgb.py
if gate ${XB}_blkte $XB; then
  run v5_xgb_d8_s42_k20_extra_aux_teach_blkte d_k20_xgb K=20 $XF BLKTE=1 THREADS=8 $PY models/v5_xgb.py
fi
wait_render
run ${MB}_blkte d_mlp $MF BLKTE=1 $PY models/v3b_realmlp.py 6 8 42
if gate ${MB}_blkte $MB; then
  run v3b_realmlp_e6_n8_s42_k20_aux_teach_blkte d_k20_mlp K=20 $MF BLKTE=1 $PY models/v3b_realmlp.py 6 8 42
fi

# (E) full-data refits for the hedge files. Soft and TabM runs are not refit: hedge.py uses their fold-average.
refit $A_CAT e_refit_cat K=20 EXTRA=1 AUX=1 TEACH=1 CROSS=1 BLOCK=1 REFIT=1 THREADS=8 $PY models/v4_catboost.py 42
refit v5_xgb_d8_s42_k20_extra_aux_teach e_refit_xgb K=20 $XF REFIT=1 $PY models/v5_xgb.py
refit v5_xgb_d8_s42_k20_extra_aux_teach3 e_refit_xgb3 K=20 EXTRA=1 AUX=1 TEACH=3 REFIT=1 $PY models/v5_xgb.py
refit v5_xgb_d8_s42_k20_extra_aux_teach_blkte e_refit_xgb_blkte K=20 $XF BLKTE=1 REFIT=1 $PY models/v5_xgb.py
wait_render
refit $A_MLP e_refit_mlp K=20 $MF REFIT=1 $PY models/v3b_realmlp.py 6 8 42
refit v3b_realmlp_e6_n8_s42_k20_aux_teach_blkte e_refit_mlp_blkte K=20 $MF BLKTE=1 REFIT=1 $PY models/v3b_realmlp.py 6 8 42

# (F) TabM. Preflight on SMOKE rows, k5 run, then a blend-gain gate (q3_f vs q3_base) decides K=20.
wait_render
if step f_preflight SMOKE=2000 $PY models/v8_tabm.py; then
  if have v8_tabm_s42_aux_teach || { wait_render; tstep $TABM_K5_SECS f_tabm_k5 $PY models/v8_tabm.py; }; then
    rm -f logs/q3_base.json logs/q3_f.json  # a failed blend must not leave the last run's JSON for the gate
    $PY blend/run_blend.py q3_base --own-only --drop '_soft|_teach3|_blkte|v8_tabm' > logs/q3_base.log 2>&1; rc1=$?
    $PY blend/run_blend.py q3_f --own-only --drop '_soft|_teach3|_blkte|v8_tabm_s42_k20' > logs/q3_f.log 2>&1; rc2=$?
    if (( rc1 || rc2 )); then
      qlog "F blend failed: F dropped"; DROP+=(v8_tabm)
    elif bgate q3_f q3_base; then
      wait_render
      have v8_tabm_s42_k20_aux_teach || tstep $TABM_K20_SECS f_tabm_k20 K=20 $PY models/v8_tabm.py
    else
      DROP+=(v8_tabm)
    fi
  else
    qlog "F k5 run failed: F dropped"; DROP+=(v8_tabm)
  fi
else
  qlog "F preflight failed: F dropped"; DROP+=(v8_tabm)
fi

# Blends: these don't go through step, so they run even when the stop file exists.
NEW='_soft|_teach3|_blkte|v8_tabm'
wait_busy
rm -f logs/q3_base.json logs/q3_all.json
$PY blend/run_blend.py q3_base --own-only --drop "$NEW" > logs/q3_base.log 2>&1 &
if (( $#DROP )); then
  $PY blend/run_blend.py q3_all --own-only --drop "${(j:|:)DROP}" > logs/q3_all.log 2>&1 &
else
  $PY blend/run_blend.py q3_all --own-only > logs/q3_all.log 2>&1 &
fi
wait
[[ -f logs/q3_all.json ]] && $PY blend/hedge.py q3_all q3_all_hedge >> $L 2>&1

$PY scripts/report_i3.py >> $L 2>&1
qlog "queue done"
