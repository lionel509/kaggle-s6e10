# Kaggle Playground S6E10: Predicting Airline Satisfaction

[Playground Series Season 6, Episode 10](https://www.kaggle.com/competitions/playground-series-s6e10): a binary `satisfaction` target scored on ROC-AUC. It ran 2026-10-01 to 2026-10-31.

**Final: private 0.96234, rank 496 / 3,747.** The two final picks:

| Pick | What | OOF | Public | Private |
|---|---|---|---|---|
| `b28` | nested L2 logistic regression on logits over our own members, Chris Deotte's 11 TFM OOFs and ~140 public OOF members | 0.962149 | 0.96174 | 0.96231 |
| `hedge_0.5` | b27 rank-averaged 50/50 with a public LB-tuned blend | n/a | 0.96196 | 0.96234 |

Our own models alone (b26) scored OOF 0.961857 and private 0.96201.

## What's here

- `analysis/`: the day-one data analysis (a01–a05). Columns against the original dataset, near-copy Hamming distances, pair interactions, residuals, teacher rankings.
- `models/`: LightGBM (`v1`, `v2` init-score), RealMLP (`v3b`), CatBoost (`v4`), XGBoost (`v5`), simple LR/kNN (`v6`), TabM and TFM runners. `features.py` holds the shared frame, which includes:
  - teacher logits from models trained on the original dataset
  - a per-Flight-Distance route profile with nested target encoding
  - rating crosses
  - label-free aux-task features
  - fold assignment: `StratifiedKFold(5, shuffle, 42)`, run as k=5, 10 and 20
- `blend/`: the fold-rank probit coordinate-descent blender and nested-by-fold stackers.
- `scripts/`: the submission gate (`gate.py`: split-half rule, then paired DeLong ≥ 2 sd on full OOF), `delong.py`, finals selection and the TFM/public-member stacks.
- `logs/`: every run's log and JSON summary.
- `tests/`: pytest for the blender, stacker, features and queue runner.

Data, OOF arrays and submissions are not in the repo. To rebuild, run `kaggle competitions download -c playground-series-s6e10`, then `uv sync`, then the model scripts.

## Post-mortem

- **CV held.** Private followed OOF almost exactly, run after run, while the public board sat flat. Choosing on OOF/DeLong was right, and the public top 6 all dropped on private.
- **The private board was a plateau.** Ranks 50–400 sit within 0.96236–0.96239, and rank 496 is only 0.00004 below rank 100. Stacking the public OOF pool puts you on it.
- **What got teams off the plateau was a generative model of the data's writer.** The rows were written by GReaT: GPT-2 fine-tuned on the original dataset, label first. The 1st, 4th and 7th place solutions scored P(row | label) with a GPT-2 or any-order transformer and applied Bayes' rule. Our day-one GPT-2 check only tested token-prefix *features* of Flight Distance (leave-one-out corr ≤ 0.10, correctly negative). It never tested a row-level generative classifier.
- **Every transformer family needed a GPU, and this run was CPU/MPS-only on a 16 GB laptop.**
