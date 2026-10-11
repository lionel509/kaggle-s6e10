"""CPU smoke of the model scripts' new switches. RUN_SMOKE=1 only; each test builds its own rig."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models"))
N = 3000
SETS = ["01", "02", "03", "04", "12", "13", "14", "23", "24", "34", "0", "1", "2", "3", "4"]


def _y_smoke(rig):
    return pd.read_csv(rig.data / "train.csv", usecols=["satisfaction"]).satisfaction.astype(str).eq("True").astype(int).values[:N]


def test_default_v5_writes_smoke_only(smoke_rig):
    smoke_rig.run("v5_xgb.py", 42, 8, EXTRA=1, AUX=1, TEACH=1)
    import json
    J = json.load(open(smoke_rig.root / "smoke" / "logs" / "v5_xgb_d8_s42_extra_aux_teach.json"))
    assert len(J["folds"]) == 5
    assert not (smoke_rig.root / "oof").exists() and not (smoke_rig.root / "logs").exists()


def test_teach3_is_its_own_name(smoke_rig):
    smoke_rig.run("v5_xgb.py", 42, 8, EXTRA=1, AUX=1, TEACH=3)
    assert (smoke_rig.root / "smoke" / "logs" / "v5_xgb_d8_s42_extra_aux_teach3.json").exists()
    assert not (smoke_rig.root / "smoke" / "logs" / "v5_xgb_d8_s42_extra_aux_teach.json").exists()


def test_blkte_names(smoke_rig):
    smoke_rig.run("v5_xgb.py", 42, 8, EXTRA=1, AUX=1, TEACH=1, BLKTE=1)
    smoke_rig.run("v3b_realmlp.py", 1, 1, AUX=1, TEACH=1, BLKTE=1)
    assert (smoke_rig.root / "smoke" / "logs" / "v5_xgb_d8_s42_extra_aux_teach_blkte.json").exists()
    assert (smoke_rig.root / "smoke" / "logs" / "v3b_realmlp_e1_n1_s42_aux_teach_blkte.json").exists()


def test_teacher_cat_writes_smoke_parquet(smoke_rig):
    smoke_rig.run("teacher_cat.py")
    df = pd.read_parquet(smoke_rig.root / "smoke" / "feat_teacher_cat2.parquet")
    assert len(df) == 999479 and not df.isna().any().any()


def test_soft_leave_then_students(smoke_rig):
    import soft
    y = _y_smoke(smoke_rig)
    for S in SETS:
        smoke_rig.run("v5_xgb.py", 42, 8, EXTRA=1, AUX=1, TEACH=1, LEAVE=S)
        smoke_rig.run("v3b_realmlp.py", 1, 1, AUX=1, TEACH=1, LEAVE=S)
        _, b = soft.rows_for(y, S)
        for fam in ("xgb", "mlp"):
            p = np.load(smoke_rig.root / "smoke" / soft.teacher_path(fam, S, "data/soft"))
            assert len(p) == len(b)
    smoke_rig.run("v5_xgb.py", 42, 8, EXTRA=1, AUX=1, TEACH=1, SOFT=1)
    smoke_rig.run("v5_xgb.py", 42, 8, EXTRA=1, AUX=1, TEACH=1, SOFT=1, K=20)
    smoke_rig.run("v3b_realmlp.py", 1, 1, AUX=1, TEACH=1, SOFT=1)
    import json
    L = smoke_rig.root / "smoke" / "logs"
    assert len(json.load(open(L / "v5_xgb_d8_s42_extra_aux_teach_soft.json"))["folds"]) == 5
    J20 = json.load(open(L / "v5_xgb_d8_s42_k20_extra_aux_teach_soft.json"))
    assert len(J20["folds"]) == 20 and J20.get("soft") is True
    assert (L / "v3b_realmlp_e1_n1_s42_aux_teach_soft.json").exists()


def test_refit_writes_hedge_files(smoke_rig):
    smoke_rig.run("v5_xgb.py", 42, 8, EXTRA=1, AUX=1, TEACH=1)
    smoke_rig.run("v5_xgb.py", 42, 8, EXTRA=1, AUX=1, TEACH=1, REFIT=1)
    smoke_rig.run("v4_catboost.py", 42, EXTRA=1)
    smoke_rig.run("v4_catboost.py", 42, EXTRA=1, REFIT=1)
    smoke_rig.run("v3b_realmlp.py", 1, 1, AUX=1, TEACH=1)
    smoke_rig.run("v3b_realmlp.py", 1, 1, AUX=1, TEACH=1, REFIT=1)
    H = smoke_rig.root / "smoke" / "subs" / "hedge"
    for n in ("v5_xgb_d8_s42_extra_aux_teach", "v4_cat_s42_extra", "v3b_realmlp_e1_n1_s42_aux_teach"):
        for kind in ("refit", "hedge"):
            assert len(np.load(H / f"{n}_{kind}_test.npy")) == N // 5
