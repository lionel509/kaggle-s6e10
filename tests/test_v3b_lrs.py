"""v3b_realmlp.py LRS switch: flat_anneal gets its own name suffix; unset keeps every existing name and param."""
import json
import pytest


def test_lrs_flat_anneal_is_its_own_name(smoke_rig, monkeypatch):
    monkeypatch.delenv("LRS", raising=False)
    smoke_rig.run("v3b_realmlp.py", 1, 1, SMOKE=2000, DEVICE="cpu", AUX=1, TEACH=1, LRS="flat_anneal")
    J = json.load(open(smoke_rig.root / "smoke" / "logs" / "v3b_realmlp_e1_n1_s42_aux_teach_fa.json"))
    assert J["params"]["lr_sched"] == "flat_anneal"


def test_lrs_unset_keeps_default_name_and_schedule(smoke_rig, monkeypatch):
    monkeypatch.delenv("LRS", raising=False)
    smoke_rig.run("v3b_realmlp.py", 1, 1, SMOKE=2000, DEVICE="cpu", AUX=1, TEACH=1)
    J = json.load(open(smoke_rig.root / "smoke" / "logs" / "v3b_realmlp_e1_n1_s42_aux_teach.json"))
    assert J["params"]["lr_sched"] == "lin_cos_log_15"
    assert not (smoke_rig.root / "smoke" / "logs" / "v3b_realmlp_e1_n1_s42_aux_teach_fa.json").exists()


def test_lrs_unknown_value_fails(smoke_rig, monkeypatch):
    monkeypatch.delenv("LRS", raising=False)
    # the LRS assert is the last line of the child traceback; a crash elsewhere (e.g. an OpenMP abort) must not match
    with pytest.raises(AssertionError, match="AssertionError: bogus"):
        smoke_rig.run("v3b_realmlp.py", 1, 1, SMOKE=2000, DEVICE="cpu", AUX=1, TEACH=1, LRS="bogus")
