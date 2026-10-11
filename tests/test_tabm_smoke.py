"""F: TabM script (models/v8_tabm.py) on CPU. RUN_SMOKE=1 only."""
import json


def test_tabm_smoke_five_folds(smoke_rig):
    smoke_rig.run("v8_tabm.py", SMOKE=2000)
    J = json.load(open(smoke_rig.root / "smoke" / "logs" / "v8_tabm_s42_aux_teach.json"))
    assert len(J["folds"]) == 5 and "oof" in J and "params" in J
    assert not (smoke_rig.root / "oof").exists()
