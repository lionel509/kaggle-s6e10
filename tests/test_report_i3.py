"""scripts/report_i3.py: writes logs/queue_i3_report.md; must never raise on missing inputs."""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _report():
    import report_i3
    return report_i3


def test_empty_root_still_writes_report(tmp_path):
    _report().main(root=str(tmp_path))
    txt = (tmp_path / "logs" / "queue_i3_report.md").read_text()
    assert "MISSING" in txt and "Gates" in txt and "no submission" in txt.lower()


def test_populated_report(tmp_path):
    L = tmp_path / "logs"; L.mkdir(); (tmp_path / "subs").mkdir()
    (L / "queue_i3_gates.jsonl").write_text(json.dumps({
        "cand": "v5_xgb_d8_s42_extra_aux_teach3", "base": "v5_xgb_d8_s42_extra_aux_teach",
        "cand_oof": 0.961300, "base_oof": 0.961266, "delta": 0.000034, "up": 4, "pass": True, "missing": False}) + "\n")
    json.dump({"oof": 0.961376, "weights": {"own:a": 0.5, "own:b": 0.5}}, open(L / "q3_base.json", "w"))
    json.dump({"oof": 0.961788, "weights": {"own:a": 0.5, "own:c": 0.5}}, open(L / "q3_all.json", "w"))
    (L / "q3_base.log").write_text("BLEND OOF 0.961376\n  split-half held-out 0.961376\n  split-half held-out 0.962039\n")
    (L / "q3_all.log").write_text("BLEND OOF 0.961788\n  split-half held-out 0.961405\n  split-half held-out 0.962105\n")
    (L / "queue_i3.log").write_text("2026-10-12 01:00:00 FAILED b_teach_mlp rc=142: killed\n")
    _report().main(root=str(tmp_path))
    txt = (L / "queue_i3_report.md").read_text()
    assert "PROMOTE" in txt
    assert "both up: yes" in txt
    assert "FAILED" in txt and "142" in txt
    assert "no submission" in txt.lower()
