"""A repair follow-up must read the requested run, including duplicate paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.gazebo.controller import PickCellController
from tests.support import ProtocolDoubleGateway, _turn
from tests.test_grasp_investigation import _answer
from tests.test_grasp_scenarios import capture_at
from visiondoctor.case import CaseService
from visiondoctor.investigation import Toolbox


def observed_case(tmp_path: Path) -> tuple[CaseService, str]:
    service = CaseService(tmp_path / "cases")
    case_id = service.create("修复前后同名文件")
    cell = PickCellController(runtime_root=tmp_path / "cell")
    for run_id, success in (("before", False), ("after", True)):
        root = tmp_path / run_id
        capture_at(root)
        command = root / "parts/A/algorithm/output.json"
        command.parent.mkdir()
        command.write_text(json.dumps({"run_id": run_id, "z": 0.183 if success else 0.253}))
        cell._write_bundle(root, run_id, {}, [], [
            {"part_id": "A", "success": success,
             "classification": "within_tolerance" if success else "missed_pick_pose"},
        ])
    service.attach_observation(case_id, tmp_path / "before")
    service.run_recheck(case_id, tmp_path / "after")
    return CaseService(service.root), case_id


def test_reloaded_followup_reads_each_runs_commands_results_and_images(tmp_path, monkeypatch):
    import visiondoctor.case.service as service_module

    service, case_id = observed_case(tmp_path)
    record = service.record(case_id)
    selected = [e for e in record.case.evidence if e.reference in {
        "parts/A/algorithm/output.json", "results/A", "parts/A/capture/rgb.png",
    }]
    seen_paths = []

    class Vision:
        def assess(self, path, **kwargs):
            seen_paths.append(path)
            return {"visible": "same view, separate capture"}

    gateway = ProtocolDoubleGateway([
        _turn("read_evidence", {"evidence_ids": [e.evidence_id for e in selected]}, 1),
        _answer({"findings": [], "hypotheses": [], "next_step": "只确认两次到位结果不同。"}),
    ])
    monkeypatch.setattr(service_module, "_model_gateway", lambda: gateway)
    monkeypatch.setattr(service_module, "_vision_gateway", Vision)
    record.running = {"calls": []}
    service._work(case_id, "比较修复前后")
    assert record.turns, record.messages
    delivered = next(m for m in record.case.transcript if m["role"] == "tool")
    rows = {e["evidence_id"]: e for e in json.loads(delivered["content"])}
    for e in selected:
        row = rows[e.evidence_id]
        assert row["bundle_id"] == e.bundle_id
        if e.reference.endswith("output.json"):
            assert json.loads(row["text"])["run_id"] == e.bundle_id
        elif e.reference == "results/A":
            assert json.loads(row["text"])["success"] == (e.bundle_id == "after")
    assert seen_paths == [tmp_path / run / "parts/A/capture/rgb.png"
                          for run in ("before", "after")]
    assert {e["bundle_id"] for e in service.view(case_id)["evidence"]} == {"before", "after"}


def test_missing_observation_adapter_refuses_instead_of_reading_another_run(tmp_path):
    service, case_id = observed_case(tmp_path)
    record = service.record(case_id)
    evidence = next(e for e in record.case.evidence if e.bundle_id == "after"
                    and e.reference.endswith("output.json"))
    toolbox = Toolbox(record.case, record.adapter, record.bundle)
    with pytest.raises(ValueError, match="observation source"):
        toolbox.read_evidence([evidence.evidence_id])
    assert evidence.evidence_id not in record.case.examined


def test_measurement_keeps_the_measured_run_and_cross_run_sources(tmp_path):
    service, case_id = observed_case(tmp_path)
    record = service.record(case_id)
    refs = {(e.bundle_id, e.reference): e.evidence_id for e in record.case.evidence}
    toolbox = Toolbox(record.case, record.adapter, record.bundle,
                      observation_dirs=tuple(Path(p) for p in record.observation_dirs))
    args = {"rgb_evidence_id": refs["after", "parts/A/capture/rgb.png"],
            "depth_evidence_id": refs["after", "parts/A/capture/depth.npy"],
            "camera_evidence_id": refs["after", "parts/A/capture/camera-info.json"],
            "roi_xyxy": [323, 274, 356, 300]}
    measured = toolbox.measure_rgbd_region(**args)
    assert toolbox._evidence(measured["evidence_id"]).bundle_id == "after"
    mixed = toolbox.measure_rgbd_region(**{
        **args, "camera_evidence_id": refs["before", "parts/A/capture/camera-info.json"],
    })
    assert toolbox._evidence(mixed["evidence_id"]).bundle_id == f"case:{case_id}"
    assert refs["before", "parts/A/capture/camera-info.json"] in mixed["sources"]


def test_runs_are_compared_by_content_at_the_software_layer(tmp_path):
    service, case_id = observed_case(tmp_path)
    record = service.record(case_id)
    toolbox = Toolbox(record.case, record.adapter, record.bundle,
                      observation_dirs=tuple(Path(p) for p in record.observation_dirs))
    compared = toolbox.compare_runs(baseline_run_id="before", run_id="after")
    assert compared["layer"] == "software"
    row = next(item for item in compared["records"]
               if item["reference"] == "parts/A/algorithm/output.json")
    assert not row["same"] and not row["byte_identical"]
    assert row["differences"] == [{"field": "z", "baseline": 0.253, "candidate": 0.183,
                                   "delta": -0.07}]
    assert row["identity_fields_omitted"] == 1
