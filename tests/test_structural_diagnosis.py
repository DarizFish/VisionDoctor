"""Structural diagnosis: structure decides what can be told apart; the host isolates."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from tests.test_case_walkthrough import BUNDLE, _case
from tests.test_grasp_investigation import _bound_case
from visiondoctor.case import (
    CaseService,
    Segment,
    SegmentFinding,
    SegmentStatus,
    software_localizations,
)
from visiondoctor.case.isolation import node_standing
from visiondoctor.case.templates import DEPTH, FRAME, GEOMETRY
from visiondoctor.geometry.transforms import compose, invert
from visiondoctor.investigation import Investigation, Toolbox, build_view, tools_for
from visiondoctor.investigation.agent import _findings
from visiondoctor.investigation.structure import Model
from visiondoctor.investigation.transform_check import _matrix, _pose
from visiondoctor.web.graph import graph_dot

RECORDS = ("detection", "calibration", "tool", "command", "motion", "result")


def _bindings(references: dict[str, str], part: str = "A") -> dict[str, str]:
    return {
        "detection": references[f"parts/{part}/algorithm/input.json"],
        "calibration": references["project/cell_calibration.yaml"],
        "tool": references["project/tool_profile.yaml"],
        "command": references[f"parts/{part}/algorithm/output.json"],
        "motion": references[f"parts/{part}/motion.json"],
        "result": references[f"parts/{part}/result.json"],
    }


def test_structure_decides_what_the_records_can_tell_apart() -> None:
    geometry = Model([GEOMETRY], RECORDS).summary()
    assert geometry["redundancy"] == 3 and not geometry["undetectable"]
    assert geometry["non_isolable_groups"] == [
        ["acquisition.frame", "calibration", "part.moved", "perception", "planning", "tool"]
    ]
    assert geometry["cannot_isolate_from"]["interface"] == []
    assert geometry["cannot_isolate_from"]["robot"] == []

    # A capture record separates the frame handoff from perception, but not the reverse.
    framed = Model([GEOMETRY, FRAME], (*RECORDS, "capture")).summary()
    assert framed["cannot_isolate_from"]["acquisition.frame"] == []
    assert "acquisition.frame" in framed["cannot_isolate_from"]["perception"]

    # Without a command, motion or outcome record the chain has no redundancy left.
    bare = Model([GEOMETRY], ("calibration", "tool")).summary()
    assert bare["redundancy"] == 0 and "interface" in bare["undetectable"]

    # Raw depth alone cannot say whether the scene was measurable; a reference run can,
    # though it still cannot tell imaging from the part's surface.
    depth = ("detection", "raw_depth", "consumed_depth", "perception_config")
    alone = Model([DEPTH], depth).summary()
    assert set(alone["undetectable"]) == {"imaging.depth", "part.surface"}
    referenced = Model([DEPTH], (*depth, "reference_depth")).summary()
    assert not referenced["undetectable"]
    assert referenced["non_isolable_groups"] == [["imaging.depth", "part.surface"]]


def test_structural_diagnosis_isolates_on_a_recorded_observation() -> None:
    case, adapter, references = _case()
    toolbox = Toolbox(case, adapter, case.observations[0])
    result = toolbox.structural_diagnose(
        template_ids=["geometry_chain"], part_id="A", bindings=_bindings(references),
    )
    statuses = {row["test"]: row["status"] for row in result["tests"]}
    assert statuses == {"A": "fail", "B": "pass", "C": "pass"}
    assert result["isolation"]["conclusion"] == "isolated"
    assert result["isolation"]["candidates"] == ["interface"]
    assert "robot" in result["isolation"]["exonerated"]
    assert result["layer"] == "software" and result["evidence_id"] in case.examined
    assert node_standing(case)["interface"] == "candidate"

    # A later call adds a template; what was bound before is carried forward.
    again = toolbox.structural_diagnose(
        template_ids=["frame_correspondence"], part_id="A",
        bindings={"detection": references["parts/A/algorithm/input.json"]},
    )
    assert again["templates"] == ["geometry_chain", "frame_correspondence"]
    assert set(again["bindings"]) == set(RECORDS)
    frame = next(row for row in again["tests"] if row["test"] == "D")
    assert frame["status"] == "unevaluated" and "capture" in frame["reason"]

    with pytest.raises(ValueError, match="不属于工件 A"):
        toolbox.structural_diagnose(
            template_ids=["geometry_chain"], part_id="A",
            bindings={"command": references["parts/B/algorithm/output.json"]},
        )
    names = {item["function"]["name"] for item in tools_for(case)}
    assert "structural_diagnose" in names
    view = build_view(case, case.observations[0])
    assert view["isolation"][0]["isolation"]["candidates"] == ["interface"]
    assert "fillcolor" in graph_dot(view["graph"], node_standing(case))


def test_a_bound_record_without_the_field_measures_nothing() -> None:
    case, adapter, references = _case()
    bindings = _bindings(references)
    # The perception record carries no flange command, so binding it as one measures nothing.
    bindings["command"] = references["parts/A/algorithm/input.json"]
    result = Toolbox(case, adapter, case.observations[0]).structural_diagnose(
        template_ids=["geometry_chain"], part_id="A", bindings=bindings,
    )
    tests = {row["test"]: row for row in result["tests"]}
    assert tests["A"]["status"] == tests["B"]["status"] == "unevaluated"
    assert "f_cmd" in tests["A"]["reason"] and "command" in result["unusable_bindings"]
    assert "robot" in result["structure"]["cannot_isolate_from"]["interface"]
    assert tests["C"]["status"] == "pass"


def test_the_workbench_shows_the_reading_behind_every_evaluated_check() -> None:
    """A residual the check table cannot phrase leaves an empty cell, which argues nothing."""

    import numpy as np

    from visiondoctor.investigation.residuals import EVALUATORS, evaluate
    from visiondoctor.web.app import _residual_text

    documents = {
        "detection": {"roi_xyxy": [0, 0, 2, 2], "status": "detected", "capture_id": "cap-1"},
        "capture": {"capture_id": "cap-2"},
        "raw_depth": np.ones((4, 4)),
        "consumed_depth": np.zeros((4, 4)),
        "reference_depth": np.ones((4, 4)),
        "perception_config": {"minimum_target_depth_ratio": 0.6},
    }
    shown = {}
    for name in ("depth_handoff", "perception_rule", "raw_measurable"):
        row = evaluate(name, {"ratio_gap": 0.05}, documents)
        shown[name] = _residual_text(row)
    assert shown["depth_handoff"] == "有效比例 原始 1.00 · 消费 0.00"
    assert shown["perception_rule"] == "与声明规则不符：比例 0.00 · 下限 0.60"
    assert shown["raw_measurable"] == "有效比例 原始 1.00 · 参考 1.00"
    # The geometry and frame evaluators are read in the recorded cases; a later one
    # that reports some other key still prints, rather than showing nothing at all.
    assert set(EVALUATORS) >= set(shown)
    assert _residual_text({"residual": {"skew_rad": 0.25}}) == "skew_rad 0.250"


def test_a_verdict_citing_the_isolation_cannot_say_the_opposite() -> None:
    case, adapter, references = _case()
    result = Toolbox(case, adapter, case.observations[0]).structural_diagnose(
        template_ids=["geometry_chain"], part_id="A", bindings=_bindings(references),
    )
    cited = result["evidence_id"]
    with pytest.raises(ValueError, match="唯一候选"):
        _findings({"findings": [{"target_id": "interface", "status": "cleared",
                                 "note": "指令没问题", "evidence_ids": [cited]}]}, case)
    with pytest.raises(ValueError, match="核算排除"):
        _findings({"findings": [{"target_id": "robot", "status": "suspect",
                                 "note": "机器人没到位", "evidence_ids": [cited]}]}, case)
    kept = _findings({"findings": [{"target_id": "interface", "status": "suspect",
                                    "note": "检验 A 违反", "evidence_ids": [cited]}]}, case)
    assert kept[0].status is SegmentStatus.SUSPECT


def _repaired_copy(source: Path, target: Path, observed_from: datetime) -> Path:
    """The fixture run as it would look after a correct fix: same records, commands now right."""

    shutil.copytree(source, target)
    manifest = json.loads((target / "bundle.json").read_text(encoding="utf-8"))
    calibration = yaml.safe_load((target / "project/cell_calibration.yaml").read_text("utf-8"))
    tool = yaml.safe_load((target / "project/tool_profile.yaml").read_text(encoding="utf-8"))
    for part in ("A", "B"):
        folder = target / "parts" / part
        detection = json.loads((folder / "algorithm/input.json").read_text(encoding="utf-8"))
        flange = _pose(compose(compose(compose(
            _matrix(calibration["camera_to_base"]), _matrix(detection["detected_part_in_camera"])),
            _matrix(calibration["pick_offset_from_part"])), invert(_matrix(tool["tool0_to_tcp"]))))
        for name, key, value in (
            ("algorithm/output.json", "commanded_flange_base", flange),
            ("motion.json", "actual_flange_base", flange),
            ("result.json", "position_error_m", 0.0),
            ("result.json", "rotation_error_rad", 0.0),
        ):
            record = json.loads((folder / name).read_text(encoding="utf-8"))
            record[key] = value
            (folder / name).write_text(json.dumps(record), encoding="utf-8")
    for artifact in manifest["artifacts"]:
        artifact["sha256"] = hashlib.sha256((target / artifact["path"]).read_bytes()).hexdigest()
    for result in manifest["results"]:
        result.update(success=True, classification="within_tolerance",
                      position_error_m=0.0, rotation_error_rad=0.0)
    manifest.update(
        run_id="run-after-fix",
        observation_started_at=observed_from.isoformat(),
        created_at=(observed_from + timedelta(minutes=5)).isoformat(),
    )
    (target / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    return target


def test_site_recheck_runs_the_same_tests_again_on_the_new_run(tmp_path: Path) -> None:
    service = CaseService(tmp_path / "cases")
    case_id = service.create("A17 工位抓偏")
    service.attach_observation(case_id, BUNDLE)
    record = service.record(case_id)
    references = {item.reference: item.evidence_id for item in record.case.evidence}
    toolbox = Toolbox(record.case, record.adapter, record.bundle)
    for part in ("A", "B"):
        toolbox.structural_diagnose(
            template_ids=["geometry_chain"], part_id=part, bindings=_bindings(references, part),
        )
    applied = service.mark_applied(case_id)
    after = _repaired_copy(BUNDLE, tmp_path / "after", applied + timedelta(minutes=1))

    outcome = service.run_recheck(case_id, after)

    assert outcome["scope"] == "site_recovered"
    rows = {(row["run_id"], row["part_id"]): row for row in service.view(case_id)["isolation"]}
    for part in ("A", "B"):
        run_id = record.bundle.run_id
        before = {test["test"]: test["status"] for test in rows[(run_id, part)]["tests"]}
        again = {test["test"]: test["status"] for test in rows[("run-after-fix", part)]["tests"]}
        assert before == {"A": "fail", "B": "pass", "C": "pass"}
        assert again == {"A": "pass", "B": "pass", "C": "pass"}
        assert rows[("run-after-fix", part)]["isolation"]["conclusion"] == "no_fault_detected"
    # The recheck is a new run; it does not rewrite what was isolated in the run under diagnosis.
    assert node_standing(record.case)["interface"] == "candidate"
    # And it survives a restart, since it is evidence in the case.
    reloaded = CaseService(tmp_path / "cases").view(case_id)["isolation"]
    assert {(row["run_id"], row["part_id"]) for row in reloaded} == set(rows)


def test_an_exonerated_node_does_not_open_the_source_layer(tmp_path: Path) -> None:
    case, adapter, bundle, references = _bound_case(tmp_path)
    toolbox = Toolbox(case, adapter, bundle)
    toolbox.structural_diagnose(
        template_ids=["geometry_chain"], part_id="A", bindings=_bindings(references),
    )
    detection = references["parts/A/algorithm/input.json"]
    turn = Investigation(case, toolbox, "感知输出偏了")
    turn.invoke("read_evidence", evidence_ids=[detection])
    turn.commit(findings=(SegmentFinding(
        segment=Segment.ALGORITHM, target_id="perception", status=SegmentStatus.SUSPECT,
        note="感知记录可疑", evidence_ids=(detection,),
    ),), next_step="看源码")
    assert node_standing(case)["perception"] == "exonerated"
    assert software_localizations(case) == []
