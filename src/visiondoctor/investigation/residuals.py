"""Residual evaluators, written in advance for the tests a template declares.

Each evaluator reads the conventional fields of the records bound to its
measurements and returns numbers against the test's threshold.  It runs only
after structural analysis has admitted the test; it names no cause.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from visiondoctor.geometry.transforms import (
    compose,
    invert,
    rotation_error_rad,
    translation_error_m,
)

from .transform_check import _matrix

Documents = dict[str, dict[str, Any]]


def _field(documents: Documents, measurement: str, key: str) -> Any:
    value = documents[measurement].get(key)
    if value is None:
        raise ValueError(f"{measurement} 记录中没有 {key}")
    return value


def _pose_gap(left, right) -> dict[str, float]:
    return {
        "position_m": round(translation_error_m(left, right), 9),
        "rotation_rad": round(rotation_error_rad(left, right), 9),
    }


def _target(documents: Documents):
    """The TCP the declared chain asks for: camera_to_base · detection · pick offset."""

    return compose(
        compose(_matrix(_field(documents, "calibration", "camera_to_base")),
                _matrix(_field(documents, "detection", "detected_part_in_camera"))),
        _matrix(_field(documents, "calibration", "pick_offset_from_part")),
    )


def command_consistency(documents: Documents) -> dict[str, Any]:
    tool = _matrix(_field(documents, "tool", "tool0_to_tcp"))
    expected = compose(_target(documents), invert(tool))
    command = _matrix(_field(documents, "command", "commanded_flange_base"))
    return {
        "residual": _pose_gap(command, expected),
        "reading": {
            "command_minus_expected_flange_m": [
                round(float(value), 9) for value in command[:3, 3] - expected[:3, 3]
            ],
        },
    }


def motion_tracking(documents: Documents) -> dict[str, Any]:
    command = _matrix(_field(documents, "command", "commanded_flange_base"))
    actual = _matrix(_field(documents, "motion", "actual_flange_base"))
    return {"residual": _pose_gap(actual, command), "reading": {}}


def outcome_consistency(documents: Documents) -> dict[str, Any]:
    tool = _matrix(_field(documents, "tool", "tool0_to_tcp"))
    landed = compose(_matrix(_field(documents, "motion", "actual_flange_base")), tool)
    target = _target(documents)
    predicted = _pose_gap(landed, target)
    measured = {
        "position_m": float(_field(documents, "result", "position_error_m")),
        "rotation_rad": float(_field(documents, "result", "rotation_error_rad")),
    }
    return {
        "residual": {key: round(abs(measured[key] - predicted[key]), 9) for key in measured},
        "reading": {"measured": measured, "predicted_from_records": predicted,
                    "limitation": "到位误差是标量，只比较大小；方向不同而大小相同的偏差不会被发现"},
    }


def frame_correspondence(documents: Documents) -> dict[str, Any]:
    detection, capture = documents["detection"], documents["capture"]
    used = _field(documents, "detection", "capture_id")
    now = _field(documents, "capture", "capture_id")
    stamp_used = detection.get("source_stamp_s")
    stamp_now = (capture.get("sensor_stamps_s") or {}).get("rgb")
    comparable = (
        bool(detection.get("clock_domain"))
        and detection.get("clock_domain") == capture.get("clock_domain")
        and isinstance(stamp_used, (int, float)) and isinstance(stamp_now, (int, float))
    )
    residual: dict[str, float] = {"identity_mismatch": float(used != now)}
    if comparable:
        residual["age_s"] = round(abs(float(stamp_now) - float(stamp_used)), 6)
    return {
        "residual": residual,
        "reading": {
            "capture_used": used, "capture_now": now,
            "stamp_used_s": stamp_used, "stamp_now_s": stamp_now,
            "clock_comparable": comparable,
        },
    }


def _roi_ratio(depth: Any, roi: list[int]) -> float:
    x0, y0, x1, y1 = (int(value) for value in roi)
    region = np.asarray(depth, dtype=float)[y0:y1, x0:x1]
    if not region.size:
        raise ValueError("roi_xyxy 在深度数组之外")
    return round(float((np.isfinite(region) & (region > 0)).mean()), 6)


def depth_handoff(documents: Documents) -> dict[str, Any]:
    roi = _field(documents, "detection", "roi_xyxy")
    raw = _roi_ratio(documents["raw_depth"], roi)
    consumed = _roi_ratio(documents["consumed_depth"], roi)
    return {
        "residual": {"ratio_gap": round(abs(raw - consumed), 6)},
        "reading": {"roi_xyxy": roi, "raw_valid_ratio": raw, "consumed_valid_ratio": consumed,
                    "limitation": "矩形区域可能含背景；有效比例不证明深度数值准确"},
    }


def perception_rule(documents: Documents) -> dict[str, Any]:
    roi = _field(documents, "detection", "roi_xyxy")
    threshold = float(_field(documents, "perception_config", "minimum_target_depth_ratio"))
    consumed = _roi_ratio(documents["consumed_depth"], roi)
    emitted = _field(documents, "detection", "status") == "detected"
    return {
        "residual": {"rule_mismatch": float(emitted != (consumed >= threshold))},
        "reading": {"consumed_valid_ratio": consumed, "declared_minimum": threshold,
                    "perception_status": documents["detection"]["status"],
                    "self_reported_ratio": documents["detection"].get("target_depth_valid_ratio"),
                    "limitation": "用矩形区域近似感知的目标掩膜，临界比例附近可能误判"},
    }


def raw_measurable(documents: Documents) -> dict[str, Any]:
    roi = _field(documents, "detection", "roi_xyxy")
    raw = _roi_ratio(documents["raw_depth"], roi)
    reference = _roi_ratio(documents["reference_depth"], roi)
    return {
        "residual": {"ratio_gap": round(abs(raw - reference), 6)},
        "reading": {"raw_valid_ratio": raw, "reference_valid_ratio": reference,
                    "limitation": "假定参考运行中工件位于同一区域、表面与光照条件相同"},
    }


EVALUATORS: dict[str, Callable[[Documents], dict[str, Any]]] = {
    "command_consistency": command_consistency,
    "motion_tracking": motion_tracking,
    "outcome_consistency": outcome_consistency,
    "frame_correspondence": frame_correspondence,
    "depth_handoff": depth_handoff,
    "perception_rule": perception_rule,
    "raw_measurable": raw_measurable,
}


def evaluate(name: str, threshold: dict[str, float], documents: Documents) -> dict[str, Any]:
    """Numbers and a pass or fail against the threshold; an identity mismatch always fails."""

    outcome = EVALUATORS[name](documents)
    residual = outcome["residual"]
    violated = [
        key for key, value in residual.items()
        if (key in {"identity_mismatch", "rule_mismatch"} and value)
        or (key in threshold and value > threshold[key])
    ]
    return {**outcome, "status": "fail" if violated else "pass", "violated": violated}
