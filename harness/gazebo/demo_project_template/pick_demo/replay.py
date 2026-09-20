from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .pipeline import derive_desired_tcp, flange_command_for_tcp


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _pose(value: Any, name: str) -> dict[str, list[float]]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    position = value.get("position")
    quaternion = value.get("quaternion_xyzw")
    if not isinstance(position, list) or not isinstance(quaternion, list):
        raise ValueError(f"{name} is not a pose")
    if len(position) != 3 or len(quaternion) != 4:
        raise ValueError(f"{name} has invalid dimensions")
    return {
        "position": [float(item) for item in position],
        "quaternion_xyzw": [float(item) for item in quaternion],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(input_path: Path, output_path: Path, log_path: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    calibration_path = root / "config" / "cell_calibration.yaml"
    tool_path = root / "config" / "tool_profile.yaml"
    source = json.loads(input_path.read_text(encoding="utf-8"))
    calibration = _read_yaml(calibration_path)
    tool = _read_yaml(tool_path)
    detected_part = _pose(source.get("detected_part_in_camera"), "detected_part_in_camera")
    desired_tcp = derive_desired_tcp(
        detected_part,
        _pose(calibration.get("camera_to_base"), "camera_to_base"),
        _pose(calibration.get("pick_offset_from_part"), "pick_offset_from_part"),
    )
    command = flange_command_for_tcp(
        desired_tcp,
        _pose(tool.get("tool0_to_tcp"), "tool0_to_tcp"),
    )
    output = {
        "schema_version": "pick-a17-command/v1",
        "run_id": str(source.get("run_id", "unknown")),
        "part_id": str(source.get("part_id", "unknown")),
        "detection_id": str(source.get("detection_id", "unknown")),
        "capture_id": source.get("capture_id"),
        "commanded_flange_base": command,
        "config_hashes": {
            "cell_calibration.yaml": _sha256(calibration_path),
            "tool_profile.yaml": _sha256(tool_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": "flange_command_published",
        "run_id": output["run_id"],
        "part_id": output["part_id"],
        "detection_id": output["detection_id"],
        "calibration_sha256": output["config_hashes"]["cell_calibration.yaml"],
        "tool_profile_sha256": output["config_hashes"]["tool_profile.yaml"],
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay one PICK-A17 vision-to-robot command")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    replay(args.input, args.output, args.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
