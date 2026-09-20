"""Bounded experiments on one real Gazebo cell; labels never enter product inputs."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import shutil
from pathlib import Path

from .controller import PickCellController

PART_MODELS = {"A": "part_A_flat", "B": "part_B_tilted"}
PART_CENTERS = {"A": [0.5, -0.1, 0.1], "B": [0.5, -0.3, 0.1]}
PANEL_NAME = "experiment_panel"
PANEL_SDF = f'''<sdf version="1.9"><model name="{PANEL_NAME}">
<static>true</static><pose>0.54 -0.24 0.24 0 0 0</pose><link name="panel">
<visual name="surface"><geometry><box><size>0.32 0.46 0.08</size></box></geometry>
<material><ambient>0.85 0.85 0.85 1</ambient><diffuse>0.85 0.85 0.85 1</diffuse></material>
</visual></link></model></sdf>'''


def service(cell: PickCellController, endpoint: str, request_type: str, request: str) -> str:
    command = shlex.join([
        "gz", "service", "-s", f"/world/pick_a17/{endpoint}",
        "--reqtype", request_type, "--reptype", "gz.msgs.Boolean", "--timeout", "5000",
        "--req", request,
    ])
    result = cell._docker("exec", cell.CONTAINER, "bash", "-lc",
                          "source /opt/ros/jazzy/setup.bash && " + command, timeout_s=15)
    if result.returncode != 0 or "data: true" not in result.stdout:
        raise RuntimeError(f"Gazebo {endpoint} failed: {(result.stdout + result.stderr)[-800:]}")
    return result.stdout.strip()


def position_parts(cell: PickCellController, offset_x: float) -> list[dict]:
    records = []
    for part, nominal in PART_CENTERS.items():
        xyz = [nominal[0] + offset_x, *nominal[1:]]
        request = (f'name: "{PART_MODELS[part]}" position {{ x: {xyz[0]} y: {xyz[1]} '
                   f'z: {xyz[2]} }} orientation {{ w: 1 }}')
        response = service(cell, "set_pose/blocking", "gz.msgs.Pose", request)
        records.append({"part": part, "position": xyz, "service_response": response})
    return records


class ExperimentController(PickCellController):
    """Timing injection selects an earlier real capture; scoring follows the moved scene."""

    def __init__(self, earlier_capture: Path | None = None, shift_x: float = 0):
        super().__init__()
        self.earlier_capture = earlier_capture
        self.shift_x = shift_x

    def _private_scoring(self) -> dict:
        scoring = copy.deepcopy(super()._private_scoring())
        for target in scoring["case_targets"].values():
            target["position"][0] += self.shift_x
        return scoring

    def _vision_input(self, workspace, capture_root, input_path, run_id, case_id):
        consumed = capture_root
        if self.earlier_capture is not None:
            consumed = capture_root.parent / "consumed_capture"
            consumed.mkdir()
            for name in ("rgb.png", "depth.npy", "camera-info.json", "capture.json"):
                shutil.copy2(self.earlier_capture / name, consumed / name)
            metadata = self._read_json(consumed / "capture.json")
            metadata["phase"] = "earlier_capture"
            self._write_json(consumed / "capture.json", metadata)
        return super()._vision_input(workspace, consumed, input_path, run_id, case_id)


def run_scene_scenario(kind: str, workspace: Path) -> dict:
    if kind not in {"physical_occlusion", "stale_capture"}:
        raise ValueError(f"unknown scene scenario: {kind}")
    cell = ExperimentController()
    cell._require_running()
    private_record = {"scenario": kind, "interventions": [], "restored": False}
    panel_added = False
    try:
        private_record["interventions"].extend(position_parts(cell, 0))
        if kind == "physical_occlusion":
            response = service(cell, "create/blocking", "gz.msgs.EntityFactory",
                               "sdf: " + json.dumps(PANEL_SDF))
            panel_added = True
            private_record["interventions"].append({"panel_create": response})
        else:
            old = cell.capture()
            cell.earlier_capture = Path(old["capture_dir"])
            cell.shift_x = 0.05
            private_record["interventions"].extend(position_parts(cell, cell.shift_x))
        result = cell.run_cycle(workspace)
        private_record["run_id"] = result["run_id"]
        cell.export_bundle(result["run_id"], cell.runtime_root / "exports")
    finally:
        if panel_added:
            private_record["panel_remove"] = service(
                cell, "remove/blocking", "gz.msgs.Entity", f'name: "{PANEL_NAME}" type: MODEL',
            )
        private_record["restored_positions"] = position_parts(cell, 0)
        private_record["restored"] = True
        label_root = cell.runtime_root / "scenario-labels"
        label_root.mkdir(exist_ok=True)
        key = private_record.get("run_id", cell._new_id("interrupted"))
        cell._write_json(label_root / f"{key}.json", private_record)
    return {**result, "experiment": private_record}


def make_insufficient_control(source: Path, cell: PickCellController | None = None) -> dict:
    """Withhold internal records from a failed cycle; preserve the visible pre-command scene."""
    cell = cell or PickCellController()
    original = cell._read_json(source / "bundle.json")
    if (not original or not original.get("results")
            or any(r["success"] for r in original["results"])):
        raise ValueError("control source must be an actual failed observation")
    run_id = cell._new_id("run")
    root = cell.runtime_root / "runs" / run_id
    selected = ("parts/A/capture/rgb.png", "parts/A/capture/camera-info.json",
                "parts/A/capture/capture.json")
    for relative in selected:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    metadata = cell._read_json(root / selected[2])
    # No depth was supplied in this control. Do not retain a depth summary or
    # paths that could be mistaken for available evidence.
    for key in ("depth_valid_ratio", "depth_streams", "depth_path", "observer_keyframes",
                "observer_clip_path"):
        metadata.pop(key, None)
    metadata["phase"] = "before_command"
    cell._write_json(root / selected[2], metadata)
    results = [{"part_id": "A", "classification": "operator_reported_failure", "success": False,
                "scope": "Reported failure; internal perception and command records not supplied"}]
    cell._write_bundle(root, run_id, {}, [], results)
    exported = cell.export_bundle(run_id, cell.runtime_root / "exports")
    labels = cell.runtime_root / "scenario-labels"
    labels.mkdir(exist_ok=True)
    cell._write_json(labels / f"{run_id}.json", {
        "scenario": "insufficient_evidence", "source_run": original["run_id"],
        "withheld": "depth, perception, command, motion, and program configuration",
    })
    return {"run_id": run_id, "bundle": str(exported)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["physical_occlusion", "stale_capture", "insufficient"])
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.kind == "insufficient":
        if args.source is None:
            parser.error("insufficient requires --source")
        result = make_insufficient_control(args.source)
    else:
        if args.workspace is None:
            parser.error("scene scenarios require --workspace")
        result = run_scene_scenario(args.kind, args.workspace)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
