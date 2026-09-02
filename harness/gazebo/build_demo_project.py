"""Create the demonstration Git bundle without distributing scoring material."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = HARNESS_ROOT.parents[1]
TEMPLATE_ROOT = HARNESS_ROOT / "demo_project_template"
DEFAULT_RUNTIME = REPOSITORY_ROOT / ".runtime" / "gazebo-pick-cell"
DEFAULT_WORKSPACE = DEFAULT_RUNTIME / "projects" / "tabletop-faulty"

# This source stays in the harness process only while a bundle is built.  The
# generated normal revision is for the operator's before/after demonstration;
# no normal source or expected target is put in an ObservationBundle.
_NORMAL_PIPELINE = '''from __future__ import annotations

from .geometry import Pose, compose, inverse


def derive_desired_tcp(
    detected_part_in_camera: Pose,
    camera_to_base: Pose,
    pick_offset_from_part: Pose,
) -> Pose:
    """Turn a camera-frame detection into the desired TCP pick pose."""

    part_in_base = compose(camera_to_base, detected_part_in_camera)
    return compose(part_in_base, pick_offset_from_part)


def flange_command_for_tcp(desired_tcp_in_base: Pose, tool0_to_tcp: Pose) -> Pose:
    """Convert a TCP target into the corresponding robot tool0 target."""

    return compose(desired_tcp_in_base, inverse(tool0_to_tcp))
'''

_PRIVATE_SCORING = {
    "position_tolerance_m": 0.006,
    "rotation_tolerance_rad": 0.02,
    "case_targets": {
        "A": {
            "position": [0.365386543, 0.172741170, 0.250243791],
            "quaternion_xyzw": [-0.709151310, 0.004273792, -0.031112506, 0.704356562],
        },
        "B": {
            "position": [0.348538342, -0.189302554, 0.250570999],
            "quaternion_xyzw": [-0.700360495, -0.101638123, 0.178814684, 0.683513114],
        },
    },
}


def _git(workspace: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(workspace), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def ensure_private_scoring() -> Path:
    """Materialize harness-only adjudication data outside exportable run directories."""

    private_root = HARNESS_ROOT / "private"
    private_root.mkdir(parents=True, exist_ok=True)
    scoring_path = private_root / "pick_a17_scoring.json"
    if not scoring_path.is_file():
        scoring_path.write_text(
            json.dumps(_PRIVATE_SCORING, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return scoring_path


def build(destination: Path | None = None, bundle_path: Path | None = None) -> dict[str, str]:
    """Build a faulty-head worktree and a bundle with normal and faulty history."""

    workspace = (destination or DEFAULT_WORKSPACE).resolve()
    bundle = (bundle_path or HARNESS_ROOT / "ur5e_pick_demo.bundle").resolve()
    if workspace.exists():
        raise FileExistsError(
            f"demonstration workspace already exists: {workspace}; "
            "keep it for reruns or choose a new destination"
        )
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEMPLATE_ROOT, workspace)
    _git(workspace, "init", "--initial-branch=main")
    _git(workspace, "config", "user.email", "pick-cell@example.invalid")
    _git(workspace, "config", "user.name", "Pick Cell Demonstration")

    normal_path = workspace / "pick_demo" / "pipeline.py"
    normal_path.write_text(_NORMAL_PIPELINE, encoding="utf-8")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-m", "PICK-A17: normal TCP tool compensation")

    faulty_source = (TEMPLATE_ROOT / "pick_demo" / "pipeline.py").read_text(encoding="utf-8")
    normal_path.write_text(faulty_source, encoding="utf-8")
    _git(workspace, "add", "pick_demo/pipeline.py")
    _git(workspace, "commit", "-m", "PICK-A17: simplify tool compensation path")

    bundle.parent.mkdir(parents=True, exist_ok=True)
    if bundle.exists():
        bundle.unlink()
    _git(workspace, "bundle", "create", str(bundle), "main")

    ensure_private_scoring()
    return {"workspace": str(workspace), "bundle": str(bundle)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the PICK-A17 demonstration project")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    result = build(args.destination, args.bundle)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
