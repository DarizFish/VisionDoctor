from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visiondoctor.sandbox.runner import CommandResult
from visiondoctor.simulation import SimulationService


def _write_capture(root: Path) -> Path:
    root.mkdir(parents=True)
    rgb = root / "rgb.png"
    depth = root / "depth.npy"
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (640, 480), (25, 31, 42))
    draw = ImageDraw.Draw(image)
    draw.ellipse((316, 236, 324, 244), fill=(60, 220, 130))
    draw.ellipse((412, 236, 420, 244), fill=(235, 80, 80))
    draw.ellipse((316, 332, 324, 340), fill=(80, 130, 240))
    image.save(rgb)
    np.save(depth, np.full((480, 640), 0.5, dtype=np.float32), allow_pickle=False)
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_id": "live",
                "captured_at": "2026-08-12T02:00:00+00:00",
                "rgb_path": rgb.name,
                "depth_path": depth.name,
                "camera_matrix": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
                "expected_pixel": [320, 240],
                "marker_axis_length_m": 0.08,
                "t_base_camera": {
                    "target_frame": "base",
                    "source_frame": "camera",
                    "matrix": np.eye(4).tolist(),
                    "length_unit": "m",
                    "angle_unit": "rad",
                    "quaternion_order": "xyzw",
                },
                "source": "gazebo",
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_project_observation_runs_contract_before_visual_motion(
    tmp_path: Path, monkeypatch,
) -> None:
    project_root = tmp_path / "visiondoctor"
    repository = tmp_path / "target"
    (project_root / "docker").mkdir(parents=True)
    repository.mkdir()
    (repository / "runner.py").write_text("# contract", encoding="utf-8")
    manifest = _write_capture(tmp_path / "capture")
    service = SimulationService(project_root, tmp_path / "runtime")
    target = np.eye(4)
    target[:3, 3] = [0.35, -0.10, 0.25]

    monkeypatch.setattr(
        "visiondoctor.simulation.DockerPythonRunner.ensure_image",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "visiondoctor.simulation.DockerPythonRunner.run_script",
        lambda *_args, **_kwargs: CommandResult(
            returncode=0,
            stdout=json.dumps(
                {
                    "results": [
                        {
                            "case_id": "live",
                            "T_base_object": target.tolist(),
                            "latency_s": 0.001,
                        }
                    ]
                }
            ),
            stderr="",
            duration_s=0.01,
        ),
    )
    motion_targets: list[dict[str, list[float]]] = []

    def fake_motion(*, target_tcp):
        motion_targets.append(target_tcp)
        return {"success": True, "tcp_translation_error_m": 0.001}

    monkeypatch.setattr(service.visual, "run_motion", fake_motion)

    result = service._run_project_observation(
        {
            "project_observation": {
                "repository_path": str(repository),
                "runner_script": "runner.py",
                "case_id": "live",
                "manifest_path": str(manifest),
            }
        }
    )

    assert result["observation_source"] == "current_project"
    assert result["project_target_tcp"]["position"] == [0.35, -0.1, 0.35]
    assert motion_targets == [result["project_target_tcp"]]

