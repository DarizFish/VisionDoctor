"""Local orchestration for the independent PICK-A17 Gazebo demonstration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .build_demo_project import DEFAULT_RUNTIME, DEFAULT_WORKSPACE, build, ensure_private_scoring
from .geometry import compose, pose, pose_error
from .models import GraspAssessment, classify_grasp


class PickCellController:
    """Operate the local Gazebo cell without reaching into application code."""

    # Reuse the previously verified ROS/Gazebo runtime. Isolation belongs to
    # this harness's source, container, mounts, runtime directory, and export
    # boundary - not a duplicate system image.
    IMAGE = "visiondoctor/ros-gazebo:jazzy-v1"
    CONTAINER = "pick-cell-harness-gazebo"
    DOMAIN_ID = "189"
    WSLG_SOURCE = "/mnt/host/wslg/.X11-unix"
    WSLG_TARGET = "/tmp/.X11-unix"
    BUNDLE_SCHEMA = "observation-bundle/v1"
    # Joint-space seeds measured on the live UR5e/MoveIt stack for the two
    # fixed tabletop poses.  They select the same feasible IK branch for the
    # faulty and repaired TCP commands; the command pose remains public input.
    IK_SEEDS = {
        "A": (
            -2.58086512252255,
            -1.62654311330553,
            -2.16252306112823,
            -2.50427761027341,
            -2.53067154558669,
            3.07515560492662,
        ),
        "B": (
            -0.924764898673514,
            -1.24619974909407,
            1.82456182612289,
            -0.580084094920351,
            -1.32475821906725,
            -3.04941313005015,
        ),
    }

    def __init__(
        self,
        harness_root: Path | None = None,
        runtime_root: Path | None = None,
    ) -> None:
        self.harness_root = (harness_root or Path(__file__).resolve().parent).resolve()
        self.runtime_root = (
            runtime_root or Path(os.environ.get("PICK_CELL_RUNTIME_ROOT", DEFAULT_RUNTIME))
        ).resolve()
        self.scene_root = self.harness_root / "scene"
        self.ros_root = self.harness_root / "ros"
        self.private_root = self.harness_root / "private"
        self.state_path = self.runtime_root / "controller-state.json"

    @property
    def default_workspace(self) -> Path:
        return self.runtime_root / "projects" / DEFAULT_WORKSPACE.name

    def bootstrap_project(self) -> dict[str, str]:
        """Create the default faulty project and its operator-facing Git bundle."""

        return build(self.default_workspace, self.harness_root / "ur5e_pick_demo.bundle")

    def status(self) -> dict[str, Any]:
        running = self._container_running()
        processes = self._container_processes() if running else ""
        socket_mounted = False
        gui_window = {"found": False, "visible": False}
        if running:
            socket_mounted = (
                self._docker(
                    "exec", self.CONTAINER, "test", "-S", f"{self.WSLG_TARGET}/X0"
                ).returncode
                == 0
            )
            if socket_mounted:
                gui_window = self._gazebo_window_state()
        latest = self._read_json(self.runtime_root / "latest-run.json")
        return {
            "image": self.IMAGE,
            "image_ready": self._image_ready(),
            "container": self.CONTAINER,
            "running": running,
            "gazebo_server_running": running and "gz sim" in processes,
            "gazebo_gui_running": running and ("gz sim gui" in processes or "gz sim" in processes),
            "gazebo_window": gui_window,
            "wslg_socket_mounted": socket_mounted,
            "move_group_running": running and "move_group" in processes,
            "camera_bridge_running": running and "/pick_cell/rgbd/image" in processes,
            "default_workspace": str(self.default_workspace),
            "latest_run": latest,
        }

    def start_cell(self) -> dict[str, Any]:
        """Start the official Gazebo GUI and the UR5e simulation in one container."""

        if self._container_running():
            return {"started": False, "reason": "already_running", "status": self.status()}
        if not self._image_ready():
            raise RuntimeError(
                f"Required verified Gazebo image {self.IMAGE} is unavailable. "
                "Restore or pull that image before starting the PICK-A17 harness."
            )
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self._remove_stale_container()
        result = self._docker(
            "run",
            "--detach",
            "--name",
            self.CONTAINER,
            "--label",
            "pick-cell.component=gazebo",
            "--network",
            "bridge",
            "--shm-size",
            "1g",
            "--memory",
            "4g",
            "--cpus",
            "4",
            "--env",
            f"ROS_DOMAIN_ID={self.DOMAIN_ID}",
            "--env",
            "HOME=/tmp",
            "--env",
            "ROS_LOG_DIR=/tmp/ros-logs",
            "--env",
            "DISPLAY=:0",
            "--env",
            "QT_QPA_PLATFORM=xcb",
            "--env",
            "QT_X11_NO_MITSHM=1",
            "--env",
            "LIBGL_ALWAYS_SOFTWARE=1",
            "--mount",
            self._mount(self.scene_root, "/opt/pick-cell/scene", readonly=True),
            "--mount",
            self._mount(self.ros_root, "/opt/pick-cell/ros", readonly=True),
            "--mount",
            self._mount(self.runtime_root, "/opt/pick-cell/runs"),
            "--mount",
            self._mount(self.WSLG_SOURCE, self.WSLG_TARGET, readonly=True),
            self.IMAGE,
            "ros2",
            "launch",
            "ur_simulation_gz",
            "ur_sim_control.launch.py",
            "ur_type:=ur5e",
            "launch_rviz:=false",
            "gazebo_gui:=true",
            "world_file:=/opt/pick-cell/scene/pick_cell_world.sdf",
            timeout_s=35.0,
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or result.stdout.strip() or "failed to start Gazebo"
            )
        self._wait_for_ros("/scaled_joint_trajectory_controller", timeout_s=95.0)
        self._start_camera_bridge()
        window = self._wait_for_window(timeout_s=35.0)
        state = {
            "started_at": self._now(),
            "container": self.CONTAINER,
            "image": self.IMAGE,
            "display": "Docker Desktop WSLg",
            "window": window,
        }
        self._write_json(self.state_path, state)
        return {"started": True, "status": self.status()}

    def stop_cell(self) -> dict[str, Any]:
        """Stop only this named container; recorded runs and workspaces remain intact."""

        if not self._container_running():
            return {"stopped": False, "reason": "not_running"}
        result = self._docker("rm", "--force", self.CONTAINER, timeout_s=30.0)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "failed to stop Gazebo container")
        return {"stopped": True}

    def capture(self, destination: Path | None = None) -> dict[str, Any]:
        """Save an actual RGB-D frame and observer-camera clip from the live cell."""

        self._require_running()
        capture_root = destination or self.runtime_root / "captures" / self._new_id("capture")
        capture_root = capture_root.resolve()
        try:
            relative = capture_root.relative_to(self.runtime_root)
        except ValueError as exc:
            raise ValueError(
                "capture destination must be inside the harness runtime directory"
            ) from exc
        capture_root.mkdir(parents=True, exist_ok=True)
        container_destination = "/opt/pick-cell/runs/" + relative.as_posix()
        result = self._docker(
            "exec",
            "--env",
            f"PICK_CELL_CAPTURE_DIR={container_destination}",
            "--env",
            "PICK_CELL_CAPTURE_TIMEOUT_S=35",
            self.CONTAINER,
            "bash",
            "-lc",
            (
                "source /opt/ros/jazzy/setup.bash && exec python3 "
                "/opt/pick-cell/ros/scene_capture_probe.py"
            ),
            timeout_s=55.0,
        )
        payload = self._structured_result(result.stdout, "PICK_CELL_CAPTURE=")
        if result.returncode != 0 or not payload or not payload.get("success"):
            detail = payload.get("error") if payload else result.stderr.strip()
            raise RuntimeError(str(detail or "Gazebo camera capture failed"))
        payload["capture_dir"] = str(capture_root)
        return payload

    def run_cycle(self, workspace: Path | None = None) -> dict[str, Any]:
        """Run the public A/B vision program and create one sanitized observation bundle."""

        self._require_running()
        project = (workspace or self.default_workspace).resolve()
        if not project.exists():
            self.bootstrap_project()
            project = self.default_workspace
        self._validate_workspace(project)
        self._start_moveit()
        run_id = self._new_id("run")
        run_root = self.runtime_root / "runs" / run_id
        run_root.mkdir(parents=True, exist_ok=False)
        project_commit = self._project_commit(project)
        self._copy_project_evidence(project, run_root)
        results: list[dict[str, Any]] = []
        timeline: list[dict[str, str]] = []
        for case_id in ("A", "B"):
            case_root = run_root / "parts" / case_id
            capture_root = case_root / "capture"
            capture = self.capture(capture_root)
            timeline.append(self._timeline("camera_capture", case_id))
            input_path = case_root / "algorithm" / "input.json"
            output_path = case_root / "algorithm" / "output.json"
            log_path = case_root / "algorithm" / "application.jsonl"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_text(
                json.dumps(self._vision_input(run_id, case_id), ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            algorithm = self._run_project(project, input_path, output_path, log_path)
            self._create_detection_overlay(
                capture_root / "rgb.png", case_root / "detection-overlay.png"
            )
            timeline.append(self._timeline("vision_and_command", case_id))
            motion = self._execute_motion(
                algorithm["commanded_flange_base"],
                run_root,
                case_id,
            )
            (case_root / "motion.json").write_text(
                json.dumps(motion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            assessment = self._assess(project, case_id, algorithm, motion)
            result = {
                "part_id": case_id,
                "classification": assessment.classification.value,
                "success": assessment.success,
                "position_error_m": assessment.position_error_m,
                "rotation_error_rad": assessment.rotation_error_rad,
                "capture": {
                    "depth_valid_ratio": capture.get("depth_valid_ratio"),
                    "frame_id": capture.get("frame_id"),
                },
            }
            (case_root / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            results.append(result)
            timeline.append(self._timeline("robot_arrival_and_assessment", case_id))
        summary = {
            "run_id": run_id,
            "project_workspace": str(project),
            "project_commit": project_commit,
            "results": results,
            "overall_success": all(item["success"] for item in results),
        }
        (run_root / "run-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        bundle_path = self._write_bundle(run_root, run_id, project_commit, timeline, results)
        summary["bundle_path"] = str(bundle_path)
        self._write_json(self.runtime_root / "latest-run.json", summary)
        return summary

    def export_bundle(self, run_id: str, destination: Path) -> Path:
        """Copy exactly the manifest-listed, sanitized artifacts to a new directory."""

        source = (self.runtime_root / "runs" / run_id).resolve()
        manifest_path = source / "bundle.json"
        manifest = self._read_json(manifest_path)
        if not manifest or manifest.get("schema_version") != self.BUNDLE_SCHEMA:
            raise FileNotFoundError(f"no valid bundle for {run_id}")
        target = (destination.resolve() / run_id).resolve()
        if target.exists():
            raise FileExistsError(f"export target already exists: {target}")
        target.mkdir(parents=True)
        for item in manifest.get("artifacts", []):
            relative = Path(str(item["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("bundle contains an unsafe artifact path")
            file_source = (source / relative).resolve()
            if not file_source.is_file() or source not in file_source.parents:
                raise ValueError("bundle artifact is outside its run directory")
            if self._sha256(file_source) != item["sha256"]:
                raise ValueError(f"bundle artifact digest mismatch: {relative.as_posix()}")
            file_target = target / relative
            file_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_source, file_target)
        shutil.copy2(manifest_path, target / "bundle.json")
        return target

    def _start_camera_bridge(self) -> None:
        if "/pick_cell/rgbd/image" in self._container_processes():
            return
        bridge = " ".join(
            [
                "source /opt/ros/jazzy/setup.bash",
                "&& ros2 run ros_gz_bridge parameter_bridge",
                "'/pick_cell/rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image'",
                "'/pick_cell/rgbd/depth_image@sensor_msgs/msg/Image[gz.msgs.Image'",
                "'/pick_cell/rgbd/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'",
                "'/pick_cell/observer@sensor_msgs/msg/Image[gz.msgs.Image'",
            ]
        )
        result = self._docker("exec", "--detach", self.CONTAINER, "bash", "-lc", bridge)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "could not start camera bridge")
        self._wait_for_ros("/pick_cell/rgbd/image", timeout_s=40.0, topic=True)

    def _start_moveit(self) -> None:
        if "move_group" not in self._container_processes():
            launch = " ".join(
                [
                    "source /opt/ros/jazzy/setup.bash",
                    "&& ros2 launch ur_moveit_config ur_moveit.launch.py",
                    "ur_type:=ur5e launch_rviz:=false use_sim_time:=true",
                ]
            )
            result = self._docker("exec", "--detach", self.CONTAINER, "bash", "-lc", launch)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "could not start MoveIt")
        self._wait_for_ros("/move_action", timeout_s=95.0, action=True)

    def _execute_motion(
        self,
        command: dict[str, Any],
        run_root: Path,
        case_id: str,
    ) -> dict[str, Any]:
        output = run_root / "parts" / case_id / "trajectory.json"
        relative = output.relative_to(self.runtime_root)
        result = self._docker(
            "exec",
            "--env",
            "PICK_CELL_TARGET_FLANGE=" + json.dumps(command, separators=(",", ":")),
            "--env",
            "PICK_CELL_IK_SEED=" + json.dumps(self.IK_SEEDS[case_id], separators=(",", ":")),
            "--env",
            "PICK_CELL_MOTION_OUTPUT=/opt/pick-cell/runs/" + relative.as_posix(),
            self.CONTAINER,
            "bash",
            "-lc",
            (
                "source /opt/ros/jazzy/setup.bash && exec python3 "
                "/opt/pick-cell/ros/grasp_cycle_probe.py"
            ),
            timeout_s=110.0,
        )
        payload = self._structured_result(result.stdout, "PICK_CELL_MOTION=")
        if payload is None:
            return {
                "success": False,
                "error": result.stderr.strip() or "motion probe did not return structured data",
                "steps": [],
            }
        return payload

    def _assess(
        self,
        workspace: Path,
        case_id: str,
        algorithm: dict[str, Any],
        motion: dict[str, Any],
    ) -> GraspAssessment:
        scoring = self._private_scoring()
        command = pose(algorithm["commanded_flange_base"])
        measured_flange = pose(motion.get("actual_flange_base", command))
        tool = self._load_pose(workspace / "config" / "tool_profile.yaml", "tool0_to_tcp")
        expected = pose(scoring["case_targets"][case_id])
        measured_tcp = compose(measured_flange, tool)
        position_error, rotation_error = pose_error(expected, measured_tcp)
        return classify_grasp(
            position_error,
            rotation_error,
            position_tolerance_m=float(scoring["position_tolerance_m"]),
            rotation_tolerance_rad=float(scoring["rotation_tolerance_rad"]),
            motion_completed=bool(motion.get("success")),
        )

    def _run_project(
        self,
        workspace: Path,
        input_path: Path,
        output_path: Path,
        log_path: Path,
    ) -> dict[str, Any]:
        environment = os.environ.copy()
        python_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(workspace) + (
            os.pathsep + python_path if python_path else ""
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pick_demo.replay",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--log",
                str(log_path),
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=35.0,
            check=False,
            env=environment,
        )
        if result.returncode != 0 or not output_path.is_file():
            raise RuntimeError(
                result.stderr.strip() or result.stdout.strip() or "project replay failed"
            )
        value = self._read_json(output_path)
        if not value or "commanded_flange_base" not in value:
            raise RuntimeError("project replay did not emit a flange command")
        return value

    def _copy_project_evidence(self, workspace: Path, run_root: Path) -> None:
        destination = run_root / "project"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("cell_calibration.yaml", "tool_profile.yaml"):
            shutil.copy2(workspace / "config" / name, destination / name)
        commit = self._project_commit(workspace)
        (destination / "revision.json").write_text(
            json.dumps(commit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def _write_bundle(
        self,
        run_root: Path,
        run_id: str,
        project_commit: dict[str, str],
        timeline: list[dict[str, str]],
        results: list[dict[str, Any]],
    ) -> Path:
        denied = ("private", "truth", "score", "answer", "patch", "expected")
        artifacts: list[dict[str, str]] = []
        for file_path in sorted(run_root.rglob("*")):
            if not file_path.is_file() or file_path.name == "bundle.json":
                continue
            relative = file_path.relative_to(run_root).as_posix()
            if any(token in relative.lower() for token in denied):
                raise ValueError(f"unsafe artifact name in observation run: {relative}")
            if file_path.suffix.lower() in {".json", ".jsonl", ".yaml", ".yml"}:
                content = file_path.read_text(encoding="utf-8").lower()
                if any(token in content for token in denied):
                    raise ValueError(f"unsafe private data in observation run: {relative}")
            artifacts.append(
                {
                    "path": relative,
                    "sha256": self._sha256(file_path),
                    "media_type": self._media_type(file_path),
                    "captured_at": self._artifact_captured_at(file_path),
                    "clock_domain": self._artifact_clock_domain(file_path),
                }
            )
        manifest = {
            "schema_version": self.BUNDLE_SCHEMA,
            "run_id": run_id,
            "source": "gazebo_read_only_environment",
            "created_at": self._now(),
            "clock": {"source": "ros_sim_time_and_utc", "quality": "single_container"},
            "project_revision": project_commit,
            "timeline": timeline,
            "results": results,
            "artifacts": artifacts,
        }
        encoded = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        if any(token in encoded.lower() for token in denied):
            raise ValueError("sanitized bundle would expose forbidden harness data")
        bundle_path = run_root / "bundle.json"
        bundle_path.write_text(encoded, encoding="utf-8")
        return bundle_path

    def _private_scoring(self) -> dict[str, Any]:
        value = self._read_json(self.private_root / "pick_a17_scoring.json")
        if not value:
            ensure_private_scoring()
            value = self._read_json(self.private_root / "pick_a17_scoring.json")
        if not value:
            raise RuntimeError("private harness scoring data is unavailable")
        return value

    @staticmethod
    def _vision_input(run_id: str, case_id: str) -> dict[str, Any]:
        detections = {
            "A": {
                "detection_id": "camera-frame-A",
                "detected_part_in_camera": {
                    "position": [1.905419740, -0.114652442, 0.176447603],
                    "quaternion_xyzw": [
                        -0.053542663,
                        0.477958803,
                        -0.714231571,
                        0.508489753,
                    ],
                },
            },
            "B": {
                "detection_id": "camera-frame-B",
                "detected_part_in_camera": {
                    "position": [1.702169790, 0.112032404, -0.020181657],
                    "quaternion_xyzw": [
                        -0.180064970,
                        0.359461189,
                        -0.642715066,
                        0.652136185,
                    ],
                },
            },
        }
        return {
            "schema_version": "pick-a17-vision/v1",
            "run_id": run_id,
            "part_id": case_id,
            **detections[case_id],
        }

    def _validate_workspace(self, workspace: Path) -> None:
        required = [
            workspace / "pick_demo" / "replay.py",
            workspace / "config" / "cell_calibration.yaml",
            workspace / "config" / "tool_profile.yaml",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError("project workspace lacks PICK-A17 replay files: " + ", ".join(missing))

    def _require_running(self) -> None:
        if not self._container_running():
            raise RuntimeError(
                "Gazebo cell is not running; start the cell before collecting evidence"
            )

    def _wait_for_ros(
        self,
        expected: str,
        *,
        timeout_s: float,
        topic: bool = False,
        action: bool = False,
    ) -> None:
        command = "topic list" if topic else "action list" if action else "node list"
        deadline = time.monotonic() + timeout_s
        latest = ""
        while time.monotonic() < deadline:
            result = self._docker(
                "exec",
                self.CONTAINER,
                "bash",
                "-lc",
                f"source /opt/ros/jazzy/setup.bash && ros2 {command}",
                timeout_s=15.0,
            )
            latest = result.stdout + result.stderr
            if expected in result.stdout:
                return
            time.sleep(1.0)
        raise TimeoutError(f"ROS endpoint {expected} did not become ready: {latest[-500:]}")

    def _wait_for_window(self, *, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        latest: dict[str, Any] = {"found": False, "visible": False}
        while time.monotonic() < deadline:
            latest = self._gazebo_window_state()
            if latest.get("visible"):
                return latest
            time.sleep(1.0)
        raise RuntimeError(f"Gazebo GUI was not visible through WSLg: {latest.get('error', '')}")

    def _gazebo_window_state(self) -> dict[str, Any]:
        processes = self._container_processes()
        gz_sim_running = "gz sim" in processes
        result = self._docker(
            "exec",
            self.CONTAINER,
            "python3",
            "/opt/pick-cell/ros/window_probe.py",
            timeout_s=15.0,
        )
        payload = self._structured_result(result.stdout, "PICK_CELL_X11=")
        display_connected = bool(payload and payload.get("connected"))
        error = ""
        if payload and isinstance(payload.get("error"), str):
            error = payload["error"]
        elif result.returncode != 0:
            error = (result.stdout + result.stderr)[-500:]
        return {
            "found": gz_sim_running,
            "visible": gz_sim_running and display_connected,
            "error": error,
        }

    def _container_running(self) -> bool:
        result = self._docker("inspect", self.CONTAINER, "--format", "{{.State.Running}}")
        return result.returncode == 0 and result.stdout.strip().lower() == "true"

    def _container_processes(self) -> str:
        result = self._docker("top", self.CONTAINER, "-eo", "pid,args")
        return result.stdout if result.returncode == 0 else ""

    def _image_ready(self) -> bool:
        return self._docker("image", "inspect", self.IMAGE).returncode == 0

    def _remove_stale_container(self) -> None:
        self._docker("rm", "--force", self.CONTAINER, timeout_s=30.0)

    @staticmethod
    def _mount(source: Path | str, target: str, *, readonly: bool = False) -> str:
        suffix = ",readonly" if readonly else ""
        return f"type=bind,src={source},dst={target}{suffix}"

    @staticmethod
    def _structured_result(output: str, prefix: str) -> dict[str, Any] | None:
        for line in output.splitlines():
            if line.startswith(prefix):
                try:
                    value = json.loads(line.removeprefix(prefix))
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
        return None

    def _docker(self, *args: str, timeout_s: float = 30.0) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["docker", *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Docker Desktop CLI is unavailable") from exc

    @staticmethod
    def _load_pose(path: Path, key: str) -> dict[str, list[float]]:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"invalid YAML object: {path}")
        return pose(value[key])

    @staticmethod
    def _project_commit(workspace: Path) -> dict[str, str]:
        result = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10.0,
            check=False,
        )
        return {"commit": result.stdout.strip() if result.returncode == 0 else "unversioned"}

    @staticmethod
    def _create_detection_overlay(source: Path, destination: Path) -> None:
        from PIL import Image, ImageDraw

        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        draw.rectangle(
            (width * 0.34, height * 0.31, width * 0.66, height * 0.72),
            outline=(80, 255, 180),
            width=4,
        )
        draw.text((width * 0.34, height * 0.26), "vision pick candidate", fill=(80, 255, 180))
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _artifact_captured_at(path: Path) -> str:
        camera_stamp = PickCellController._capture_camera_stamp(path)
        if camera_stamp is not None:
            return f"{camera_stamp:.6f}"
        if path.name in {"motion.json", "trajectory.json"}:
            motion = PickCellController._read_json(path)
            trace = motion.get("joint_trajectory") if motion else None
            if isinstance(trace, list) and trace:
                first = trace[0]
                stamp = first.get("stamp_s") if isinstance(first, dict) else None
                if isinstance(stamp, (int, float)):
                    return f"{stamp:.6f}"
        return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()

    @staticmethod
    def _artifact_clock_domain(path: Path) -> str:
        if PickCellController._capture_camera_stamp(path) is not None:
            return "ros_sim_time_s"
        if path.name in {"motion.json", "trajectory.json"}:
            return "ros_sim_time_s"
        return "utc_file_mtime"

    @staticmethod
    def _capture_camera_stamp(path: Path) -> float | None:
        for parent in (path.parent, *path.parents):
            camera_info = parent / "camera-info.json"
            if camera_info.is_file():
                camera = PickCellController._read_json(camera_info)
                stamp = camera.get("stamp_s") if camera else None
                return float(stamp) if isinstance(stamp, (int, float)) else None
        return None

    @staticmethod
    def _media_type(path: Path) -> str:
        return {
            ".json": "application/json",
            ".jsonl": "application/x-ndjson",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".gif": "image/gif",
            ".mp4": "video/mp4",
            ".npy": "application/x-npy",
            ".yaml": "application/yaml",
        }.get(path.suffix.lower(), "application/octet-stream")

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _timeline(event: str, part_id: str) -> dict[str, str]:
        return {"at": datetime.now(UTC).isoformat(), "event": event, "part_id": part_id}

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
