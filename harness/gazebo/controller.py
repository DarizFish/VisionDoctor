"""Local orchestration for the independent PICK-A17 Gazebo demonstration."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
import weakref
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .build_demo_project import DEFAULT_RUNTIME, DEFAULT_WORKSPACE, build, ensure_private_scoring
from .geometry import compose, pose, pose_error
from .models import GraspAssessment, classify_grasp
from .sensor_faults import apply_sensor_fault


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
    # Gazebo renders every camera and its GUI in software.  At four CPUs the
    # container was throttled in nearly every scheduling period and simulated
    # time ran at about half of real time; twelve leave headroom for the host.
    CPU_LIMIT = "12"
    AGENT_SCRIPT = "/opt/pick-cell/ros/cell_agent.py"
    AGENT_CLIENT = "/opt/pick-cell/ros/cell_agent_client.py"
    # The fixture only occupies the low table zone.  Each cycle first reaches
    # this point above a part, then performs the visible downward pick and
    # returns to the same clearance height.
    PREGRASP_CLEARANCE_M = 0.16
    # Joint-space seeds measured on the live UR5e/MoveIt stack for the two
    # compact-fixture poses.  Both workstations are on the verified stable
    # side of the robot; the command pose remains public input.
    IK_SEEDS = {
        "A": (2.831821001, -2.091303128, -1.907909934, -0.713175869, 1.570797940, 1.241024775),
        # Keep B on the right-hand elbow branch for both its raised
        # pregrasp and its downward pick.  The neutral seed admits a second
        # valid IK solution that MoveIt cannot connect between these stages.
        "B": (2.831821001, -2.091303128, -1.907909934, -0.713175869, 1.570797940, 1.241024775),
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
        #: Warm interpreters for project programs, by workspace, restarted when code changes.
        self._workers: dict[Path, dict[str, Any]] = {}
        self._workers_lock = threading.Lock()
        self._deferred_clips = False
        weakref.finalize(self, _stop_workers, self._workers)

    @property
    def default_workspace(self) -> Path:
        return self.runtime_root / "projects" / DEFAULT_WORKSPACE.name

    @property
    def selected_workspace(self) -> Path:
        """Return the last console-selected workspace, or the safe default."""

        state = self._read_json(self.state_path) or {}
        selected = state.get("selected_workspace")
        if isinstance(selected, str) and selected:
            return Path(selected)
        return self.default_workspace

    def remember_workspace(self, workspace: Path) -> Path:
        """Persist the console's workspace selection across browser sessions."""

        selected = workspace.resolve()
        state = self._read_json(self.state_path) or {}
        state["selected_workspace"] = str(selected)
        self._write_json(self.state_path, state)
        return selected

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
            "selected_workspace": str(self.selected_workspace),
            "latest_run": latest,
        }

    def start_cell(self) -> dict[str, Any]:
        """Start the official Gazebo GUI and the UR5e simulation in one container."""

        if self._container_running():
            # Warm what the first pick needs, so the click does not pay for it.
            self._warm_cell(verify=True)
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
            self.CPU_LIMIT,
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
            "description_file:=/opt/pick-cell/scene/ur5e_pick_tool.urdf.xacro",
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
        window = self._wait_for_window(timeout_s=60.0)
        self._warm_cell(verify=True)
        state = self._read_json(self.state_path) or {}
        state.update(
            {
            "started_at": self._now(),
            "container": self.CONTAINER,
            "image": self.IMAGE,
            "display": "Docker Desktop WSLg",
            "window": window,
            }
        )
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

    def capture(
        self, destination: Path | None = None, *, defer_clip: bool = False,
    ) -> dict[str, Any]:
        """Save an actual RGB-D frame and observer-camera clip from the live cell.

        ``defer_clip`` returns once the RGB-D set and its record are written; the
        observer clip is finished in the agent and flushed before a bundle is sealed.
        """

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
        payload = self._agent_call(
            {"op": "capture", "dir": container_destination, "defer_clip": defer_clip},
            timeout_s=45.0,
        )
        self._deferred_clips = self._deferred_clips or defer_clip
        if not payload.get("success"):
            raise RuntimeError(str(payload.get("error") or "Gazebo camera capture failed"))
        payload["capture_dir"] = str(capture_root)
        return payload

    def run_cycle(
        self, workspace: Path | None = None, *, sensor_fault: str | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run the public A/B vision program and create one sanitized observation bundle."""

        if sensor_fault not in {None, "target_depth_dropout", "rgb_underexposure"}:
            raise ValueError(f"unknown sensor experiment: {sensor_fault}")
        self._require_running()
        project = (workspace or self.default_workspace).resolve()
        if not project.exists():
            self.bootstrap_project()
            project = self.default_workspace
        self._validate_workspace(project)
        self._progress = progress
        self._emit("prepare")
        self._deferred_clips = False
        self._warm_cell()
        run_id = self._new_id("run")
        run_root = self.runtime_root / "runs" / run_id
        run_root.mkdir(parents=True, exist_ok=False)
        if sensor_fault:
            # Grounding for the evaluator lives outside the exportable run.
            scenario_path = self.runtime_root / "scenario-labels" / f"{run_id}.json"
            scenario_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_json(scenario_path, {"run_id": run_id, "sensor_fault": sensor_fault,
                                          "scope": "injected sensor data before perception"})
        project_commit = self._project_commit(project)
        self._copy_project_evidence(project, run_root)
        results: list[dict[str, Any]] = []
        timeline: list[dict[str, str]] = []
        observation_started_at = self._now()
        for case_id in ("A", "B"):
            case_root = run_root / "parts" / case_id
            capture_root = case_root / "capture"
            self._emit("capture", case_id)
            capture = self.capture(capture_root, defer_clip=True)
            capture_meta = self._read_json(capture_root / "capture.json")
            capture_meta["phase"] = "before_command"
            self._write_json(capture_root / "capture.json", capture_meta)
            if sensor_fault:
                apply_sensor_fault(capture_root, sensor_fault)
                capture.update(self._read_json(capture_root / "capture.json"))
            timeline.append(self._timeline("camera_capture", case_id))
            input_path = case_root / "algorithm" / "input.json"
            output_path = case_root / "algorithm" / "output.json"
            log_path = case_root / "algorithm" / "application.jsonl"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            self._emit("perception", case_id)
            detection = self._vision_input(project, capture_root, input_path, run_id, case_id)
            if detection.get("status") == "detected":
                self._emit("command", case_id)
                algorithm = self._run_project(project, input_path, output_path, log_path)
                timeline.append(self._timeline("vision_and_command", case_id))
                motion = self._execute_motion(
                    algorithm["commanded_flange_base"], run_root, case_id,
                )
                assessment = self._assess(project, case_id, algorithm, motion)
                verdict = {
                    "classification": assessment.classification.value,
                    "success": assessment.success,
                    "position_error_m": assessment.position_error_m,
                    "rotation_error_rad": assessment.rotation_error_rad,
                }
            else:
                # Missing perception is itself an observable failed cycle. Do
                # not invent a pose or silently drop the run before exporting.
                algorithm = {
                    "run_id": run_id, "part_id": case_id,
                    "capture_id": detection.get("capture_id"),
                    "detection_id": detection.get("detection_id"),
                    "status": "no_command", "reason": detection.get("status"),
                }
                self._write_json(output_path, algorithm)
                timeline.append(self._timeline("vision_and_command", case_id))
                motion = {"success": False, "steps": [], "skipped_reason": "perception_no_pose"}
                verdict = {"classification": "perception_unavailable", "success": False}
                self._emit("no_command", case_id, str(detection.get("status")))
            (case_root / "motion.json").write_text(
                json.dumps(motion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            result = {
                "part_id": case_id,
                **verdict,
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
            self._emit("part_done", case_id, str(verdict["classification"]))
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
        self._emit("bundle")
        self._flush_clips()
        bundle_path = self._write_bundle(
            run_root, run_id, project_commit, timeline, results,
            observation_started_at=observation_started_at,
        )
        summary["bundle_path"] = str(bundle_path)
        self._write_json(self.runtime_root / "latest-run.json", summary)
        self._emit("done", detail=run_id)
        self._progress = None
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

    def _warm_cell(self, *, verify: bool = False) -> None:
        """MoveIt and the cell agent running; ``verify`` also waits for the agent to answer."""

        processes = self._container_processes()
        self._start_moveit(processes)
        self._ensure_agent(processes, verify=verify)

    def _start_moveit(self, processes: str | None = None) -> None:
        """Launch move_group once; the agent waits for its action server when it is used."""

        if "move_group" in (processes if processes is not None else self._container_processes()):
            return
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

    def _ensure_agent(
        self, processes: str | None = None, *, verify: bool = True, timeout_s: float = 45.0,
    ) -> None:
        """Start the resident cell agent if needed; ``verify`` waits until it answers."""

        running = "cell_agent.py" in (
            processes if processes is not None else self._container_processes()
        )
        if running and not verify:
            return
        if not running:
            launch = (
                "source /opt/ros/jazzy/setup.bash && exec python3 "
                f"{self.AGENT_SCRIPT} > /tmp/cell-agent.log 2>&1"
            )
            result = self._docker("exec", "--detach", self.CONTAINER, "bash", "-lc", launch)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "could not start the cell agent")
        deadline = time.monotonic() + timeout_s
        latest = ""
        while time.monotonic() < deadline:
            try:
                if self._agent_call({"op": "ping"}, timeout_s=10.0, retry=False).get("success"):
                    return
            except RuntimeError as exc:
                latest = str(exc)
            time.sleep(0.5)
        raise TimeoutError(f"cell agent did not answer: {latest}")

    def _flush_clips(self) -> None:
        if not self._deferred_clips:
            return
        result = self._agent_call({"op": "flush"}, timeout_s=60.0)
        self._deferred_clips = False
        if not result.get("success"):
            raise RuntimeError("observer clips were not written: " + "; ".join(result["errors"]))

    def _agent_call(
        self, request: dict[str, Any], *, timeout_s: float, retry: bool = True,
    ) -> dict[str, Any]:
        """Send one request to the agent, relaying its progress events as they arrive."""

        try:
            process = subprocess.Popen(
                ["docker", "exec", "-i", self.CONTAINER, "python3", self.AGENT_CLIENT],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Docker Desktop CLI is unavailable") from exc
        watchdog = threading.Timer(timeout_s, process.kill)
        watchdog.start()
        result: dict[str, Any] | None = None
        unavailable: str | None = None
        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(json.dumps(request))
            process.stdin.close()
            for line in process.stdout:
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if message.get("type") == "event":
                    step = message.get("step")
                    self._emit(f"agent_{message.get('event')}", detail=step)
                elif message.get("type") == "result":
                    result = message
                    break
                elif message.get("type") == "unavailable":
                    unavailable = str(message.get("error"))
                    break
            process.wait(timeout=10.0)
        finally:
            watchdog.cancel()
            if process.poll() is None:
                process.kill()
        if unavailable is not None:
            if not retry:
                raise RuntimeError(f"cell agent unavailable: {unavailable}")
            self._ensure_agent(verify=True)
            return self._agent_call(request, timeout_s=timeout_s, retry=False)
        if result is None:
            detail = process.stderr.read().strip() if process.stderr else ""
            raise RuntimeError(detail or f"cell agent returned no result for {request.get('op')}")
        result.pop("type", None)
        return result

    def _emit(self, stage: str, part: str | None = None, detail: str | None = None) -> None:
        """Tell whoever started the cycle where it is; never lets a viewer break the run."""

        listener = getattr(self, "_progress", None)
        if listener is None:
            return
        with contextlib.suppress(Exception):
            listener({"stage": stage, "part": part, "detail": detail, "at": time.monotonic()})

    def _execute_motion(
        self,
        command: dict[str, Any],
        run_root: Path,
        case_id: str,
    ) -> dict[str, Any]:
        output = run_root / "parts" / case_id / "trajectory.json"
        relative = output.relative_to(self.runtime_root)
        pregrasp = self._top_down_pregrasp(command)
        self._emit("motion", case_id)
        try:
            return self._agent_call(
                {
                    "op": "grasp",
                    "target": command,
                    "pregrasp": pregrasp,
                    "seed": list(self.IK_SEEDS[case_id]),
                    "output": "/opt/pick-cell/runs/" + relative.as_posix(),
                },
                timeout_s=480.0,
            )
        except RuntimeError as exc:
            return {"success": False, "error": str(exc), "steps": []}

    def _assess(
        self,
        workspace: Path,
        case_id: str,
        algorithm: dict[str, Any],
        motion: dict[str, Any],
    ) -> GraspAssessment:
        scoring = self._private_scoring()
        if motion.get("actual_flange_base") is None:
            return classify_grasp(
                None, None, position_tolerance_m=float(scoring["position_tolerance_m"]),
                rotation_tolerance_rad=float(scoring["rotation_tolerance_rad"]),
                motion_completed=False,
            )
        measured_flange = pose(motion["actual_flange_base"])
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
        code, stdout, stderr = self._run_program(
            workspace, "pick_demo.replay",
            ["--input", str(input_path), "--output", str(output_path), "--log", str(log_path)],
        )
        if code != 0 or not output_path.is_file():
            raise RuntimeError(stderr.strip() or stdout.strip() or "project replay failed")
        value = self._read_json(output_path)
        if not value or "commanded_flange_base" not in value:
            raise RuntimeError("project replay did not emit a flange command")
        return value

    def _copy_project_evidence(self, workspace: Path, run_root: Path) -> None:
        destination = run_root / "project"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("cell_calibration.yaml", "tool_profile.yaml", "perception.yaml"):
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
        *, observation_started_at: str | None = None,
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
                    "phase": self._artifact_phase(file_path),
                    "layer": self.artifact_layer(relative),
                }
            )
        manifest = {
            "schema_version": self.BUNDLE_SCHEMA,
            "run_id": run_id,
            "source": "gazebo_read_only_environment",
            "created_at": self._now(),
            "observation_started_at": observation_started_at,
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

    def _vision_input(
        self, workspace: Path, capture_root: Path, input_path: Path,
        run_id: str, case_id: str,
    ) -> dict[str, Any]:
        """Execute the bound project's perception on the actual captured pixels."""
        code, _, stderr = self._run_program(
            workspace, "pick_demo.perception",
            ["--capture", str(capture_root), "--part", case_id, "--run-id", run_id,
             "--output", str(input_path)],
        )
        if code != 0:
            raise RuntimeError(stderr.strip() or "perception program failed")
        return json.loads(input_path.read_text(encoding="utf-8"))

    def warm_project(self, workspace: Path) -> None:
        """Start the project's program interpreter before anyone clicks."""

        self._project_worker(workspace.resolve())

    def _run_program(
        self, workspace: Path, module: str, args: list[str], *, timeout_s: float = 35.0,
    ) -> tuple[int, str, str]:
        """Run one of the project's own programs, as ``python -m``, in a warm interpreter."""

        workspace = workspace.resolve()
        worker = self._project_worker(workspace)
        with worker["lock"]:
            worker["process"].stdin.write(json.dumps({"module": module, "args": args}) + "\n")
            worker["process"].stdin.flush()
            try:
                line = worker["lines"].get(timeout=timeout_s)
            except queue.Empty:
                line = None
        if line is None:
            with self._workers_lock:
                if self._workers.get(workspace) is worker:
                    del self._workers[workspace]
            _stop_worker(worker)
            raise RuntimeError(f"{module} did not finish; its interpreter was restarted")
        reply = json.loads(line)
        return int(reply["code"]), str(reply["stdout"]), str(reply["stderr"])

    def _project_worker(self, workspace: Path) -> dict[str, Any]:
        code = sorted(
            (path.relative_to(workspace).as_posix(), path.stat().st_mtime_ns, path.stat().st_size)
            for path in (workspace / "pick_demo").rglob("*.py")
        )
        with self._workers_lock:
            worker = self._workers.get(workspace)
            if worker and worker["code"] == code and worker["process"].poll() is None:
                return worker
            if worker:
                _stop_worker(worker)
            self.runtime_root.mkdir(parents=True, exist_ok=True)
            log = (self.runtime_root / "project-worker.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, "-u", str(self.harness_root / "project_worker.py")],
                cwd=workspace,
                env={**os.environ, "PYTHONPATH": str(workspace), "PYTHONIOENCODING": "utf-8"},
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                text=True, encoding="utf-8", errors="replace",
            )
            lines: queue.Queue[str | None] = queue.Queue()

            def pump() -> None:
                assert process.stdout is not None
                for line in process.stdout:
                    if line.startswith("{"):
                        lines.put(line)
                lines.put(None)

            threading.Thread(target=pump, daemon=True).start()
            worker = {"code": code, "process": process, "lines": lines,
                      "lock": threading.Lock(), "log": log}
            self._workers[workspace] = worker
            return worker

    @classmethod
    def _top_down_pregrasp(cls, command: dict[str, Any]) -> dict[str, list[float]]:
        """Return the vertical-clearance point for this top-down demonstration."""

        target = pose(command)
        position = list(target["position"])
        position[2] += cls.PREGRASP_CLEARANCE_M
        return {"position": position, "quaternion_xyzw": list(target["quaternion_xyzw"])}

    def _validate_workspace(self, workspace: Path) -> None:
        required = [
            workspace / "pick_demo" / "replay.py",
            workspace / "pick_demo" / "perception.py",
            workspace / "config" / "cell_calibration.yaml",
            workspace / "config" / "tool_profile.yaml",
            workspace / "config" / "perception.yaml",
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
        # Reaching the X display does not put a window on the desktop.  After monitors
        # are attached or rescaled while WSLg runs, it can render the window yet never
        # show it; restarting WSL (wsl --shutdown) re-reads the display layout.
        on_desktop = _desktop_window_visible("Gazebo Sim")
        if gz_sim_running and display_connected and on_desktop is False and not error:
            error = (
                "Gazebo GUI is running but its window is not on the Windows desktop; "
                "if displays changed since WSL started, run wsl --shutdown and start the cell again"
            )
        return {
            "found": gz_sim_running,
            "visible": gz_sim_running and display_connected and on_desktop is not False,
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
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def artifact_layer(relative: str) -> str:
        """What the cell program read, wrote or loaded is software; the rest is observed."""

        if "/algorithm/" in f"/{relative}" or relative.startswith("project/"):
            return "software"
        return "observation"

    @staticmethod
    def _artifact_phase(path: Path) -> str | None:
        for parent in (path.parent, *path.parents):
            capture = parent / "capture.json"
            if capture.is_file():
                return (PickCellController._read_json(capture) or {}).get("phase")
        if path.name in {"trajectory.json", "motion.json"}:
            return "command_execution"
        return None

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


def _stop_worker(worker: dict[str, Any]) -> None:
    process = worker["process"]
    if process.poll() is None:
        process.kill()
    worker["log"].close()


def _stop_workers(workers: dict[Path, dict[str, Any]]) -> None:
    for worker in list(workers.values()):
        _stop_worker(worker)
    workers.clear()


def _desktop_window_visible(title_prefix: str) -> bool | None:
    """Whether a visible top-level Windows window starts with this title; None off Windows."""

    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found = False
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def check(window, _data):
        nonlocal found
        if user32.IsWindowVisible(window):
            buffer = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(window, buffer, 256)
            if buffer.value.startswith(title_prefix):
                found = True
                return False
        return True

    user32.EnumWindows(callback_type(check), 0)
    return found
