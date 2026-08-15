from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from visiondoctor.adapters.base import AdapterUnavailableError, ExternalGateResult
from visiondoctor.geometry import (
    make_transform,
    project_origin,
    quaternion_xyzw_from_rotation,
    rotation_error_rad,
    translation_error_m,
)
from visiondoctor.geometry.transforms import rotation_from_euler
from visiondoctor.sandbox.runner import DockerPythonRunner
from visiondoctor.schemas import (
    CandidateVersion,
    ExecutionResult,
    ExecutionStatus,
    FailureCategory,
    PoseTransform,
)
from visiondoctor.vision import DeterministicRgbdPoseEstimator


@dataclass(frozen=True)
class GazeboAvailability:
    available: bool
    reason: str
    runtime: str = "none"
    image: str | None = None


@dataclass(frozen=True)
class GazeboContractResult:
    success: bool
    failure_category: FailureCategory | None
    retryable: bool
    payload: dict[str, Any]
    stdout: str
    stderr: str
    container_logs: str
    duration_s: float

    @property
    def infrastructure_error(self) -> bool:
        return self.failure_category in {
            FailureCategory.SIMULATOR_UNAVAILABLE,
            FailureCategory.PLANNING_EXECUTION_TRANSIENT,
            FailureCategory.INTERNAL_ORCHESTRATION_ERROR,
        }


class GazeboAdapter:
    """Optional ROS 2/Gazebo boundary; importing Core never imports ROS packages."""

    IMAGE = "visiondoctor/ros-gazebo:jazzy-v1"
    REQUIRED_PACKAGES = (
        "ros_gz_sim",
        "ros_gz_bridge",
        "cv_bridge",
        "moveit_ros_move_group",
        "ur_simulation_gz",
    )
    RGBD_WORLD = "visiondoctor_rgbd.sdf"
    RGBD_PROBE = "gazebo_rgbd_capture_probe.py"
    RGBD_TOPIC = "/visiondoctor/rgbd/image"

    @classmethod
    def availability(cls) -> GazeboAvailability:
        missing = [
            name
            for name in ("rclpy", "ros_gz_interfaces", "moveit_msgs")
            if importlib.util.find_spec(name) is None
        ]
        if not missing:
            return GazeboAvailability(
                available=True,
                reason="ROS 2 Python interfaces detected on the host",
                runtime="host",
            )
        if not DockerPythonRunner.available():
            return GazeboAvailability(
                available=False,
                reason=(
                    "host is missing optional ROS 2 packages and the Docker engine is unavailable: "
                    + ", ".join(missing)
                ),
            )
        runner = DockerPythonRunner(image=cls.IMAGE)
        if not runner.image_exists():
            return GazeboAvailability(
                available=False,
                reason=f"ROS/Gazebo image is not built: {cls.IMAGE}",
                runtime="docker",
                image=cls.IMAGE,
            )
        package_checks = " && ".join(f"ros2 pkg prefix {name}" for name in cls.REQUIRED_PACKAGES)
        try:
            process = subprocess.run(
                [
                    runner.docker_executable,
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    cls.IMAGE,
                    "bash",
                    "-lc",
                    package_checks + " && command -v gz",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
                env=runner._docker_environment(runner.docker_executable),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return GazeboAvailability(
                available=False,
                reason=f"ROS/Gazebo package probe failed: {exc}",
                runtime="docker",
                image=cls.IMAGE,
            )
        if process.returncode != 0:
            return GazeboAvailability(
                available=False,
                reason=process.stderr.strip() or "ROS/Gazebo package probe failed",
                runtime="docker",
                image=cls.IMAGE,
            )
        return GazeboAvailability(
            available=True,
            reason="ROS 2 Jazzy, Gazebo, MoveIt 2, and UR simulation packages detected",
            runtime="docker",
            image=cls.IMAGE,
        )

    @classmethod
    def build_image(cls, project_root: Path) -> None:
        runner = DockerPythonRunner(image=cls.IMAGE)
        runner.build_image(project_root / "docker" / "ros-gazebo.Dockerfile", project_root)

    @classmethod
    def run_fixed_motion_contract(
        cls,
        project_root: Path,
        *,
        target_tcp: dict[str, list[float]] | None = None,
        startup_timeout_s: float = 90.0,
        motion_timeout_s: float = 120.0,
    ) -> GazeboContractResult:
        cls().require_available()
        project_root = project_root.resolve()
        probe_root = (project_root / "ros").resolve()
        probe = probe_root / "moveit_fixed_motion_probe.py"
        if not probe.is_file():
            raise FileNotFoundError(probe)
        runner = DockerPythonRunner(image=cls.IMAGE)
        container_name = f"visiondoctor-gazebo-{uuid.uuid4().hex[:10]}"
        domain_id = str(100 + int(uuid.uuid4().hex[:4], 16) % 100)
        environment = runner._docker_environment(runner.docker_executable)
        started = time.perf_counter()
        stdout = ""
        stderr = ""
        logs = ""
        payload: dict[str, Any] = {"success": False, "error": "contract did not start"}
        failure_category: FailureCategory | None = None
        retryable = False
        launch = subprocess.run(
            [
                runner.docker_executable,
                "run",
                "--detach",
                "--name",
                container_name,
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges=true",
                "--pids-limit",
                "512",
                "--memory",
                "4g",
                "--cpus",
                "4",
                "--shm-size",
                "1g",
                "--user",
                "1000:1000",
                "--env",
                f"ROS_DOMAIN_ID={domain_id}",
                "--env",
                "HOME=/tmp",
                "--env",
                "ROS_LOG_DIR=/tmp/ros-logs",
                "--env",
                "QT_QPA_PLATFORM=offscreen",
                "--env",
                "GZ_IP=127.0.0.1",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=1g",
                "--mount",
                f"type=bind,src={probe_root},dst=/visiondoctor/ros,readonly",
                cls.IMAGE,
                "ros2",
                "launch",
                "ur_simulation_gz",
                "ur_sim_control.launch.py",
                "ur_type:=ur5e",
                "launch_rviz:=false",
                "gazebo_gui:=false",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            env=environment,
        )
        if launch.returncode != 0:
            raise AdapterUnavailableError(
                "failed to start the Gazebo container: " + (launch.stderr or launch.stdout)
            )
        try:
            cls._wait_for_ros_output(
                runner,
                container_name,
                domain_id,
                ("node", "list"),
                "/scaled_joint_trajectory_controller",
                startup_timeout_s,
            )
            moveit = cls._docker_exec(
                runner,
                container_name,
                domain_id,
                (
                    "ros2",
                    "launch",
                    "ur_moveit_config",
                    "ur_moveit.launch.py",
                    "ur_type:=ur5e",
                    "launch_rviz:=false",
                    "use_sim_time:=true",
                ),
                detached=True,
                timeout_s=15.0,
            )
            if moveit.returncode != 0:
                raise RuntimeError(moveit.stderr or "failed to start MoveIt")
            cls._wait_for_ros_output(
                runner,
                container_name,
                domain_id,
                ("action", "list", "-t"),
                "/move_action [moveit_msgs/action/MoveGroup]",
                startup_timeout_s,
            )
            probe_run = cls._docker_exec(
                runner,
                container_name,
                domain_id,
                ("python3", "/visiondoctor/ros/moveit_fixed_motion_probe.py"),
                detached=False,
                timeout_s=motion_timeout_s,
                extra_environment=(
                    {"VISIONDOCTOR_TARGET_TCP": json.dumps(target_tcp, separators=(",", ":"))}
                    if target_tcp
                    else None
                ),
            )
            stdout = probe_run.stdout
            stderr = probe_run.stderr
            marker = next(
                (
                    line.removeprefix("VISIONDOCTOR_RESULT=")
                    for line in stdout.splitlines()
                    if line.startswith("VISIONDOCTOR_RESULT=")
                ),
                None,
            )
            if marker is None:
                failure_category = FailureCategory.INTERNAL_ORCHESTRATION_ERROR
                payload = {
                    "success": False,
                    "error": "Gazebo probe did not emit a structured result",
                }
            else:
                payload = json.loads(marker)
            if probe_run.returncode != 0:
                payload["success"] = False
            if not payload.get("success") and failure_category is None:
                failure_category, retryable = cls._classify_fixed_motion_failure(
                    payload, probe_run.returncode
                )
        except Exception as exc:
            failure_category, retryable = cls._classify_contract_exception(exc)
            stderr = (stderr + "\n" + str(exc)).strip()
            payload = {"success": False, "error": str(exc)}
        finally:
            log_result = subprocess.run(
                [runner.docker_executable, "logs", container_name],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
                env=environment,
            )
            logs = log_result.stdout + log_result.stderr
            subprocess.run(
                [runner.docker_executable, "rm", "--force", container_name],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env=environment,
            )
        return GazeboContractResult(
            success=bool(payload.get("success")),
            failure_category=failure_category,
            retryable=retryable,
            payload=payload,
            stdout=stdout,
            stderr=stderr,
            container_logs=logs,
            duration_s=time.perf_counter() - started,
        )

    @staticmethod
    def _classify_fixed_motion_failure(
        payload: dict[str, Any], returncode: int
    ) -> tuple[FailureCategory, bool]:
        error = str(payload.get("error") or "")
        steps = payload.get("steps") or ()
        moveit_codes = [
            int(step["moveit_error_code"])
            for step in steps
            if isinstance(step, dict) and "moveit_error_code" in step
        ]
        explicit_code = payload.get("moveit_error_code")
        if explicit_code is not None:
            moveit_codes.append(int(explicit_code))
        error_match = re.search(r"MoveIt failed .*?error\s+(-?\d+)", error)
        if error_match:
            moveit_codes.append(int(error_match.group(1)))
        if any(code != 1 for code in moveit_codes):
            return FailureCategory.PLANNING_EXECUTION_TRANSIENT, True
        if "MoveIt" in error or "motion" in error.lower():
            return FailureCategory.PLANNING_EXECUTION_TRANSIENT, True
        if {
            "tcp_translation_error_m",
            "tcp_rotation_error_rad",
        }.issubset(payload):
            return FailureCategory.DETERMINISTIC_QA_FAILURE, False
        if returncode != 0:
            return FailureCategory.INTERNAL_ORCHESTRATION_ERROR, False
        return FailureCategory.DETERMINISTIC_QA_FAILURE, False

    @staticmethod
    def _classify_contract_exception(
        exc: Exception,
    ) -> tuple[FailureCategory, bool]:
        if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired, RuntimeError)):
            return FailureCategory.SIMULATOR_UNAVAILABLE, True
        return FailureCategory.INTERNAL_ORCHESTRATION_ERROR, False

    @classmethod
    def run_rgbd_capture_contract(
        cls,
        project_root: Path,
        output_root: Path,
        *,
        case_id: str = "scene-049",
        startup_timeout_s: float = 60.0,
        capture_timeout_s: float = 45.0,
    ) -> GazeboContractResult:
        """Capture a real Gazebo RGB-D frame and compare vision output to runtime truth."""
        cls().require_available()
        project_root = project_root.resolve()
        probe_root = (project_root / "ros").resolve()
        world = probe_root / cls.RGBD_WORLD
        probe = probe_root / cls.RGBD_PROBE
        for required in (world, probe):
            if not required.is_file():
                raise FileNotFoundError(required)
        output_root = output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        runner = DockerPythonRunner(image=cls.IMAGE)
        container_name = f"visiondoctor-gazebo-rgbd-{uuid.uuid4().hex[:10]}"
        domain_id = str(200 + int(uuid.uuid4().hex[:4], 16) % 20)
        environment = runner._docker_environment(runner.docker_executable)
        started = time.perf_counter()
        stdout = ""
        stderr = ""
        logs = ""
        payload: dict[str, Any] = {"success": False, "error": "capture did not start"}
        failure_category: FailureCategory | None = None
        retryable = False
        launch = subprocess.run(
            [
                runner.docker_executable,
                "run",
                "--detach",
                "--name",
                container_name,
                "--label",
                "visiondoctor.component=gazebo-rgbd-capture",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges=true",
                "--pids-limit",
                "256",
                "--memory",
                "3g",
                "--cpus",
                "3",
                "--shm-size",
                "1g",
                "--user",
                "1000:1000",
                "--env",
                f"ROS_DOMAIN_ID={domain_id}",
                "--env",
                "HOME=/tmp",
                "--env",
                "ROS_LOG_DIR=/tmp/ros-logs",
                "--env",
                "GZ_IP=127.0.0.1",
                "--env",
                "QT_QPA_PLATFORM=offscreen",
                "--env",
                "LIBGL_ALWAYS_SOFTWARE=1",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=1g",
                "--mount",
                f"type=bind,src={probe_root},dst=/visiondoctor/ros,readonly",
                "--mount",
                f"type=bind,src={output_root},dst=/visiondoctor/output",
                cls.IMAGE,
                "gz",
                "sim",
                "-s",
                "-r",
                "--headless-rendering",
                "-v",
                "3",
                f"/visiondoctor/ros/{cls.RGBD_WORLD}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            env=environment,
        )
        if launch.returncode != 0:
            raise AdapterUnavailableError(
                "failed to start the Gazebo RGB-D container: "
                + (launch.stderr or launch.stdout)
            )
        try:
            cls._wait_for_command_output(
                runner,
                container_name,
                domain_id,
                ("gz", "topic", "-l"),
                cls.RGBD_TOPIC,
                startup_timeout_s,
            )
            bridge = cls._docker_exec(
                runner,
                container_name,
                domain_id,
                (
                    "ros2",
                    "run",
                    "ros_gz_bridge",
                    "parameter_bridge",
                    "/visiondoctor/rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image",
                    "/visiondoctor/rgbd/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
                    (
                        "/visiondoctor/rgbd/camera_info@"
                        "sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
                    ),
                ),
                detached=True,
                timeout_s=15.0,
            )
            if bridge.returncode != 0:
                raise RuntimeError(bridge.stderr or "failed to start the Gazebo RGB-D bridge")
            cls._wait_for_ros_output(
                runner,
                container_name,
                domain_id,
                ("topic", "list"),
                "/visiondoctor/rgbd/depth_image",
                startup_timeout_s,
            )
            probe_run = cls._docker_exec(
                runner,
                container_name,
                domain_id,
                ("python3", f"/visiondoctor/ros/{cls.RGBD_PROBE}"),
                detached=False,
                timeout_s=capture_timeout_s,
                extra_environment={
                    "VISIONDOCTOR_OUTPUT_DIR": "/visiondoctor/output",
                    "VISIONDOCTOR_CAPTURE_TIMEOUT_S": str(capture_timeout_s - 5.0),
                },
            )
            stdout = probe_run.stdout
            stderr = probe_run.stderr
            marker = next(
                (
                    line.removeprefix("VISIONDOCTOR_RGBD_RESULT=")
                    for line in stdout.splitlines()
                    if line.startswith("VISIONDOCTOR_RGBD_RESULT=")
                ),
                None,
            )
            if probe_run.returncode != 0 or marker is None:
                raise RuntimeError(stderr or "Gazebo RGB-D probe did not return a result")
            capture = json.loads(marker)
            if not capture.get("success"):
                raise RuntimeError(str(capture.get("error", "Gazebo RGB-D capture failed")))

            camera_pose = cls._query_gazebo_model_pose(
                runner, container_name, domain_id, "vision_camera"
            )
            target_pose = cls._query_gazebo_model_pose(
                runner, container_name, domain_id, "vision_target"
            )
            optical_rotation = np.array(
                [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                dtype=float,
            )
            body_t_optical = make_transform(optical_rotation, np.zeros(3))
            t_base_camera = camera_pose @ body_t_optical
            t_base_object = target_pose @ body_t_optical
            t_camera_object = np.linalg.inv(t_base_camera) @ t_base_object
            camera_matrix = np.asarray(capture["camera_matrix"], dtype=float)
            rgb_path = output_root / str(capture["rgb_path"])
            depth_path = output_root / str(capture["depth_path"])
            estimated = DeterministicRgbdPoseEstimator().estimate(
                rgb_path, depth_path, camera_matrix
            )
            translation_error = translation_error_m(
                estimated.as_array(), t_camera_object
            )
            rotation_error = rotation_error_rad(estimated.as_array(), t_camera_object)
            expected_pixel = project_origin(t_camera_object, camera_matrix)
            depth_valid_ratio = float(capture["depth_valid_ratio"])
            stamp_spread = float(capture["message_stamp_spread_s"])
            succeeded = (
                translation_error <= 0.005
                and rotation_error <= 0.01745
                and depth_valid_ratio >= 0.60
                and stamp_spread <= 0.1
            )
            manifest = {
                "schema_version": 1,
                "case_id": case_id,
                "captured_at": capture["captured_at"],
                "rgb_path": rgb_path.name,
                "depth_path": depth_path.name,
                "camera_matrix": camera_matrix.tolist(),
                "expected_pixel": list(expected_pixel),
                "marker_axis_length_m": 0.08,
                "t_base_camera": PoseTransform.from_array(
                    "base", "camera", t_base_camera
                ).model_dump(mode="json"),
                "source": "gazebo",
                "camera_frame_id": capture["frame_id"],
            }
            manifest_path = output_root / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            reference_path = output_root / "qa_reference.json"
            reference_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "case_id": case_id,
                        "captured_at": capture["captured_at"],
                        "reference_t_base_object": PoseTransform.from_array(
                            "base", "object", t_base_object
                        ).model_dump(mode="json"),
                        "source_type": "gazebo_truth",
                        "provider": "gz model runtime state",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            payload = {
                **capture,
                "success": succeeded,
                "backend": "gazebo_rgbd",
                "runtime_uid": 1000,
                "case_id": case_id,
                "manifest_path": str(manifest_path),
                "reference_path": str(reference_path),
                "reference_source": "gazebo_truth",
                "estimated_t_camera_object": estimated.model_dump(mode="json"),
                "rgbd_translation_error_m": translation_error,
                "rgbd_rotation_error_rad": rotation_error,
                "acceptance": {
                    "translation_error_m": 0.005,
                    "rotation_error_rad": 0.01745,
                    "depth_valid_ratio": 0.60,
                    "message_stamp_spread_s": 0.1,
                },
            }
            if not succeeded:
                payload["error"] = "Gazebo RGB-D capture did not satisfy acceptance criteria"
                failure_category = FailureCategory.EVIDENCE_INCOMPLETE
        except Exception as exc:
            failure_category, retryable = cls._classify_contract_exception(exc)
            stderr = (stderr + "\n" + str(exc)).strip()
            payload = {"success": False, "error": str(exc), "backend": "gazebo_rgbd"}
        finally:
            log_result = subprocess.run(
                [runner.docker_executable, "logs", container_name],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
                env=environment,
            )
            logs = log_result.stdout + log_result.stderr
            subprocess.run(
                [runner.docker_executable, "rm", "--force", container_name],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env=environment,
            )
        duration = time.perf_counter() - started
        persisted_payload = {
            **payload,
            "duration_s": duration,
            "image": cls.IMAGE,
            "failure_category": (
                failure_category.value if failure_category is not None else None
            ),
            "retryable": retryable,
            "infrastructure_error": failure_category
            in {
                FailureCategory.SIMULATOR_UNAVAILABLE,
                FailureCategory.PLANNING_EXECUTION_TRANSIENT,
                FailureCategory.INTERNAL_ORCHESTRATION_ERROR,
            },
        }
        (output_root / "gazebo_rgbd_capture.json").write_text(
            json.dumps(persisted_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_root / "gazebo_rgbd_capture.log").write_text(logs, encoding="utf-8")
        return GazeboContractResult(
            success=bool(payload.get("success")),
            failure_category=failure_category,
            retryable=retryable,
            payload=persisted_payload,
            stdout=stdout,
            stderr=stderr,
            container_logs=logs,
            duration_s=duration,
        )

    @classmethod
    def _query_gazebo_model_pose(
        cls,
        runner: DockerPythonRunner,
        container_name: str,
        domain_id: str,
        model_name: str,
    ) -> np.ndarray:
        process = cls._docker_exec(
            runner,
            container_name,
            domain_id,
            ("gz", "model", "-m", model_name, "-p"),
            detached=False,
            timeout_s=15.0,
        )
        if process.returncode != 0:
            raise RuntimeError(process.stderr or f"Gazebo truth unavailable for {model_name}")
        lines = [
            re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
            for line in process.stdout.splitlines()
        ]
        pose_index = next(
            (index for index, line in enumerate(lines) if line.startswith("- Pose")), None
        )
        if pose_index is None or pose_index + 2 >= len(lines):
            raise ValueError(f"Gazebo emitted an invalid pose for {model_name}")

        def values(line: str) -> np.ndarray:
            match = re.fullmatch(r"\[([^]]+)\]", line)
            if match is None:
                raise ValueError(f"Gazebo pose vector is invalid: {line}")
            parsed = np.fromstring(match.group(1), sep=" ", dtype=float)
            if parsed.shape != (3,) or not np.isfinite(parsed).all():
                raise ValueError(f"Gazebo pose vector is invalid: {line}")
            return parsed

        position = values(lines[pose_index + 1])
        roll, pitch, yaw = values(lines[pose_index + 2])
        return make_transform(rotation_from_euler(roll, pitch, yaw), position)

    @classmethod
    def _wait_for_command_output(
        cls,
        runner: DockerPythonRunner,
        container_name: str,
        domain_id: str,
        command: tuple[str, ...],
        expected: str,
        timeout_s: float,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        last_output = ""
        while time.monotonic() < deadline:
            process = cls._docker_exec(
                runner,
                container_name,
                domain_id,
                command,
                detached=False,
                timeout_s=15.0,
            )
            last_output = process.stdout + process.stderr
            if process.returncode == 0 and expected in process.stdout:
                return
            time.sleep(0.5)
        raise TimeoutError(f"command readiness timed out waiting for {expected}: {last_output}")

    @staticmethod
    def _docker_exec(
        runner: DockerPythonRunner,
        container_name: str,
        domain_id: str,
        command: tuple[str, ...],
        *,
        detached: bool,
        timeout_s: float,
        extra_environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment_arguments = [
            value
            for key, item in (extra_environment or {}).items()
            for value in ("--env", f"{key}={item}")
        ]
        arguments = [
            runner.docker_executable,
            "exec",
            *(('--detach',) if detached else ()),
            "--env",
            f"ROS_DOMAIN_ID={domain_id}",
            *environment_arguments,
            container_name,
            "/ros_entrypoint.sh",
            *command,
        ]
        return subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
            env=runner._docker_environment(runner.docker_executable),
        )

    @classmethod
    def _wait_for_ros_output(
        cls,
        runner: DockerPythonRunner,
        container_name: str,
        domain_id: str,
        command: tuple[str, ...],
        expected: str,
        timeout_s: float,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        last_output = ""
        while time.monotonic() < deadline:
            process = cls._docker_exec(
                runner,
                container_name,
                domain_id,
                ("ros2", *command),
                detached=False,
                timeout_s=15.0,
            )
            last_output = process.stdout + process.stderr
            if process.returncode == 0 and expected in process.stdout:
                return
            time.sleep(0.5)
        raise TimeoutError(f"ROS readiness check timed out waiting for {expected}: {last_output}")

    def require_available(self) -> None:
        status = self.availability()
        if not status.available:
            raise AdapterUnavailableError(status.reason)


class GazeboFixedMotionGate:
    """Independent real-robot-simulation gate for the presentation scene."""

    def __init__(self, project_root: Path, *, case_id: str = "scene-main") -> None:
        self.project_root = project_root.resolve()
        self.case_id = case_id

    def evaluate(
        self, candidate: CandidateVersion, execution: ExecutionResult
    ) -> ExternalGateResult:
        case = next((item for item in execution.case_results if item.case_id == self.case_id), None)
        if (
            case is None
            or case.status is not ExecutionStatus.SUCCESS
            or case.robot_outputs is None
        ):
            return ExternalGateResult(
                gate_id=f"GAZEBO-{candidate.candidate_id}",
                name="gazebo_moveit_fixed_motion",
                case_id=self.case_id,
                passed=False,
                failure_category=FailureCategory.EVIDENCE_INCOMPLETE,
                retryable=False,
                details="selected execution has no valid presentation-scene TCP target",
                payload={"success": False},
            )
        matrix = np.asarray(case.robot_outputs.target_tcp.matrix, dtype=float)
        target_tcp = {
            "position": matrix[:3, 3].tolist(),
            "quaternion_xyzw": quaternion_xyzw_from_rotation(matrix[:3, :3]).tolist(),
        }
        try:
            contract = GazeboAdapter.run_fixed_motion_contract(
                self.project_root, target_tcp=target_tcp
            )
        except (AdapterUnavailableError, FileNotFoundError, OSError) as exc:
            return ExternalGateResult(
                gate_id=f"GAZEBO-{candidate.candidate_id}",
                name="gazebo_moveit_fixed_motion",
                case_id=self.case_id,
                passed=False,
                failure_category=FailureCategory.SIMULATOR_UNAVAILABLE,
                retryable=True,
                details=str(exc),
                payload={"success": False, "error": str(exc)},
            )
        translation_error = contract.payload.get("tcp_translation_error_m")
        rotation_error = contract.payload.get("tcp_rotation_error_rad")
        details = (
            f"translation_error_m={translation_error}; rotation_error_rad={rotation_error}; "
            f"sequence={contract.payload.get('motion_sequence')}"
            if contract.success
            else str(contract.payload.get("error", "fixed motion contract failed"))
        )
        return ExternalGateResult(
            gate_id=f"GAZEBO-{candidate.candidate_id}",
            name="gazebo_moveit_fixed_motion",
            case_id=self.case_id,
            passed=contract.success,
            failure_category=contract.failure_category,
            retryable=contract.retryable,
            details=details,
            payload={
                **contract.payload,
                "duration_s": contract.duration_s,
                "image": GazeboAdapter.IMAGE,
            },
            logs=contract.container_logs,
        )
