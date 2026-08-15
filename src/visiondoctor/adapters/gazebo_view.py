from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from visiondoctor.adapters.gazebo import GazeboAdapter
from visiondoctor.sandbox.runner import DockerPythonRunner


class GazeboVisualAdapter:
    """Runs the official Gazebo Qt GUI in the verified Docker image via WSLg."""

    IMAGE = GazeboAdapter.IMAGE
    CONTAINER = "visiondoctor-gazebo-view"
    DISPLAY_BACKEND = "docker-desktop-wslg"
    WSLG_X11_SOURCE = "/mnt/host/wslg/.X11-unix"
    WSLG_X11_TARGET = "/tmp/.X11-unix"
    DOMAIN_ID = "188"
    MOTION_ATTEMPTS = 2
    WINDOW_X = 80
    WINDOW_Y = 60
    WINDOW_WIDTH = 1120
    WINDOW_HEIGHT = 880

    def __init__(self, project_root: Path, session_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.session_root = session_root.resolve()

    def status(self) -> dict[str, Any]:
        inspect = self._docker(
            ("inspect", self.CONTAINER, "--format", "{{.State.Running}}"), timeout_s=15.0
        )
        running = inspect.returncode == 0 and inspect.stdout.strip().lower() == "true"
        processes = ""
        wslg_socket_mounted = False
        window: dict[str, Any] = {"found": False, "visible": False}
        if running:
            top = self._docker(("top", self.CONTAINER, "-eo", "pid,args"), timeout_s=15.0)
            if top.returncode == 0:
                processes = top.stdout
            socket_probe = self._docker(
                (
                    "exec",
                    self.CONTAINER,
                    "test",
                    "-S",
                    f"{self.WSLG_X11_TARGET}/X0",
                ),
                timeout_s=15.0,
            )
            wslg_socket_mounted = socket_probe.returncode == 0
            if wslg_socket_mounted:
                window = self._window_state()
        return {
            "container": self.CONTAINER,
            "image": self.IMAGE,
            "display_backend": self.DISPLAY_BACKEND,
            "wslg_x11_source": self.WSLG_X11_SOURCE,
            "wslg_socket_mounted": wslg_socket_mounted,
            "gazebo_gui_running": running and "gz sim gui" in processes,
            "gazebo_server_running": running and "gz sim server" in processes,
            "move_group_running": running and "moveit_ros_move_group/move_group" in processes,
            "gazebo_window_visible": bool(window.get("visible")),
            "gazebo_window_geometry": window.get("geometry"),
        }

    def start(
        self,
        *,
        run_motion: bool = True,
        startup_timeout_s: float = 90.0,
        motion_timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        self.require_available()
        if self.status()["gazebo_gui_running"]:
            raise RuntimeError("Gazebo GUI is already running; use gazebo-view --status")
        self.session_root.mkdir(parents=True, exist_ok=True)
        runner = DockerPythonRunner(image=self.IMAGE)
        container_started = False
        try:
            self._remove_stale_container()
            probe_root = (self.project_root / "ros").resolve()
            launch = self._docker(
                (
                    "run",
                    "--detach",
                    "--name",
                    self.CONTAINER,
                    "--label",
                    "visiondoctor.component=gazebo-view",
                    "--network",
                    "bridge",
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
                    f"ROS_DOMAIN_ID={self.DOMAIN_ID}",
                    "--env",
                    "HOME=/tmp",
                    "--env",
                    "ROS_LOG_DIR=/tmp/ros-logs",
                    "--env",
                    "GZ_IP=127.0.0.1",
                    "--env",
                    "DISPLAY=:0",
                    "--env",
                    "QT_QPA_PLATFORM=xcb",
                    "--env",
                    "QT_X11_NO_MITSHM=1",
                    "--env",
                    "LIBGL_ALWAYS_SOFTWARE=1",
                    "--tmpfs",
                    "/tmp:rw,nosuid,nodev,size=1g",
                    "--mount",
                    f"type=bind,src={probe_root},dst=/visiondoctor/ros,readonly",
                    "--mount",
                    (
                        f"type=bind,src={self.WSLG_X11_SOURCE},"
                        f"dst={self.WSLG_X11_TARGET},readonly"
                    ),
                    self.IMAGE,
                    "ros2",
                    "launch",
                    "ur_simulation_gz",
                    "ur_sim_control.launch.py",
                    "ur_type:=ur5e",
                    "launch_rviz:=false",
                    "gazebo_gui:=true",
                    f"world_file:=/visiondoctor/ros/{GazeboAdapter.RGBD_WORLD}",
                ),
                timeout_s=30.0,
            )
            if launch.returncode != 0:
                raise RuntimeError(launch.stderr or "failed to start visual Gazebo container")
            container_started = True
            GazeboAdapter._wait_for_ros_output(
                runner,
                self.CONTAINER,
                self.DOMAIN_ID,
                ("node", "list"),
                "/scaled_joint_trajectory_controller",
                startup_timeout_s,
            )
            window = self._ensure_window_visible(min(startup_timeout_s, 30.0))
            result: dict[str, Any] = {
                "success": True,
                "container": self.CONTAINER,
                "image": self.IMAGE,
                "display_backend": self.DISPLAY_BACKEND,
                "wslg_x11_source": self.WSLG_X11_SOURCE,
                "motion_executed": False,
                "window": window,
            }
            if run_motion:
                motion = self.run_motion(
                    startup_timeout_s=startup_timeout_s,
                    motion_timeout_s=motion_timeout_s,
                    _session_ready=True,
                    _force_moveit_start=True,
                )
                result.update({"motion_executed": True, "motion_result": motion})
            result["status"] = self.status()
            (self.session_root / "latest-session.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return result
        except Exception:
            if container_started:
                self._docker(("logs", self.CONTAINER), timeout_s=30.0)
            self.stop()
            raise

    def run_motion(
        self,
        *,
        target_tcp: dict[str, list[float]] | None = None,
        startup_timeout_s: float = 90.0,
        motion_timeout_s: float = 120.0,
        _session_ready: bool = False,
        _force_moveit_start: bool = False,
    ) -> dict[str, Any]:
        """Execute the fixed MoveIt motion in the already visible Gazebo session."""

        status = (
            {
                "gazebo_gui_running": True,
                "gazebo_server_running": True,
                "move_group_running": False,
            }
            if _session_ready
            else self.status()
        )
        if not status["gazebo_gui_running"] or not status["gazebo_server_running"]:
            raise RuntimeError("Gazebo GUI session is not running")
        runner = DockerPythonRunner(image=self.IMAGE)
        if _force_moveit_start or not status["move_group_running"]:
            moveit = GazeboAdapter._docker_exec(
                runner,
                self.CONTAINER,
                self.DOMAIN_ID,
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
            GazeboAdapter._wait_for_ros_output(
                runner,
                self.CONTAINER,
                self.DOMAIN_ID,
                ("action", "list", "-t"),
                "/move_action [moveit_msgs/action/MoveGroup]",
                startup_timeout_s,
            )
        attempts: list[dict[str, Any]] = []
        payload: dict[str, Any] | None = None
        for attempt in range(1, self.MOTION_ATTEMPTS + 1):
            motion = GazeboAdapter._docker_exec(
                runner,
                self.CONTAINER,
                self.DOMAIN_ID,
                ("python3", "/visiondoctor/ros/moveit_fixed_motion_probe.py"),
                detached=False,
                timeout_s=motion_timeout_s,
                extra_environment=(
                    {
                        "VISIONDOCTOR_TARGET_TCP": json.dumps(
                            target_tcp, separators=(",", ":")
                        )
                    }
                    if target_tcp
                    else None
                ),
            )
            marker = next(
                (
                    line.removeprefix("VISIONDOCTOR_RESULT=")
                    for line in motion.stdout.splitlines()
                    if line.startswith("VISIONDOCTOR_RESULT=")
                ),
                None,
            )
            parsed: dict[str, Any] | None = None
            parse_error = ""
            if marker is not None:
                try:
                    value = json.loads(marker)
                    if isinstance(value, dict):
                        parsed = value
                    else:
                        parse_error = "structured motion result was not an object"
                except json.JSONDecodeError as exc:
                    parse_error = f"invalid structured motion result: {exc}"

            if parsed is None:
                attempts.append(
                    {
                        "attempt": attempt,
                        "returncode": motion.returncode,
                        "failure_category": "internal_orchestration_error",
                        "retryable": False,
                        "error": (
                            parse_error
                            or motion.stderr.strip()
                            or motion.stdout.strip()
                            or "fixed motion did not return a result"
                        ),
                    }
                )
                break

            if motion.returncode == 0 and parsed.get("success"):
                payload = parsed
                attempts.append(
                    {
                        "attempt": attempt,
                        "returncode": motion.returncode,
                        "success": True,
                    }
                )
                break

            category, retryable = GazeboAdapter._classify_fixed_motion_failure(
                parsed, motion.returncode
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "returncode": motion.returncode,
                    "success": False,
                    "failure_category": category.value,
                    "retryable": retryable,
                    "payload": parsed,
                    "stderr": motion.stderr.strip(),
                }
            )
            if not retryable or attempt == self.MOTION_ATTEMPTS:
                break

        if payload is None:
            failure = {
                "success": False,
                "attempt_count": len(attempts),
                "attempts": attempts,
            }
            self._write_json("latest-motion-failure.json", failure)
            last = attempts[-1]
            detail = last.get("payload") or {}
            reason = str(detail.get("error") or last.get("error") or "fixed motion failed")
            category = str(last.get("failure_category") or "unknown")
            raise RuntimeError(
                f"{reason} (category={category}, attempts={len(attempts)})"
            )

        payload["attempt_count"] = len(attempts)
        payload["recovered_from_transient"] = len(attempts) > 1
        self.session_root.mkdir(parents=True, exist_ok=True)
        self._write_json("latest-motion.json", payload)
        return payload

    def _ensure_window_visible(self, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        last: dict[str, Any] = {"found": False, "visible": False}
        while time.monotonic() < deadline:
            last = self._window_state(reposition=True)
            if last.get("visible"):
                return last
            time.sleep(0.5)
        raise RuntimeError(
            "Gazebo GUI process started but its official 3D window was not visible: "
            + str(last.get("error") or "window was not mapped by WSLg")
        )

    def _window_state(self, *, reposition: bool = False) -> dict[str, Any]:
        script = f"""
import ctypes
import ctypes.util
import json

lib = ctypes.util.find_library("X11")
if not lib:
    print(json.dumps({{"found": False, "visible": False, "error": "libX11 missing"}}))
    raise SystemExit(0)
x = ctypes.cdll.LoadLibrary(lib)
x.XOpenDisplay.restype = ctypes.c_void_p
x.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
x.XDefaultRootWindow.restype = ctypes.c_ulong
x.XQueryTree.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)), ctypes.POINTER(ctypes.c_uint)]
x.XFetchName.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_char_p)]

class Attr(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int),
        ("width", ctypes.c_int), ("height", ctypes.c_int),
        ("border_width", ctypes.c_int), ("depth", ctypes.c_int),
        ("visual", ctypes.c_void_p), ("root", ctypes.c_ulong),
        ("class_", ctypes.c_int), ("bit_gravity", ctypes.c_int),
        ("win_gravity", ctypes.c_int), ("backing_store", ctypes.c_int),
        ("backing_planes", ctypes.c_ulong), ("backing_pixel", ctypes.c_ulong),
        ("save_under", ctypes.c_int), ("colormap", ctypes.c_ulong),
        ("map_installed", ctypes.c_int), ("map_state", ctypes.c_int),
        ("all_event_masks", ctypes.c_long), ("your_event_mask", ctypes.c_long),
        ("do_not_propagate_mask", ctypes.c_long),
        ("override_redirect", ctypes.c_int), ("screen", ctypes.c_void_p)]

x.XGetWindowAttributes.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
    ctypes.POINTER(Attr)]
x.XMoveResizeWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
    ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
x.XMapRaised.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
x.XFlush.argtypes = [ctypes.c_void_p]

d = x.XOpenDisplay(None)
if not d:
    print(json.dumps({{"found": False, "visible": False, "error": "cannot open DISPLAY"}}))
    raise SystemExit(0)
root = x.XDefaultRootWindow(d)

def children(window):
    returned_root = ctypes.c_ulong()
    parent = ctypes.c_ulong()
    values = ctypes.POINTER(ctypes.c_ulong)()
    count = ctypes.c_uint()
    if not x.XQueryTree(d, window, ctypes.byref(returned_root), ctypes.byref(parent),
                        ctypes.byref(values), ctypes.byref(count)):
        return []
    return [(values[index], parent.value) for index in range(count.value)]

def name(window):
    value = ctypes.c_char_p()
    if x.XFetchName(d, window, ctypes.byref(value)) and value.value:
        return value.value.decode("utf-8", "replace")
    return ""

def find(window):
    for child, _ in children(window):
        if name(child) == "Gazebo Sim":
            return child, window
        found = find(child)
        if found:
            return found
    return None

found = find(root)
if not found:
    print(json.dumps({{"found": False, "visible": False}}))
    raise SystemExit(0)
window, frame = found
if {str(reposition)}:
    x.XMoveResizeWindow(d, frame, {self.WINDOW_X}, {self.WINDOW_Y},
                        {self.WINDOW_WIDTH}, {self.WINDOW_HEIGHT})
    x.XMapRaised(d, frame)
    x.XMapRaised(d, window)
    x.XFlush(d)
attr = Attr()
x.XGetWindowAttributes(d, window, ctypes.byref(attr))
print(json.dumps({{"found": True, "visible": attr.map_state == 2,
    "geometry": {{"x": attr.x, "y": attr.y,
        "width": attr.width, "height": attr.height}}}}))
"""
        result = self._docker(
            ("exec", self.CONTAINER, "python3", "-c", script), timeout_s=15.0
        )
        if result.returncode != 0:
            return {
                "found": False,
                "visible": False,
                "error": result.stderr.strip() or result.stdout.strip(),
            }
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {
                "found": False,
                "visible": False,
                "error": result.stdout.strip() or "window probe returned invalid output",
            }
        return value if isinstance(value, dict) else {"found": False, "visible": False}

    def _write_json(self, name: str, value: dict[str, Any]) -> None:
        self.session_root.mkdir(parents=True, exist_ok=True)
        (self.session_root / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def stop(self) -> dict[str, Any]:
        removed = self._docker(("rm", "--force", self.CONTAINER), timeout_s=30.0)
        container_removed = removed.returncode == 0 or "No such container" in removed.stderr
        return {
            "container": self.CONTAINER,
            "container_removed": container_removed,
        }

    def require_available(self) -> None:
        if not DockerPythonRunner.available():
            raise RuntimeError("Docker engine is unavailable")
        runner = DockerPythonRunner(image=self.IMAGE)
        if not runner.image_exists():
            raise RuntimeError(f"required image is not built: {self.IMAGE}")
        socket_check = self._docker(
            (
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges=true",
                "--user",
                "1000:1000",
                "--mount",
                (
                    f"type=bind,src={self.WSLG_X11_SOURCE},"
                    f"dst={self.WSLG_X11_TARGET},readonly"
                ),
                self.IMAGE,
                "test",
                "-S",
                f"{self.WSLG_X11_TARGET}/X0",
            ),
            timeout_s=30.0,
        )
        if socket_check.returncode != 0:
            raise RuntimeError(
                "Docker Desktop WSLg X11 socket is unavailable: "
                f"{socket_check.stderr.strip()}"
            )

    def _remove_stale_container(self) -> None:
        inspect = self._docker(("inspect", self.CONTAINER), timeout_s=15.0)
        if inspect.returncode == 0:
            removed = self._docker(("rm", "--force", self.CONTAINER), timeout_s=30.0)
            if removed.returncode != 0:
                raise RuntimeError(removed.stderr or "cannot remove stale visual container")

    def _docker(
        self, command: tuple[str, ...], *, timeout_s: float
    ) -> subprocess.CompletedProcess[str]:
        executable = DockerPythonRunner.find_docker()
        if executable is None:
            return subprocess.CompletedProcess(
                args=list(command), returncode=127, stdout="", stderr="docker CLI is unavailable"
            )
        return subprocess.run(
            [executable, *command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
            env=DockerPythonRunner._docker_environment(executable),
        )
