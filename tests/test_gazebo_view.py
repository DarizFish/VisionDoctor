from __future__ import annotations

import json
import subprocess
from pathlib import Path

from visiondoctor.adapters.gazebo_view import GazeboVisualAdapter
from visiondoctor.cli import build_parser


def _completed(*, stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_gazebo_view_status_recognizes_container_processes(
    tmp_path: Path, monkeypatch,
) -> None:
    adapter = GazeboVisualAdapter(tmp_path, tmp_path / "session")
    process_list = "\n".join(
        (
            "gz sim server",
            "gz sim gui",
            "/opt/ros/jazzy/lib/moveit_ros_move_group/move_group",
        )
    )

    def fake_docker(command, *, timeout_s):
        del timeout_s
        if command[0] == "inspect":
            return _completed(stdout="true\n")
        if command[0] == "exec" and "python3" in command:
            return _completed(
                stdout='{"found":true,"visible":true,'
                '"geometry":{"x":38,"y":59,"width":1120,"height":880}}\n'
            )
        return _completed(stdout=process_list)

    monkeypatch.setattr(adapter, "_docker", fake_docker)

    assert adapter.status() == {
        "container": "visiondoctor-gazebo-view",
        "image": "visiondoctor/ros-gazebo:jazzy-v1",
        "display_backend": "docker-desktop-wslg",
        "wslg_x11_source": "/mnt/host/wslg/.X11-unix",
        "wslg_socket_mounted": True,
        "gazebo_gui_running": True,
        "gazebo_server_running": True,
        "move_group_running": True,
        "gazebo_window_visible": True,
        "gazebo_window_geometry": {"x": 38, "y": 59, "width": 1120, "height": 880},
    }


def test_cli_exposes_gazebo_view_lifecycle() -> None:
    parser = build_parser()

    status = parser.parse_args(["gazebo-view", "--status"])
    stop = parser.parse_args(["gazebo-view", "--stop"])
    assert status.status and not status.stop
    assert stop.stop and not stop.status


def test_gazebo_view_start_persists_container_motion_result(
    tmp_path: Path, monkeypatch,
) -> None:
    adapter = GazeboVisualAdapter(tmp_path, tmp_path / "session")
    statuses = iter(
        (
            {"gazebo_gui_running": False},
            {
                "gazebo_gui_running": True,
                "gazebo_server_running": True,
                "move_group_running": True,
            },
        )
    )
    docker_exec_results = iter(
        (
            _completed(),
            _completed(
                stdout=(
                    'VISIONDOCTOR_RESULT={"success":true,'
                    '"tcp_translation_error_m":0.001}\n'
                )
            ),
        )
    )
    docker_commands: list[tuple[str, ...]] = []

    def fake_docker(command, *, timeout_s):
        del timeout_s
        docker_commands.append(command)
        return _completed()

    monkeypatch.setattr(adapter, "require_available", lambda: None)
    monkeypatch.setattr(adapter, "status", lambda: next(statuses))
    monkeypatch.setattr(adapter, "_remove_stale_container", lambda: None)
    monkeypatch.setattr(
        adapter,
        "_ensure_window_visible",
        lambda _timeout: {"found": True, "visible": True},
    )
    monkeypatch.setattr(adapter, "_docker", fake_docker)
    monkeypatch.setattr(
        "visiondoctor.adapters.gazebo_view.GazeboAdapter._wait_for_ros_output",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "visiondoctor.adapters.gazebo_view.GazeboAdapter._docker_exec",
        lambda *_args, **_kwargs: next(docker_exec_results),
    )

    result = adapter.start()

    assert result["success"] is True
    assert result["motion_executed"] is True
    assert result["container"] == "visiondoctor-gazebo-view"
    assert result["display_backend"] == "docker-desktop-wslg"
    assert result["window"]["visible"] is True
    assert result["motion_result"]["tcp_translation_error_m"] == 0.001
    assert (tmp_path / "session" / "latest-session.json").is_file()
    launch = next(command for command in docker_commands if command[0] == "run")
    assert "DISPLAY=:0" in launch
    assert "world_file:=/visiondoctor/ros/visiondoctor_rgbd.sdf" in launch
    assert (
        "type=bind,src=/mnt/host/wslg/.X11-unix,"
        "dst=/tmp/.X11-unix,readonly"
    ) in launch


def test_gazebo_view_passes_project_target_to_motion_probe(
    tmp_path: Path, monkeypatch,
) -> None:
    adapter = GazeboVisualAdapter(tmp_path, tmp_path / "session")
    target = {
        "position": [0.4, 0.0, 0.3],
        "quaternion_xyzw": [0.5, -0.5, 0.5, -0.5],
    }
    calls: list[dict[str, str] | None] = []
    monkeypatch.setattr(
        adapter,
        "status",
        lambda: {
            "gazebo_gui_running": True,
            "gazebo_server_running": True,
            "move_group_running": True,
        },
    )

    def fake_exec(*_args, extra_environment=None, **_kwargs):
        calls.append(extra_environment)
        return _completed(
            stdout=(
                'VISIONDOCTOR_RESULT={"success":true,'
                '"tcp_translation_error_m":0.001}\n'
            )
        )

    monkeypatch.setattr(
        "visiondoctor.adapters.gazebo_view.GazeboAdapter._docker_exec",
        fake_exec,
    )

    result = adapter.run_motion(target_tcp=target)

    assert result["success"] is True
    assert result["attempt_count"] == 1
    assert result["recovered_from_transient"] is False
    assert calls == [
        {"VISIONDOCTOR_TARGET_TCP": json.dumps(target, separators=(",", ":"))}
    ]


def test_gazebo_view_retries_moveit_transient_once(
    tmp_path: Path, monkeypatch,
) -> None:
    adapter = GazeboVisualAdapter(tmp_path, tmp_path / "session")
    monkeypatch.setattr(
        adapter,
        "status",
        lambda: {
            "gazebo_gui_running": True,
            "gazebo_server_running": True,
            "move_group_running": True,
        },
    )
    executions = iter(
        (
            _completed(
                returncode=1,
                stdout=(
                    'VISIONDOCTOR_RESULT={"success":false,'
                    '"error":"MoveIt failed VALIDATION_POSE: error 99999",'
                    '"failure_stage":"VALIDATION_POSE",'
                    '"moveit_error_code":99999,"steps":[]}\n'
                ),
            ),
            _completed(
                stdout=(
                    'VISIONDOCTOR_RESULT={"success":true,'
                    '"tcp_translation_error_m":0.002}\n'
                )
            ),
        )
    )
    monkeypatch.setattr(
        "visiondoctor.adapters.gazebo_view.GazeboAdapter._docker_exec",
        lambda *_args, **_kwargs: next(executions),
    )

    result = adapter.run_motion()

    assert result["success"] is True
    assert result["attempt_count"] == 2
    assert result["recovered_from_transient"] is True


def test_gazebo_view_preserves_structured_failure_detail(
    tmp_path: Path, monkeypatch,
) -> None:
    adapter = GazeboVisualAdapter(tmp_path, tmp_path / "session")
    monkeypatch.setattr(
        adapter,
        "status",
        lambda: {
            "gazebo_gui_running": True,
            "gazebo_server_running": True,
            "move_group_running": True,
        },
    )
    failed = _completed(
        returncode=1,
        stdout=(
            'VISIONDOCTOR_RESULT={"success":false,'
            '"error":"MoveIt failed VALIDATION_POSE: error 99999",'
            '"failure_stage":"VALIDATION_POSE",'
            '"moveit_error_code":99999,"steps":[]}\n'
        ),
    )
    monkeypatch.setattr(
        "visiondoctor.adapters.gazebo_view.GazeboAdapter._docker_exec",
        lambda *_args, **_kwargs: failed,
    )

    try:
        adapter.run_motion()
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("motion failure should be reported")

    assert "error 99999" in message
    assert "planning_execution_transient" in message
    failure = json.loads(
        (tmp_path / "session" / "latest-motion-failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["attempt_count"] == 2
    assert failure["attempts"][-1]["payload"]["failure_stage"] == "VALIDATION_POSE"


def test_gazebo_view_stop_removes_container(tmp_path: Path, monkeypatch) -> None:
    adapter = GazeboVisualAdapter(tmp_path, tmp_path / "session")
    monkeypatch.setattr(adapter, "_docker", lambda *_args, **_kwargs: _completed())

    result = adapter.stop()

    assert result["container_removed"] is True
