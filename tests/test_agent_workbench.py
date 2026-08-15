from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from visiondoctor.api.app import ApiSettings, create_app
from visiondoctor.intake import IntakeService
from visiondoctor.llm import AssistantTurn, ToolCall
from visiondoctor.observability import read_agent_events
from visiondoctor.storage import SqliteRunRepository


class IntakeGatewayDouble:
    model = "intake-test-model"

    def complete(
        self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]
    ) -> AssistantTurn:
        assert "trusted_readiness" in str(messages[-1]["content"])
        assert tools[0]["function"]["name"] == "submit_intake_response"
        arguments = {
            "assistant_message": "信息来源已核对，可以进入调查。",
            "understanding": "视觉软件在故障提交后出现位姿回归。",
            "questions": [],
            "source_actions": ["使用代码 Diff 和隔离的案例证据"],
            "risk_notes": ["验收标准由用户确认"],
        }
        return AssistantTurn(
            content="",
            tool_calls=(
                ToolCall(
                    call_id="intake-call-1",
                    name="submit_intake_response",
                    arguments=arguments,
                ),
            ),
            finish_reason="tool_calls",
            raw_message={"role": "assistant", "tool_calls": []},
        )


class FakeSimulationService:
    def __init__(self) -> None:
        self.actions: list[tuple[str, str]] = []

    def status(self) -> dict[str, Any]:
        return {
            "visual": {
                "gazebo_gui_running": False,
                "gazebo_server_running": False,
                "move_group_running": False,
                "wslg_socket_mounted": False,
            },
            "active_operation": None,
            "latest_operation": None,
        }

    def start(self, action: str, *, case_id: str) -> dict[str, Any]:
        self.actions.append((action, case_id))
        return {"operation_id": "SIM-TEST", "action": action, "status": "RUNNING"}


def _git(repository: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return process.stdout.strip()


def _build_repository(root: Path) -> tuple[str, str]:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "VisionDoctor Test")
    _git(root, "config", "user.email", "visiondoctor@example.invalid")
    (root / "runner.py").write_text("print('runner')\n", encoding="utf-8")
    (root / "test_runner.py").write_text("print('tests')\n", encoding="utf-8")
    (root / "vision.py").write_text("ORDER = 'correct'\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "baseline")
    baseline = _git(root, "rev-parse", "HEAD")
    (root / "vision.py").write_text("ORDER = 'faulty'\n", encoding="utf-8")
    _git(root, "add", "vision.py")
    _git(root, "commit", "-q", "-m", "fault")
    return baseline, _git(root, "rev-parse", "HEAD")


def test_intake_agent_combines_user_repository_and_case_sources(tmp_path: Path) -> None:
    baseline, faulty = _build_repository(tmp_path / "repository")
    manifest = tmp_path / "manifest.json"
    reference = tmp_path / "qa_reference.json"
    manifest.write_text("{}\n", encoding="utf-8")
    reference.write_text("{}\n", encoding="utf-8")
    service = IntakeService(tmp_path / "intake", gateway=IntakeGatewayDouble())

    result = service.turn(
        message="升级视觉节点之后，机器人验证位姿出现稳定偏移。",
        session_id=None,
        repository_path=str(tmp_path / "repository"),
        baseline_commit=baseline,
        faulty_commit=faulty,
        runner_script="runner.py",
        test_runner_script="test_runner.py",
        robot_backend="dataset",
        cases=(
            {
                "case_id": "case-001",
                "manifest_path": str(manifest),
                "reference_path": str(reference),
            },
        ),
        acceptance_confirmed=True,
        patch_policy_confirmed=True,
        simulation_status={"visual": {"gazebo_gui_running": False}},
    )

    assert result["readiness"]["ready_to_run"] is True
    assert result["repository"]["changed_files"] == ["vision.py"]
    assert result["response"]["model"] == "intake-test-model"
    assert len(service.transcript(result["session_id"])) == 2


def test_agent_timeline_exposes_hashed_model_and_tool_proof(tmp_path: Path) -> None:
    (tmp_path / "trace.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "event": "state_transition",
                "previous": "DIAGNOSING",
                "current": "ROOT_CAUSE_CONFIRMED",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "model_audit.jsonl").write_text(
        json.dumps(
            {
                "request_id": "MODEL-123",
                "created_at": "2026-01-01T00:00:01Z",
                "model": "deepseek-v4-flash",
                "duration_s": 1.25,
                "request_sha256": "a" * 64,
                "response_sha256": "b" * 64,
                "tool_names": ["inspect_commit_diff", "submit_diagnosis"],
                "status": "succeeded",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    events = read_agent_events(tmp_path)

    assert any(item["actor"] == "deepseek-v4-flash" for item in events)
    assert any(item["title"] == "工具调用 · inspect_commit_diff" for item in events)
    model_event = next(item for item in events if item["kind"] == "model")
    assert model_event["proof"]["request_sha256"] == "a" * 64


def test_api_exposes_nonblocking_simulation_controls(tmp_path: Path) -> None:
    settings = ApiSettings(
        data_root=tmp_path / "server",
        database_path=tmp_path / "server" / "state.sqlite3",
    )
    repository = SqliteRunRepository(settings.database_path)
    simulation = FakeSimulationService()
    app = create_app(
        settings,
        repository,
        intake_service=IntakeService(tmp_path / "intake", gateway=IntakeGatewayDouble()),
        simulation_service=simulation,  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        assert client.get("/api/v1/simulation").status_code == 200
        response = client.post(
            "/api/v1/simulation/actions",
            json={"action": "run_motion", "case_id": "presentation"},
        )

    assert response.status_code == 202
    assert response.json()["operation_id"] == "SIM-TEST"
    assert simulation.actions == [("run_motion", "presentation")]


def test_api_deletes_idle_session_but_blocks_session_with_queued_job(
    tmp_path: Path,
) -> None:
    settings = ApiSettings(
        data_root=tmp_path / "server",
        database_path=tmp_path / "server" / "state.sqlite3",
    )
    repository = SqliteRunRepository(settings.database_path)
    app = create_app(settings, repository, simulation_service=FakeSimulationService())

    with TestClient(app) as client:
        idle = client.post("/api/v1/sessions", json={}).json()
        deleted = client.delete(f"/api/v1/sessions/{idle['session_id']}")
        assert deleted.status_code == 200
        assert deleted.json()["recoverable"] is True

        active = client.post("/api/v1/sessions", json={}).json()
        job = repository.enqueue_run_job(
            job_id="JOB-delete-guard",
            run_id="RUN-delete-guard",
            workspace_path=tmp_path / "workspace",
            sandbox="docker",
            robot_backend="dataset",
        )
        app.state.session_service.add_job(active["session_id"], job.job_id)
        blocked = client.delete(f"/api/v1/sessions/{active['session_id']}")

    assert blocked.status_code == 409
    assert "正在运行" in blocked.json()["detail"]
