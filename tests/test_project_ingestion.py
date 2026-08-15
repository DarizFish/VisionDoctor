from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visiondoctor.api.app import ApiSettings, create_app
from visiondoctor.llm import AssistantTurn, ModelProtocolError, ToolCall
from visiondoctor.llm.tools import ProjectGraphInspector
from visiondoctor.projects.catalog import ProjectCatalog
from visiondoctor.projects.ingestion import ProjectIngestionService
from visiondoctor.projects.models import (
    AmbiguityOption,
    AmbiguityStatus,
    AssetKind,
    ComponentKind,
    ProjectAmbiguity,
    UnderstandingStatus,
)
from visiondoctor.projects.resolver import ProjectSemanticResolver
from visiondoctor.sessions import DiagnosisSessionService
from visiondoctor.storage import SqliteRunRepository


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout.strip()


def _repository(root: Path, files: dict[str, str | bytes]) -> Path:
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.name", "VisionDoctor Project Test")
    _git(root, "config", "user.email", "project-test@example.invalid")
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "project snapshot")
    return root


def test_ros2_project_becomes_component_and_data_flow_graph(tmp_path: Path) -> None:
    repository = _repository(
        tmp_path / "robot-vision",
        {
            "ros2_ws/src/camera_driver/package.xml": (
                "<package><name>camera_driver</name></package>"
            ),
            "ros2_ws/src/camera_driver/camera.py": (
                "import rclpy\nfrom sensor_msgs.msg import Image, CameraInfo\n"
                "TOPICS = ['/camera/color/image_raw', '/camera/depth/image_raw']\n"
            ),
            "ros2_ws/src/detector/package.xml": "<package><name>detector</name></package>",
            "ros2_ws/src/detector/pose.py": (
                "import cv2\ndef estimate(rgb, depth):\n"
                "    return cv2.solvePnP([], [], [], [])\n"
            ),
            "ros2_ws/src/geometry/package.xml": "<package><name>geometry</name></package>",
            "ros2_ws/src/geometry/transform.py": (
                "def camera_to_base(T_base_camera, T_camera_object):\n"
                "    return T_base_camera @ T_camera_object\n"
            ),
            "ros2_ws/src/robot_task/package.xml": "<package><name>robot_task</name></package>",
            "ros2_ws/src/robot_task/move.py": (
                "import moveit_commander\n"
                "def target_tcp(object_pose_base): return object_pose_base\n"
            ),
            "launch/system.launch.py": "# gazebo ros_gz launch\n",
            "config/camera_intrinsics.yaml": "fx: 600\n",
            "runner.py": "print('run')\n",
            "test_runner.py": "print('Ran 1 test')\n",
        },
    )

    project = ProjectIngestionService().discover(repository)

    kinds = {item.kind for item in project.components}
    assert {"ROS 2", "OpenCV", "MoveIt 2", "Gazebo"} <= set(
        project.runtime.frameworks
    )
    assert ComponentKind.SENSOR in kinds
    assert ComponentKind.PERCEPTION_PROCESSOR in kinds
    assert ComponentKind.GEOMETRY_PROCESSOR in kinds
    assert ComponentKind.ROBOT_INTERFACE in kinds
    assert any(
        item.signal == "object_pose_camera" for item in project.relations
    )
    assert any(item.signal == "object_pose_base" for item in project.relations)
    assert project.source.repository_path == str(repository.resolve())


def test_plain_python_service_maps_models_configs_and_calibration_without_ros(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path / "vision-service",
        {
            "server.py": "from fastapi import FastAPI\napp = FastAPI()\n",
            "algorithm/inference.py": (
                "import onnxruntime\nMODEL = 'weights/best.onnx'\n"
                "def detection(image): return []\n"
            ),
            "weights/best.onnx": b"bounded-model-fixture",
            "resource/camera_calib.json": '{"fx": 600}',
            "config/prod.yaml": "threshold: 0.5\n",
            "samples/frame.png": b"not-decoded-by-project-scanner",
            "runner.py": "print('run')\n",
            "test_runner.py": "print('Ran 1 test')\n",
        },
    )

    project = ProjectIngestionService().discover(repository)

    assert "ROS 2" not in project.runtime.frameworks
    assert {"Python", "ONNX Runtime"} <= set(project.runtime.frameworks)
    assert any(item.kind is ComponentKind.PERCEPTION_PROCESSOR for item in project.components)
    assert any(item.kind is ComponentKind.MODEL for item in project.components)
    assert any(item.kind is AssetKind.MODEL for item in project.assets)
    assert any(item.kind is AssetKind.CALIBRATION for item in project.assets)
    assert any(item.kind.value == "uses" for item in project.relations)


def test_critical_ambiguities_require_confirmation_and_update_project_knowledge(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path / "ambiguous-project",
        {
            "vision.py": "def detection(image): return []\n",
            "config/calibration.yaml": "fx: 600\n",
            "resource/camera_extrinsic.yaml": "x: 0.1\n",
            "runner.py": "print('one')\n",
            "tools/run.py": "print('two')\n",
            "test_runner.py": "print('Ran 1 test')\n",
            "tests/test_pipeline.py": "def test_pipeline(): assert True\n",
        },
    )
    catalog = ProjectCatalog(tmp_path / "projects")
    project = catalog.ingest(repository)

    assert project.understanding_status is UnderstandingStatus.NEEDS_CONFIRMATION
    calibration = next(
        item for item in project.ambiguities if item.key == "production_calibration"
    )
    assert calibration.status is AmbiguityStatus.PENDING
    selected = calibration.options[0].option_id

    updated = catalog.confirm(project.project_id, calibration.ambiguity_id, selected)

    resolved = next(
        item for item in updated.ambiguities if item.ambiguity_id == calibration.ambiguity_id
    )
    assert resolved.status is AmbiguityStatus.CONFIRMED
    assert updated.knowledge.confirmed_facts["production_calibration"] == selected
    assert catalog.get(project.project_id).revision == updated.revision


class _ProjectGateway:
    model = "project-understanding-test"

    def __init__(self, component_id: str, evidence_path: str) -> None:
        self.component_id = component_id
        self.evidence_path = evidence_path
        self.calls = 0

    def complete(
        self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]
    ) -> AssistantTurn:
        self.calls += 1
        tool_names = {item["function"]["name"] for item in tools}
        assert "canonical project model" in messages[0]["content"]
        if self.calls == 1:
            assert tool_names == {"inspect_project_component"}
            return AssistantTurn(
                content="",
                tool_calls=(
                    ToolCall(
                        call_id="inspect-project-component",
                        name="inspect_project_component",
                        arguments={"component_id": self.component_id},
                    ),
                ),
                finish_reason="tool_calls",
                raw_message={"role": "assistant", "tool_calls": []},
            )
        if self.calls == 2:
            assert tool_names == {"read_repository_file"}
            return AssistantTurn(
                content="",
                tool_calls=(
                    ToolCall(
                        call_id="read-project-source",
                        name="read_repository_file",
                        arguments={"path": self.evidence_path},
                    ),
                ),
                finish_reason="tool_calls",
                raw_message={"role": "assistant", "tool_calls": []},
            )
        assert "submit_project_understanding" in tool_names
        assert any(item.get("role") == "tool" for item in messages)
        return AssistantTurn(
            content="",
            tool_calls=(
                ToolCall(
                    call_id="project-understanding-1",
                    name="submit_project_understanding",
                    arguments={
                        "summary": "该项目把相机坐标系中的目标位姿转换到机器人基坐标系。",
                        "component_interpretations": [
                            {
                                "component_id": self.component_id,
                                "kind": "geometry_processor",
                                "role": "执行相机到机器人基座的位姿变换",
                                "confidence": 0.98,
                                "evidence_paths": [self.evidence_path],
                            }
                        ],
                        "relation_suggestions": [],
                        "ambiguities": [],
                    },
                ),
            ),
            finish_reason="tool_calls",
            raw_message={"role": "assistant", "tool_calls": []},
        )


def test_semantic_resolver_can_only_update_known_components_and_paths(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path / "semantic-project",
        {
            "utils/trans.py": (
                "def camera_to_base(pose, extrinsic):\n    return extrinsic @ pose\n"
            ),
            "runner.py": "print('run')\n",
            "test_runner.py": "print('Ran 1 test')\n",
        },
    )
    project = ProjectIngestionService().discover(repository)
    component = next(
        item for item in project.components if "utils/trans.py" in item.source_paths
    )
    resolver = ProjectSemanticResolver(
        _ProjectGateway(component.component_id, "utils/trans.py")
    )

    understood = resolver.understand(project)

    interpreted = next(
        item for item in understood.components if item.component_id == component.component_id
    )
    assert interpreted.kind is ComponentKind.GEOMETRY_PROCESSOR
    assert interpreted.inferred_by == "model"
    assert understood.knowledge.model == "project-understanding-test"
    assert understood.understanding_status is UnderstandingStatus.UNDERSTOOD
    assert resolver.gateway.calls == 3


def test_project_summary_rejects_internal_development_vocabulary(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path / "customer-facing-summary-project",
        {
            "vision.py": "def detection(image): return []\n",
            "runner.py": "print('run')\n",
            "test_runner.py": "print('Ran 1 test')\n",
        },
    )
    project = ProjectIngestionService().discover(repository)
    component = next(item for item in project.components if "vision.py" in item.source_paths)
    resolver = ProjectSemanticResolver(
        _SummaryOnlyProjectGateway(component.component_id, "vision.py")
    )

    with pytest.raises(ModelProtocolError, match="internal implementation vocabulary"):
        resolver._apply(
            project,
            {
                "summary": (
                    "该项目没有脚本化 fallback，采用 fail-closed 策略并由 QA 与 "
                    "Orchestrator 处理。"
                ),
                "component_interpretations": [],
                "relation_suggestions": [],
                "ambiguities": [],
            },
            additional_allowed_paths={"vision.py"},
        )


class _RepairingProjectGateway:
    model = "repairing-project-understanding-test"

    def __init__(self, component_id: str, inspected_path: str) -> None:
        self.component_id = component_id
        self.inspected_path = inspected_path
        self.calls = 0

    def complete(
        self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]
    ) -> AssistantTurn:
        self.calls += 1
        names = {item["function"]["name"] for item in tools}
        if self.calls == 1:
            assert names == {"inspect_project_component"}
            return AssistantTurn(
                content="",
                tool_calls=(
                    ToolCall(
                        call_id="repair-inspect-component",
                        name="inspect_project_component",
                        arguments={"component_id": self.component_id},
                    ),
                ),
                finish_reason="tool_calls",
                raw_message={"role": "assistant", "tool_calls": []},
            )
        if self.calls == 2:
            assert names == {"read_repository_file"}
            return AssistantTurn(
                content="",
                tool_calls=(
                    ToolCall(
                        call_id="repair-read-source",
                        name="read_repository_file",
                        arguments={"path": self.inspected_path},
                    ),
                ),
                finish_reason="tool_calls",
                raw_message={"role": "assistant", "tool_calls": []},
            )
        if self.calls == 3:
            return AssistantTurn(
                content="",
                tool_calls=(
                    ToolCall(
                        call_id="repair-invalid-result",
                        name="submit_project_understanding",
                        arguments={
                            "summary": "该项目通过固定版本中的真实文件建立机器视觉处理链路。",
                            "component_interpretations": [
                                {
                                    "component_id": "COMP-000000000000",
                                    "kind": "unknown",
                                    "role": "invalid node used to exercise protocol repair",
                                    "confidence": 0.5,
                                    "evidence_paths": [self.inspected_path],
                                }
                            ],
                            "relation_suggestions": [],
                            "ambiguities": [],
                        },
                    ),
                ),
                finish_reason="tool_calls",
                raw_message={"role": "assistant", "tool_calls": []},
            )
        assert names == {"submit_project_understanding"}
        assert any(
            item.get("role") == "system"
            and "COMP-000000000000" in str(item.get("content", ""))
            for item in messages
        )
        return AssistantTurn(
            content="",
            tool_calls=(
                ToolCall(
                    call_id="repair-valid-result",
                    name="submit_project_understanding",
                    arguments={
                        "summary": "项目职责来自固定代码版本中实际读取的文件。",
                        "component_interpretations": [
                            {
                                "component_id": self.component_id,
                                "kind": "perception_processor",
                                "role": "Runs the repository's visual processing entrypoint.",
                                "confidence": 0.95,
                                "evidence_paths": [self.inspected_path],
                            }
                        ],
                        "relation_suggestions": [],
                        "ambiguities": [],
                    },
                ),
            ),
            finish_reason="tool_calls",
            raw_message={"role": "assistant", "tool_calls": []},
        )


def test_semantic_resolver_accepts_inspected_pinned_paths_and_repairs_in_place(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path / "semantic-repair-project",
        {
            "vision.py": "def detection(image): return []\n",
            "engineering-notes.txt": "vision.py is the production inference entrypoint\n",
            "runner.py": "print('run')\n",
            "test_runner.py": "print('Ran 1 test')\n",
        },
    )
    project = ProjectIngestionService().discover(repository)
    component = next(item for item in project.components if "vision.py" in item.source_paths)
    canonical_paths = {
        path for item in project.components for path in item.source_paths
    } | {item.path for item in project.assets}
    assert "engineering-notes.txt" not in canonical_paths
    gateway = _RepairingProjectGateway(component.component_id, "engineering-notes.txt")

    understood = ProjectSemanticResolver(gateway).understand(project)

    interpreted = next(
        item for item in understood.components if item.component_id == component.component_id
    )
    assert interpreted.inferred_by == "model"
    assert "engineering-notes.txt" in interpreted.evidence
    assert gateway.calls == 4


class _SummaryOnlyProjectGateway:
    model = "summary-only-project-understanding-test"

    def __init__(self, component_id: str, source_path: str) -> None:
        self.component_id = component_id
        self.source_path = source_path
        self.calls = 0

    def complete(
        self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]
    ) -> AssistantTurn:
        del messages, tools
        self.calls += 1
        if self.calls == 1:
            return AssistantTurn(
                content="",
                tool_calls=(
                    ToolCall(
                        call_id="summary-inspect-component",
                        name="inspect_project_component",
                        arguments={"component_id": self.component_id},
                    ),
                ),
                finish_reason="tool_calls",
                raw_message={"role": "assistant", "tool_calls": []},
            )
        if self.calls == 2:
            return AssistantTurn(
                content="",
                tool_calls=(
                    ToolCall(
                        call_id="summary-read-source",
                        name="read_repository_file",
                        arguments={"path": self.source_path},
                    ),
                ),
                finish_reason="tool_calls",
                raw_message={"role": "assistant", "tool_calls": []},
            )
        return AssistantTurn(
            content="",
            tool_calls=(
                ToolCall(
                    call_id="summary-only-result",
                    name="submit_project_understanding",
                    arguments={
                        "summary": "该项目处理机器视觉输入，并通过已有测试验证输出。",
                        "component_interpretations": [],
                        "relation_suggestions": [],
                        "ambiguities": [],
                    },
                ),
            ),
            finish_reason="tool_calls",
            raw_message={"role": "assistant", "tool_calls": []},
        )


def test_repeated_project_understanding_replaces_model_overlay(tmp_path: Path) -> None:
    repository = _repository(
        tmp_path / "repeat-understanding-project",
        {
            "vision.py": "def detection(image): return []\n",
            "runner.py": "print('run')\n",
            "test_runner.py": "print('Ran 1 test')\n",
        },
    )
    catalog = ProjectCatalog(tmp_path / "repeat-projects")
    discovered = catalog.ingest(repository)
    component = next(item for item in discovered.components if "vision.py" in item.source_paths)
    first = catalog.understand(
        discovered.project_id,
        _ProjectGateway(component.component_id, "vision.py"),
    )
    assert any(item.inferred_by == "model" for item in first.components)

    second = catalog.understand(
        discovered.project_id,
        _SummaryOnlyProjectGateway(component.component_id, "vision.py"),
    )

    assert second.revision == first.revision + 1
    assert not any(item.inferred_by == "model" for item in second.components)
    assert second.knowledge.summary.startswith("该项目")
    assert len(second.relations) == len(discovered.relations)


def test_semantic_base_retains_only_valid_confirmed_model_ambiguities(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path / "confirmed-model-project",
        {
            "camera.py": "def capture(): return 'image'\n",
            "vision.py": "def detection(image): return []\n",
            "camera-calibration.yaml": "fx: 600\n",
            "runner.py": "print('run')\n",
            "test_runner.py": "print('Ran 1 test')\n",
        },
    )
    discovered = ProjectIngestionService().discover(repository)
    component = discovered.components[0]
    asset = discovered.assets[0]
    option_values = (
        (component.component_id, component.name),
        (asset.asset_id, asset.path),
    )
    evidence = (component.source_paths[0],)
    confirmed = ProjectAmbiguity(
        ambiguity_id="AMB-aaaaaaaaaaaa",
        key="model:production-processor",
        question="生产链路使用哪个处理组件？",
        options=tuple(
            AmbiguityOption(option_id=option_id, label=label, evidence=evidence)
            for option_id, label in option_values
        ),
        recommended_option_id=option_values[0][0],
        selected_option_id=option_values[0][0],
        status=AmbiguityStatus.CONFIRMED,
        evidence=evidence,
        inferred_by="model",
    )
    previous = discovered.model_copy(
        update={
            "ambiguities": (*discovered.ambiguities, confirmed),
            "revision": 8,
        }
    )

    base = ProjectCatalog._semantic_base(
        previous,
        ProjectIngestionService().discover(repository),
    )

    assert base.revision == 8
    assert any(item.ambiguity_id == confirmed.ambiguity_id for item in base.ambiguities)
    assert base.knowledge.confirmed_facts[confirmed.key] == option_values[0][0]


class _PrematureProjectGateway:
    model = "premature-project-understanding-test"

    def complete(
        self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]
    ) -> AssistantTurn:
        del messages, tools
        return AssistantTurn(
            content="",
            tool_calls=(
                ToolCall(
                    call_id="premature-project-result",
                    name="submit_project_understanding",
                    arguments={
                        "summary": "没有读取源码的无效项目理解",
                        "component_interpretations": [],
                        "relation_suggestions": [],
                        "ambiguities": [],
                    },
                ),
            ),
            finish_reason="tool_calls",
            raw_message={"role": "assistant", "tool_calls": []},
        )


class _BudgetAwareProjectGateway:
    model = "budget-aware-project-understanding-test"

    def __init__(self, component_id: str, source_path: str) -> None:
        self.component_id = component_id
        self.source_path = source_path
        self.tool_sets: list[set[str]] = []
        self.budget_notice_seen = False
        self.plain_text_returned = False

    def complete(
        self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]
    ) -> AssistantTurn:
        names = {item["function"]["name"] for item in tools}
        self.tool_sets.append(names)
        self.budget_notice_seen = any(
            item.get("role") == "system"
            and "exploration budget is exhausted" in str(item.get("content", ""))
            for item in messages[1:]
        )
        if names == {"inspect_project_component"}:
            return AssistantTurn(
                content="",
                tool_calls=(
                    ToolCall(
                        call_id="budget-component",
                        name="inspect_project_component",
                        arguments={"component_id": self.component_id},
                    ),
                ),
                finish_reason="tool_calls",
                raw_message={"role": "assistant", "tool_calls": []},
            )
        if names == {"read_repository_file"}:
            return AssistantTurn(
                content="",
                tool_calls=(
                    ToolCall(
                        call_id="budget-source",
                        name="read_repository_file",
                        arguments={"path": self.source_path},
                    ),
                ),
                finish_reason="tool_calls",
                raw_message={"role": "assistant", "tool_calls": []},
            )
        if not self.budget_notice_seen:
            raise AssertionError("budget notice must follow required evidence stages")
        if not self.plain_text_returned:
            self.plain_text_returned = True
            return AssistantTurn(
                content="先用普通文字返回一次。",
                tool_calls=(),
                finish_reason="stop",
                raw_message={"role": "assistant", "content": "先用普通文字返回一次。"},
            )
        assert "submit_project_understanding" in names
        return AssistantTurn(
            content="",
            tool_calls=(
                ToolCall(
                    call_id="budget-result",
                    name="submit_project_understanding",
                    arguments={
                        "summary": "根据代表性组件和真实源码形成项目理解。",
                        "component_interpretations": [],
                        "relation_suggestions": [],
                        "ambiguities": [],
                    },
                ),
            ),
            finish_reason="tool_calls",
            raw_message={"role": "assistant", "tool_calls": []},
        )


class _ParallelRequiredProjectGateway:
    model = "parallel-required-project-understanding-test"

    def __init__(self, component_id: str, source_path: str) -> None:
        self.component_id = component_id
        self.source_path = source_path
        self.calls = 0

    def complete(
        self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]
    ) -> AssistantTurn:
        del messages
        self.calls += 1
        names = {item["function"]["name"] for item in tools}
        if names == {"inspect_project_component"}:
            calls = tuple(
                ToolCall(
                    call_id=f"parallel-component-{index}",
                    name="inspect_project_component",
                    arguments={"component_id": self.component_id},
                )
                for index in range(4)
            )
        elif names == {"read_repository_file"}:
            calls = tuple(
                ToolCall(
                    call_id=f"parallel-source-{index}",
                    name="read_repository_file",
                    arguments={"path": self.source_path},
                )
                for index in range(4)
            )
        else:
            calls = (
                ToolCall(
                    call_id="parallel-required-result",
                    name="submit_project_understanding",
                    arguments={
                        "summary": "该项目处理机器视觉数据，并用独立测试检查输出效果。",
                        "component_interpretations": [],
                        "relation_suggestions": [],
                        "ambiguities": [],
                    },
                ),
            )
        return AssistantTurn(
            content="",
            tool_calls=calls,
            finish_reason="tool_calls",
            raw_message={"role": "assistant", "tool_calls": []},
        )


class _MalformedOnceProjectGateway(_BudgetAwareProjectGateway):
    def __init__(self, component_id: str, source_path: str) -> None:
        super().__init__(component_id, source_path)
        self.protocol_error_returned = False

    def complete(
        self, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...]
    ) -> AssistantTurn:
        budget_notice = any(
            item.get("role") == "system"
            and "exploration budget is exhausted" in str(item.get("content", ""))
            for item in messages[1:]
        )
        if budget_notice and not self.protocol_error_returned:
            self.protocol_error_returned = True
            self.budget_notice_seen = True
            self.plain_text_returned = True
            self.tool_sets.append({item["function"]["name"] for item in tools})
            raise ModelProtocolError("provider returned malformed tool arguments")
        return super().complete(messages, tools)


def test_semantic_resolver_rejects_model_that_does_not_inspect_real_source(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path / "uninspected-project",
        {
            "vision.py": "def detection(image): return []\n",
            "runner.py": "print('run')\n",
            "test_runner.py": "print('test')\n",
        },
    )
    project = ProjectIngestionService().discover(repository)
    resolver = ProjectSemanticResolver(_PrematureProjectGateway(), max_tool_iterations=1)

    with pytest.raises(ModelProtocolError, match="did not produce a valid"):
        resolver.understand(project)


def test_semantic_resolver_enforces_hard_exploration_budget(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path / "budgeted-project",
        {
            "vision.py": "def detection(image): return []\n",
            "runner.py": "print('run')\n",
            "test_runner.py": "print('test')\n",
        },
    )
    project = ProjectIngestionService().discover(repository)
    component = next(item for item in project.components if "vision.py" in item.source_paths)
    gateway = _BudgetAwareProjectGateway(component.component_id, "vision.py")
    resolver = ProjectSemanticResolver(
        gateway, max_tool_iterations=4, max_exploratory_calls=2
    )

    understood = resolver.understand(project)

    assert understood.knowledge.summary == "根据代表性组件和真实源码形成项目理解。"
    assert len(gateway.tool_sets) == 4
    assert gateway.budget_notice_seen is True
    assert gateway.plain_text_returned is True


def test_required_evidence_stage_counts_only_first_parallel_call(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path / "parallel-required-project",
        {
            "vision.py": "def detection(image): return []\n",
            "runner.py": "print('run')\n",
            "test_runner.py": "print('test')\n",
        },
    )
    project = ProjectIngestionService().discover(repository)
    component = next(item for item in project.components if "vision.py" in item.source_paths)
    gateway = _ParallelRequiredProjectGateway(component.component_id, "vision.py")

    understood = ProjectSemanticResolver(
        gateway,
        max_tool_iterations=3,
        max_exploratory_calls=2,
    ).understand(project)

    assert understood.knowledge.model == gateway.model
    assert gateway.calls == 3


def test_semantic_resolver_retries_one_malformed_provider_tool_response(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path / "protocol-repair-project",
        {
            "vision.py": "def detection(image): return []\n",
            "runner.py": "print('run')\n",
            "test_runner.py": "print('test')\n",
        },
    )
    project = ProjectIngestionService().discover(repository)
    component = next(item for item in project.components if "vision.py" in item.source_paths)
    gateway = _MalformedOnceProjectGateway(component.component_id, "vision.py")
    resolver = ProjectSemanticResolver(
        gateway, max_tool_iterations=4, max_exploratory_calls=2
    )

    understood = resolver.understand(project)

    assert gateway.protocol_error_returned is True
    assert understood.knowledge.model == gateway.model


def test_project_graph_inspector_traces_impact_without_filesystem_access(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path / "impact-project",
        {
            "camera.py": "def camera_driver(): return 'image_raw'\n",
            "detector.py": "def detection(rgb): return []\n",
            "runner.py": "print('run')\n",
            "test_runner.py": "print('Ran 1 test')\n",
        },
    )
    project = ProjectIngestionService().discover(repository)
    sensor = next(item for item in project.components if item.kind is ComponentKind.SENSOR)
    inspector = ProjectGraphInspector(project)

    result = inspector.execute(
        "trace_project_dependencies",
        {"component_id": sensor.component_id, "direction": "downstream", "depth": 3},
    )

    assert sensor.component_id in inspector.inspected_components
    assert result["data"]["start_component_id"] == sensor.component_id
    assert "repository_path" not in str(result)


def test_multiple_sessions_reuse_one_long_lived_project(tmp_path: Path) -> None:
    repository = _repository(
        tmp_path / "shared-project",
        {
            "vision.py": "def detection(image): return []\n",
            "runner.py": "print('run')\n",
            "test_runner.py": "print('Ran 1 test')\n",
        },
    )
    service = DiagnosisSessionService(tmp_path / "server" / "sessions")
    first = service.create()
    second = service.create()

    first = service.connect_repository(first["session_id"], str(repository))
    second = service.connect_repository(second["session_id"], str(repository))

    assert first["project"]["project_id"] == second["project"]["project_id"]
    assert len(service.projects.list()) == 1
    service.record_incident(first["session_id"], "INC-PROJECT-001")
    stored = service.projects.get(first["project"]["project_id"])
    assert stored.incident_ids == ("INC-PROJECT-001",)


def test_parallel_ingestion_reuses_one_atomic_project_record(tmp_path: Path) -> None:
    repository = _repository(
        tmp_path / "parallel-project",
        {
            "vision.py": "def detection(image): return []\n",
            "runner.py": "print('run')\n",
            "test_runner.py": "print('test')\n",
        },
    )
    catalog = ProjectCatalog(tmp_path / "projects")

    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(lambda _index: catalog.ingest(repository), range(8)))

    assert len({item.project_id for item in values}) == 1
    assert len({item.revision for item in values}) == 1
    assert len(catalog.list()) == 1


def test_assisted_manifest_maps_arbitrary_layout_and_confirms_runtime(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path / "unconventional-layout",
        {
            "odd/device/input_alpha.py": "def camera_driver(): return 'frame_packet'\n",
            "odd/logic/infer_beta.py": "def detection(frame): return []\n",
            "ops/runner.py": "print('production')\n",
            "tools/run.py": "print('development')\n",
            "checks/test_primary.py": "def test_primary(): assert True\n",
            "checks/test_secondary.py": "def test_secondary(): assert True\n",
            "resources/camera_calibration.yaml": "fx: 600\n",
            "resources/backup_calibration.yaml": "fx: 590\n",
            "captures/golden.bin": b"bounded-user-declared-dataset",
            "visiondoctor.yaml": """
project:
  name: 包装线字符检查
components:
  - id: line_camera
    type: sensor
    name: 产线相机
    role: 采集待检包装图像
    source_paths: [odd/device/input_alpha.py]
    consumes: []
    produces: [frame_packet]
  - id: character_reader
    type: perception_processor
    name: 字符识别器
    role: 读取包装上的批次字符
    source_paths: [odd/logic/infer_beta.py]
    consumes: [frame_packet]
    produces: [recognized_text]
relations:
  - source: line_camera
    type: data_flow
    target: character_reader
    signal: frame_packet
runtime:
  entrypoints: [ops/runner.py]
validation:
  runner: ops/runner.py
  test_runner: checks/test_primary.py
  datasets: [captures/golden.bin]
confirmations:
  production_calibration: resources/camera_calibration.yaml
""",
        },
    )

    project = ProjectIngestionService().discover(repository)

    assert project.name == "包装线字符检查"
    camera = next(item for item in project.components if item.name == "产线相机")
    reader = next(item for item in project.components if item.name == "字符识别器")
    assert camera.kind is ComponentKind.SENSOR
    assert reader.kind is ComponentKind.PERCEPTION_PROCESSOR
    assert camera.inferred_by == "confirmed"
    assert project.runtime.entrypoints == ("ops/runner.py",)
    assert project.validation.selected_runner == "ops/runner.py"
    assert project.validation.selected_test_runner == "checks/test_primary.py"
    assert project.validation.dataset_candidates == ("captures/golden.bin",)
    assert any(
        relation.source_id == camera.component_id
        and relation.target_id == reader.component_id
        and relation.signal == "frame_packet"
        and relation.inferred_by == "confirmed"
        for relation in project.relations
    )
    calibration = next(
        item for item in project.ambiguities if item.key == "production_calibration"
    )
    assert calibration.status is AmbiguityStatus.CONFIRMED
    assert not [
        item for item in project.ambiguities if item.status is AmbiguityStatus.PENDING
    ]

    service = DiagnosisSessionService(tmp_path / "server" / "sessions")
    session = service.create()
    session = service.connect_repository(session["session_id"], str(repository))
    project_check = next(
        item
        for item in session["readiness"]["checks"]
        if item["key"] == "project_understanding"
    )
    assert project_check["ready"] is True
    assert session["execution_contract"] == {
        "runner_script": "ops/runner.py",
        "test_runner_script": "checks/test_primary.py",
    }


def test_assisted_manifest_rejects_paths_outside_the_pinned_commit(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path / "unsafe-manifest",
        {
            "pipeline.py": "def detection(image): return []\n",
            "runner.py": "print('run')\n",
            "test_runner.py": "print('test')\n",
            "visiondoctor.yaml": """
components:
  - id: invented
    type: perception_processor
    source_paths: [../outside.py]
""",
        },
    )

    with pytest.raises(ValueError, match="不存在的组件文件"):
        ProjectIngestionService().discover(repository)


def test_assisted_manifest_can_split_one_scanned_ros_package_into_components(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path / "split-package",
        {
            "ros_ws/src/combined/package.xml": (
                "<package><name>combined</name></package>"
            ),
            "ros_ws/src/combined/camera.py": (
                "import rclpy\ndef camera_driver(): return 'rgb'\n"
            ),
            "ros_ws/src/combined/ocr.py": "def ocr(rgb): return 'LOT-42'\n",
            "runner.py": "print('run')\n",
            "test_runner.py": "print('test')\n",
            "visiondoctor.yaml": """
components:
  - id: camera
    type: sensor
    source_paths: [ros_ws/src/combined/camera.py]
    consumes: []
    produces: [rgb]
  - id: reader
    type: perception_processor
    source_paths: [ros_ws/src/combined/ocr.py]
    consumes: [rgb]
    produces: [text]
relations:
  - source: camera
    type: data_flow
    target: reader
    signal: rgb
""",
        },
    )

    project = ProjectIngestionService().discover(repository)
    declared = [item for item in project.components if item.inferred_by == "confirmed"]

    assert len(declared) == 2
    assigned_paths = [path for item in project.components for path in item.source_paths]
    assert assigned_paths.count("ros_ws/src/combined/camera.py") == 1
    assert assigned_paths.count("ros_ws/src/combined/ocr.py") == 1
    assert project.inventory["components"] == len(project.components)
    assert project.inventory["relations"] == len(project.relations)


def test_catalog_carries_valid_human_confirmation_across_new_commit(
    tmp_path: Path,
) -> None:
    repository = _repository(
        tmp_path / "evolving-project",
        {
            "vision.py": "def detection(image): return []\n",
            "runner.py": "print('production')\n",
            "tools/run.py": "print('development')\n",
            "test_runner.py": "print('test')\n",
        },
    )
    catalog = ProjectCatalog(tmp_path / "projects")
    first = catalog.ingest(repository)
    question = next(item for item in first.ambiguities if item.key == "runner_entrypoint")
    confirmed = catalog.confirm(first.project_id, question.ambiguity_id, "runner.py")

    (repository / "vision.py").write_text(
        "def detection(image):\n    return [{'label': 'box'}]\n", encoding="utf-8"
    )
    _git(repository, "add", "vision.py")
    _git(repository, "commit", "-q", "-m", "update detector")
    updated = catalog.ingest(repository)

    carried = next(item for item in updated.ambiguities if item.key == "runner_entrypoint")
    assert carried.status is AmbiguityStatus.CONFIRMED
    assert carried.selected_option_id == "runner.py"
    assert updated.validation.selected_runner == "runner.py"
    assert updated.created_at == confirmed.created_at
    assert updated.revision == confirmed.revision + 1


def test_project_http_api_exposes_catalog_and_user_confirmation(
    tmp_path: Path,
) -> None:
    repository_path = _repository(
        tmp_path / "api-project",
        {
            "vision.py": "def detection(image): return []\n",
            "runner.py": "print('run')\n",
            "test_runner.py": "print('test')\n",
            "config/camera_calibration.yaml": "fx: 600\n",
            "config/backup_calibration.yaml": "fx: 590\n",
        },
    )
    settings = ApiSettings(
        data_root=tmp_path / "server",
        database_path=tmp_path / "server" / "state.sqlite3",
    )
    runs = SqliteRunRepository(settings.database_path)
    app = create_app(settings, runs)

    with TestClient(app) as client:
        created = client.post("/api/v1/sessions", json={})
        assert created.status_code == 201
        session_id = created.json()["session_id"]
        connected = client.post(
            f"/api/v1/sessions/{session_id}/repository",
            json={
                "repository_path": str(repository_path),
                "semantic_understanding": False,
            },
        )
        assert connected.status_code == 200
        project = connected.json()["project"]
        project_id = project["project_id"]

        listed = client.get("/api/v1/projects")
        fetched = client.get(f"/api/v1/projects/{project_id}")
        assert listed.status_code == 200
        assert [item["project_id"] for item in listed.json()] == [project_id]
        assert listed.json()[0]["component_count"] == len(project["components"])
        assert fetched.status_code == 200
        ambiguity = next(
            item
            for item in fetched.json()["ambiguities"]
            if item["key"] == "production_calibration"
        )
        selection = ambiguity["options"][0]["option_id"]
        confirmed = client.post(
            f"/api/v1/sessions/{session_id}/project/confirm",
            json={
                "ambiguity_id": ambiguity["ambiguity_id"],
                "option_id": selection,
            },
        )
        sessions = client.get("/api/v1/sessions")

    assert confirmed.status_code == 200
    assert sessions.status_code == 200
    assert sessions.json()[0]["project"] == {
        "project_id": project_id,
        "name": project["name"],
        "head_commit": project["source"]["head_commit"],
    }
    confirmed_project = confirmed.json()["project"]
    question = next(
        item
        for item in confirmed_project["ambiguities"]
        if item["ambiguity_id"] == ambiguity["ambiguity_id"]
    )
    assert question["status"] == "confirmed"
