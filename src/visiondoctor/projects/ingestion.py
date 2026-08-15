from __future__ import annotations

import hashlib
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from visiondoctor.projects.manifest import apply_assisted_manifest
from visiondoctor.projects.models import (
    AmbiguityOption,
    AssetKind,
    ComponentKind,
    ProjectAmbiguity,
    ProjectAsset,
    ProjectComponent,
    ProjectKnowledge,
    ProjectRelation,
    RelationKind,
    RuntimeProfile,
    SourceProject,
    UnderstandingStatus,
    ValidationProfile,
    VisionProject,
)

_MAX_TRACKED_FILES = 12_000
_MAX_TEXT_FILE_BYTES = 256_000
_MAX_COMPONENTS = 250
_MAX_ASSETS = 1000

_SOURCE_SUFFIXES = {".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".cu", ".rs"}
_CONFIG_SUFFIXES = {".yaml", ".yml", ".json", ".toml", ".xml", ".ini", ".cfg"}
_MODEL_SUFFIXES = {".onnx", ".pt", ".pth", ".engine", ".weights", ".tflite"}
_DATA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".npy", ".npz", ".pcd"}
_RECORDING_SUFFIXES = {".bag", ".db3", ".mcap"}
_ROBOT_SUFFIXES = {".urdf", ".xacro", ".srdf"}
_DOC_NAMES = {"readme.md", "readme.rst", "readme.txt"}
_DEPLOYMENT_NAMES = {
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    "package.xml",
    "cmakelists.txt",
    "requirements.txt",
    "pyproject.toml",
}


@dataclass(frozen=True)
class _TrackedFile:
    path: str
    object_id: str
    size: int


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        check=False,
    )


def _identifier(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _safe_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
        raise ValueError("repository contains an unsafe tracked path")
    return path.as_posix()


class ProjectIngestionService:
    """Commit-pinned deterministic discovery for arbitrary Git repository layouts."""

    def discover(self, repository_path: Path) -> VisionProject:
        repository = repository_path.expanduser().resolve()
        if not repository.is_dir():
            raise ValueError("项目文件夹不存在")
        inside = _git(repository, "rev-parse", "--is-inside-work-tree")
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            raise ValueError("所选文件夹不是 Git 项目")
        head = _git(repository, "rev-parse", "HEAD").stdout.strip()
        if len(head) < 7:
            raise ValueError("项目没有可读取的 Git 版本")
        branch = _git(repository, "branch", "--show-current").stdout.strip() or "detached"
        remote_result = _git(repository, "remote", "get-url", "origin")
        remote = remote_result.stdout.strip() if remote_result.returncode == 0 else None
        tracked = self._tracked_files(repository, head)
        text_cache: dict[str, str] = {}

        def read_text(path: str) -> str:
            if path not in text_cache:
                item = next(entry for entry in tracked if entry.path == path)
                text_cache[path] = self._read_text(repository, head, item)
            return text_cache[path]

        frameworks, framework_evidence = self._frameworks(tracked, read_text)
        assets = self._assets(tracked)
        components = self._components(tracked, assets, read_text)
        relations = self._relations(components, assets, read_text)
        runtime = self._runtime(tracked, frameworks, framework_evidence, read_text)
        validation = self._validation(tracked)
        ambiguities = self._ambiguities(components, assets, validation)
        status = (
            UnderstandingStatus.NEEDS_CONFIRMATION
            if ambiguities
            else UnderstandingStatus.DISCOVERED
        )
        suffix_counts = Counter(
            PurePosixPath(item.path).suffix.lower() or "[no_extension]" for item in tracked
        )
        inventory = {
            "tracked_files": len(tracked),
            "source_files": sum(
                PurePosixPath(item.path).suffix.lower() in _SOURCE_SUFFIXES for item in tracked
            ),
            "configuration_files": sum(
                PurePosixPath(item.path).suffix.lower() in _CONFIG_SUFFIXES for item in tracked
            ),
            "components": len(components),
            "assets": len(assets),
            "relations": len(relations),
            **{f"suffix:{key}": count for key, count in suffix_counts.most_common(20)},
        }
        project_id = _identifier("PROJECT", str(repository).casefold())
        project = VisionProject(
            project_id=project_id,
            name=repository.name,
            source=SourceProject(
                repository_path=str(repository),
                head_commit=head,
                branch=branch,
                remote_url=remote,
            ),
            inventory=inventory,
            components=components,
            assets=assets,
            relations=relations,
            runtime=runtime,
            validation=validation,
            ambiguities=ambiguities,
            knowledge=ProjectKnowledge(),
            understanding_status=status,
        )
        manifest = next(
            (
                item
                for item in tracked
                if item.path.lower()
                in {"visiondoctor.yaml", "visiondoctor.yml", "visiondoctor.json"}
            ),
            None,
        )
        if manifest is not None:
            content = read_text(manifest.path)
            if not content:
                raise ValueError("项目辅助描述无法读取或超过大小限制")
            project = apply_assisted_manifest(
                project,
                manifest_path=manifest.path,
                content=content,
                tracked_paths={item.path for item in tracked},
            )
        return project

    @staticmethod
    def _tracked_files(repository: Path, commit: str) -> tuple[_TrackedFile, ...]:
        result = _git(repository, "ls-tree", "-r", "-l", commit)
        if result.returncode != 0:
            raise ValueError("无法读取项目文件清单")
        items: list[_TrackedFile] = []
        for line in result.stdout.splitlines():
            match = re.match(r"^\d+\s+blob\s+([0-9a-f]+)\s+(\d+|-)\t(.+)$", line)
            if not match:
                continue
            object_id, size_value, path_value = match.groups()
            path = _safe_path(path_value)
            size = int(size_value) if size_value != "-" else 0
            items.append(_TrackedFile(path=path, object_id=object_id, size=size))
            if len(items) > _MAX_TRACKED_FILES:
                raise ValueError("项目文件超过 12000 个，请排除生成物后重试")
        if not items:
            raise ValueError("项目当前版本没有可读取的文件")
        return tuple(items)

    @staticmethod
    def _read_text(repository: Path, commit: str, item: _TrackedFile) -> str:
        if item.size > _MAX_TEXT_FILE_BYTES:
            return ""
        suffix = PurePosixPath(item.path).suffix.lower()
        name = PurePosixPath(item.path).name.lower()
        text_suffixes = _SOURCE_SUFFIXES | _CONFIG_SUFFIXES | {".md", ".txt", ".launch.py"}
        if suffix not in text_suffixes and name not in _DEPLOYMENT_NAMES:
            return ""
        result = _git(repository, "show", f"{commit}:{item.path}")
        if result.returncode != 0 or "\x00" in result.stdout:
            return ""
        return result.stdout[:_MAX_TEXT_FILE_BYTES]

    @staticmethod
    def _frameworks(
        tracked: tuple[_TrackedFile, ...], read_text: Any
    ) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
        paths = {item.path.lower(): item.path for item in tracked}
        evidence: dict[str, set[str]] = defaultdict(set)
        for item in tracked:
            lower = item.path.lower()
            name = PurePosixPath(lower).name
            text = read_text(item.path).lower()
            if name == "package.xml" or "rclpy" in text or "rclcpp" in text:
                evidence["ROS 2"].add(item.path)
            if (
                name == "pyproject.toml"
                or lower.endswith("requirements.txt")
                or lower.endswith(".py")
            ):
                evidence["Python"].add(item.path)
            if name in {"cmakelists.txt", "package.xml"} or lower.endswith((".cpp", ".cc")):
                evidence["C++"].add(item.path)
            if name in {
                "dockerfile",
                "compose.yml",
                "compose.yaml",
                "docker-compose.yml",
                "docker-compose.yaml",
            }:
                evidence["Docker"].add(item.path)
            if "opencv" in text or re.search(r"(^|\W)cv2(\W|$)", text):
                evidence["OpenCV"].add(item.path)
            if "onnxruntime" in text or lower.endswith(".onnx"):
                evidence["ONNX Runtime"].add(item.path)
            if "tensorrt" in text or lower.endswith(".engine"):
                evidence["TensorRT"].add(item.path)
            if "moveit" in text or "moveit" in lower:
                evidence["MoveIt 2"].add(item.path)
            if "gazebo" in text or "ros_gz" in text or "gazebo" in lower:
                evidence["Gazebo"].add(item.path)
        del paths
        ordered = tuple(sorted(evidence))
        return ordered, {key: tuple(sorted(value)[:20]) for key, value in evidence.items()}

    def _assets(self, tracked: tuple[_TrackedFile, ...]) -> tuple[ProjectAsset, ...]:
        assets: list[ProjectAsset] = []
        for item in tracked:
            path = PurePosixPath(item.path)
            lower = item.path.lower()
            suffix = path.suffix.lower()
            name = path.name.lower()
            kind: AssetKind | None = None
            evidence = "file extension and repository path"
            if suffix in _MODEL_SUFFIXES:
                kind = AssetKind.MODEL
            elif suffix in _ROBOT_SUFFIXES:
                kind = AssetKind.ROBOT_DESCRIPTION
            elif suffix in _RECORDING_SUFFIXES:
                kind = AssetKind.RECORDING
            elif suffix in _DATA_SUFFIXES and any(
                token in lower for token in ("data", "dataset", "sample", "bag", "record")
            ):
                kind = AssetKind.DATASET
            elif suffix in _CONFIG_SUFFIXES:
                kind = (
                    AssetKind.CALIBRATION
                    if any(
                        token in lower
                        for token in ("calib", "intrinsic", "extrinsic", "handeye", "camera_info")
                    )
                    else AssetKind.CONFIGURATION
                )
            elif name in _DEPLOYMENT_NAMES:
                kind = AssetKind.DEPLOYMENT
            elif name in _DOC_NAMES:
                kind = AssetKind.DOCUMENTATION
            if kind is None:
                continue
            assets.append(
                ProjectAsset(
                    asset_id=_identifier("ASSET", item.path),
                    kind=kind,
                    path=item.path,
                    size_bytes=item.size,
                    fingerprint=item.object_id,
                    evidence=(evidence,),
                )
            )
            if len(assets) >= _MAX_ASSETS:
                break
        return tuple(assets)

    def _components(
        self,
        tracked: tuple[_TrackedFile, ...],
        assets: tuple[ProjectAsset, ...],
        read_text: Any,
    ) -> tuple[ProjectComponent, ...]:
        package_roots = {
            str(PurePosixPath(item.path).parent): item.path
            for item in tracked
            if PurePosixPath(item.path).name.lower() == "package.xml"
        }
        grouped: dict[str, list[_TrackedFile]] = defaultdict(list)
        source_files = [
            item
            for item in tracked
            if PurePosixPath(item.path).suffix.lower() in _SOURCE_SUFFIXES
            and not self._ignored_source(item.path)
        ]
        for item in source_files:
            root = next(
                (
                    package_root
                    for package_root in sorted(package_roots, key=len, reverse=True)
                    if not package_root
                    or item.path == package_root
                    or item.path.startswith(package_root + "/")
                ),
                None,
            )
            key = root if root is not None else item.path
            grouped[key].append(item)
        components: list[ProjectComponent] = []
        for key, files in sorted(grouped.items()):
            combined = "\n".join(read_text(item.path)[:80_000] for item in files[:8])
            kind, confidence, evidence = self._infer_component(key, combined)
            paths = tuple(item.path for item in files[:50])
            if (
                kind is ComponentKind.UNKNOWN
                and len(files) == 1
                and not self._looks_executable(combined)
            ):
                continue
            consumes, produces = self._signals(kind, combined)
            runtime: dict[str, str] = {}
            if key in package_roots:
                runtime = {"type": "ros2_package", "package_root": key or "."}
            elif any(path.endswith(".py") for path in paths):
                runtime = {"type": "python_module"}
            components.append(
                ProjectComponent(
                    component_id=_identifier("COMP", key),
                    kind=kind,
                    name=PurePosixPath(key).name or "repository-root",
                    role=self._role(kind),
                    source_paths=paths,
                    runtime=runtime,
                    consumes=consumes,
                    produces=produces,
                    confidence=confidence,
                    evidence=tuple(evidence[:20]),
                )
            )
            if len(components) >= _MAX_COMPONENTS:
                break
        for asset in assets:
            if asset.kind is AssetKind.MODEL and len(components) < _MAX_COMPONENTS:
                components.append(
                    ProjectComponent(
                        component_id=_identifier("COMP", asset.path, "model"),
                        kind=ComponentKind.MODEL,
                        name=PurePosixPath(asset.path).name,
                        role="视觉模型或推理权重",
                        source_paths=(asset.path,),
                        consumes=(),
                        produces=("model_inference",),
                        confidence=1.0,
                        evidence=(f"model asset: {asset.path}",),
                    )
                )
        return tuple(components)

    @staticmethod
    def _ignored_source(path: str) -> bool:
        parts = {part.lower() for part in PurePosixPath(path).parts}
        return bool(parts & {"test", "tests", "vendor", "third_party", "build", "dist", ".venv"})

    @staticmethod
    def _looks_executable(text: str) -> bool:
        lowered = text.lower()
        return any(
            token in lowered
            for token in ("def main(", "if __name__", "rclpy.init", "rclcpp::init", "fastapi(")
        )

    @staticmethod
    def _infer_component(path: str, text: str) -> tuple[ComponentKind, float, list[str]]:
        corpus = f"{path}\n{text[:200_000]}".lower()
        signals: dict[ComponentKind, tuple[str, ...]] = {
            ComponentKind.SENSOR: (
                "camera_driver", "realsense", "image_raw", "camerainfo", "videocapture",
            ),
            ComponentKind.PERCEPTION_PROCESSOR: (
                "detector", "detection", "segment", "ocr", "inference", "solvepnp",
                "onnxruntime", "yolo", "findcontours",
            ),
            ComponentKind.GEOMETRY_PROCESSOR: (
                "camera_to_base", "pose_transform", "transform", "tf2", "quaternion",
                "homography", "t_base_camera", "t_camera_object",
            ),
            ComponentKind.ROBOT_INTERFACE: (
                "moveit", "joint_states", "followjointtrajectory", "robot_interface",
                "target_tcp", "ur5",
            ),
            ComponentKind.TASK_EXECUTOR: (
                "task_executor", "pipeline", "workflow", "run_task", "execute_task",
            ),
            ComponentKind.DATA_SOURCE: (
                "dataset", "dataloader", "rosbag", "mcap", "image_loader",
            ),
            ComponentKind.VISUALIZATION: (
                "visualization", "visualizer", "rviz", "debug_image", "markerarray",
            ),
            ComponentKind.STORAGE: ("sqlite", "database", "artifact_store", "storage"),
            ComponentKind.EXTERNAL_SERVICE: (
                "fastapi", "grpc", "httpx", "flask", "external_service",
            ),
        }
        scored: list[tuple[int, ComponentKind, list[str]]] = []
        for kind, tokens in signals.items():
            found = [token for token in tokens if token in corpus]
            scored.append((len(found), kind, found))
        score, kind, found = max(scored, key=lambda item: (item[0], item[1].value))
        if score == 0:
            return ComponentKind.UNKNOWN, 0.35, (f"source candidate: {path}",)
        confidence = min(0.95, 0.55 + score * 0.10)
        return kind, confidence, [f"matched signal '{token}' in {path}" for token in found]

    @staticmethod
    def _signals(kind: ComponentKind, text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        lowered = text.lower()
        if kind is ComponentKind.SENSOR:
            outputs = ["rgb"]
            if "depth" in lowered or "pointcloud" in lowered:
                outputs.extend(("depth", "camera_info", "point_cloud"))
            return (), tuple(outputs)
        if kind is ComponentKind.PERCEPTION_PROCESSOR:
            consumes = ["rgb"]
            if "depth" in lowered:
                consumes.append("depth")
            if "ocr" in lowered:
                output = "text"
            elif "segment" in lowered:
                output = "segmentation_mask"
            elif "pose" in lowered or "solvepnp" in lowered:
                output = "object_pose_camera"
            else:
                output = "detections"
            return tuple(consumes), (output,)
        if kind is ComponentKind.GEOMETRY_PROCESSOR:
            return ("object_pose_camera", "camera_to_base_transform"), ("object_pose_base",)
        if kind is ComponentKind.ROBOT_INTERFACE:
            return ("object_pose_base",), ("robot_state", "actual_tcp_pose")
        if kind is ComponentKind.TASK_EXECUTOR:
            return ("perception_result",), ("task_result",)
        if kind is ComponentKind.VISUALIZATION:
            return ("perception_result",), ("visualization",)
        return (), ()

    @staticmethod
    def _role(kind: ComponentKind) -> str:
        return {
            ComponentKind.SENSOR: "采集视觉输入",
            ComponentKind.PERCEPTION_PROCESSOR: "从视觉输入提取任务结果",
            ComponentKind.GEOMETRY_PROCESSOR: "处理坐标、位姿或几何关系",
            ComponentKind.MODEL: "提供视觉推理模型",
            ComponentKind.ROBOT_INTERFACE: "连接机器人运动与状态",
            ComponentKind.TASK_EXECUTOR: "编排视觉任务执行",
            ComponentKind.DATA_SOURCE: "提供数据或回放输入",
            ComponentKind.VISUALIZATION: "展示视觉或机器人结果",
            ComponentKind.STORAGE: "保存运行数据与工件",
            ComponentKind.EXTERNAL_SERVICE: "连接外部服务",
            ComponentKind.UNKNOWN: "尚待确认的源码组件",
        }[kind]

    def _relations(
        self,
        components: tuple[ProjectComponent, ...],
        assets: tuple[ProjectAsset, ...],
        read_text: Any,
    ) -> tuple[ProjectRelation, ...]:
        relations: list[ProjectRelation] = []
        seen: set[tuple[str, RelationKind, str, str]] = set()
        for source in components:
            for target in components:
                if source.component_id == target.component_id:
                    continue
                shared = sorted(set(source.produces) & set(target.consumes))
                for signal in shared:
                    key = (source.component_id, RelationKind.DATA_FLOW, target.component_id, signal)
                    if key in seen:
                        continue
                    seen.add(key)
                    relations.append(
                        ProjectRelation(
                            relation_id=_identifier("REL", *map(str, key)),
                            source_id=source.component_id,
                            kind=RelationKind.DATA_FLOW,
                            target_id=target.component_id,
                            signal=signal,
                            confidence=0.72,
                            evidence=(f"producer and consumer share signal: {signal}",),
                        )
                    )
        for component in components:
            text = "\n".join(read_text(path) for path in component.source_paths[:8]).lower()
            for asset in assets:
                basename = PurePosixPath(asset.path).name.lower()
                if basename not in text and asset.path.lower() not in text:
                    continue
                kind = {
                    AssetKind.CALIBRATION: RelationKind.CALIBRATED_BY,
                    AssetKind.CONFIGURATION: RelationKind.CONFIGURED_BY,
                    AssetKind.MODEL: RelationKind.USES,
                    AssetKind.DATASET: RelationKind.VALIDATED_BY,
                }.get(asset.kind)
                if kind is None:
                    continue
                key = (component.component_id, kind, asset.asset_id, "")
                if key in seen:
                    continue
                seen.add(key)
                relations.append(
                    ProjectRelation(
                        relation_id=_identifier("REL", *map(str, key)),
                        source_id=component.component_id,
                        kind=kind,
                        target_id=asset.asset_id,
                        confidence=0.92,
                        evidence=(f"{component.source_paths[0]} references {asset.path}",),
                    )
                )
        return tuple(relations[:1000])

    @staticmethod
    def _runtime(
        tracked: tuple[_TrackedFile, ...],
        frameworks: tuple[str, ...],
        evidence: dict[str, tuple[str, ...]],
        read_text: Any,
    ) -> RuntimeProfile:
        del read_text
        paths = tuple(item.path for item in tracked)
        package_roots = tuple(
            sorted(
                str(PurePosixPath(path).parent) or "."
                for path in paths
                if PurePosixPath(path).name.lower() == "package.xml"
            )
        )
        entrypoints = tuple(
            path
            for path in paths
            if PurePosixPath(path).name.lower()
            in {"main.py", "server.py", "app.py", "runner.py", "launch.sh"}
        )[:50]
        launch_files = tuple(
            path for path in paths if path.endswith(".launch.py") or "/launch/" in path.lower()
        )[:100]
        containers = tuple(
            path
            for path in paths
            if PurePosixPath(path).name.lower()
            in {
                "dockerfile",
                "compose.yml",
                "compose.yaml",
                "docker-compose.yml",
                "docker-compose.yaml",
            }
        )[:50]
        return RuntimeProfile(
            frameworks=frameworks,
            ros_packages=package_roots,
            entrypoints=entrypoints,
            launch_files=launch_files,
            containers=containers,
            evidence=evidence,
        )

    @staticmethod
    def _validation(tracked: tuple[_TrackedFile, ...]) -> ValidationProfile:
        paths = tuple(item.path for item in tracked)
        datasets = tuple(
            path
            for path in paths
            if any(token in path.lower() for token in ("dataset", "test_data", "samples", "bags"))
            and PurePosixPath(path).suffix.lower()
            in _DATA_SUFFIXES | _RECORDING_SUFFIXES | {".json"}
        )[:100]
        runners = tuple(
            path
            for path in paths
            if PurePosixPath(path).name.lower()
            in {"runner.py", "run.py", "main.py", "server.py"}
        )[:30]
        tests = tuple(
            path
            for path in paths
            if PurePosixPath(path).name.lower() in {"test_runner.py", "pytest.ini"}
            or path.lower().startswith("tests/")
            and PurePosixPath(path).suffix.lower() == ".py"
        )[:100]
        return ValidationProfile(
            dataset_candidates=datasets,
            runner_candidates=runners,
            test_candidates=tests,
            selected_runner=runners[0] if len(runners) == 1 else None,
            selected_test_runner=tests[0] if len(tests) == 1 else None,
        )

    @staticmethod
    def _ambiguities(
        components: tuple[ProjectComponent, ...],
        assets: tuple[ProjectAsset, ...],
        validation: ValidationProfile,
    ) -> tuple[ProjectAmbiguity, ...]:
        ambiguities: list[ProjectAmbiguity] = []
        calibration = [item for item in assets if item.kind is AssetKind.CALIBRATION]
        if len(calibration) > 1:
            ambiguities.append(
                ProjectAmbiguity(
                    ambiguity_id=_identifier(
                        "AMB",
                        "production_calibration",
                        *(item.path for item in calibration),
                    ),
                    key="production_calibration",
                    question="检测到多份相机标定，实际运行使用哪一份？",
                    options=tuple(
                        AmbiguityOption(
                            option_id=item.asset_id,
                            label=item.path,
                            evidence=item.evidence,
                        )
                        for item in calibration[:30]
                    ),
                    evidence=tuple(item.path for item in calibration[:30]),
                )
            )
        if len(validation.runner_candidates) > 1:
            ambiguities.append(
                ProjectAmbiguity(
                    ambiguity_id=_identifier(
                        "AMB", "runner_entrypoint", *validation.runner_candidates
                    ),
                    key="runner_entrypoint",
                    question="项目存在多个启动入口，复现视觉任务时通常使用哪一个？",
                    options=tuple(
                        AmbiguityOption(option_id=path, label=path, evidence=(path,))
                        for path in validation.runner_candidates
                    ),
                    evidence=validation.runner_candidates,
                )
            )
        if len(validation.test_candidates) > 1:
            preferred = next(
                (
                    path
                    for path in validation.test_candidates
                    if PurePosixPath(path).name == "test_runner.py"
                ),
                None,
            )
            ambiguities.append(
                ProjectAmbiguity(
                    ambiguity_id=_identifier("AMB", "test_entrypoint", *validation.test_candidates),
                    key="test_entrypoint",
                    question="项目包含多组测试，哪一个最能复现当前视觉链路？",
                    options=tuple(
                        AmbiguityOption(option_id=path, label=path, evidence=(path,))
                        for path in validation.test_candidates[:30]
                    ),
                    recommended_option_id=preferred,
                    evidence=validation.test_candidates[:30],
                )
            )
        producers: dict[str, list[ProjectComponent]] = defaultdict(list)
        for component in components:
            for signal in component.produces:
                producers[signal].append(component)
        for signal, candidates in sorted(producers.items()):
            if len(candidates) <= 1 or len(ambiguities) >= 8:
                continue
            ambiguities.append(
                ProjectAmbiguity(
                    ambiguity_id=_identifier(
                        "AMB",
                        "producer",
                        signal,
                        *(item.component_id for item in candidates),
                    ),
                    key=f"producer:{signal}",
                    question=f"多个组件都可能产生“{signal}”，生产链路实际使用哪一个？",
                    options=tuple(
                        AmbiguityOption(
                            option_id=item.component_id,
                            label=f"{item.name}（{item.source_paths[0]}）",
                            evidence=item.evidence,
                        )
                        for item in candidates[:30]
                    ),
                    evidence=tuple(item.source_paths[0] for item in candidates[:30]),
                )
            )
        return tuple(ambiguities)
