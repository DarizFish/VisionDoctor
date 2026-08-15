from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any

import yaml

from visiondoctor.projects.models import (
    AmbiguityStatus,
    ComponentKind,
    ProjectComponent,
    ProjectRelation,
    RelationKind,
    UnderstandingStatus,
    VisionProject,
)

_ALLOWED_KEYS = {"project", "components", "relations", "runtime", "validation", "confirmations"}


def _identifier(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def apply_assisted_manifest(
    project: VisionProject,
    *,
    manifest_path: str,
    content: str,
    tracked_paths: set[str],
) -> VisionProject:
    """Apply a bounded user-authored mapping without weakening repository evidence checks."""

    try:
        value = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ValueError(f"{manifest_path} 不是有效的辅助项目描述") from exc
    if not isinstance(value, dict) or set(value) - _ALLOWED_KEYS:
        raise ValueError(f"{manifest_path} 包含不支持的顶层字段")
    scanner_components = list(project.components)
    aliases: dict[str, str] = {
        component.component_id: component.component_id for component in scanner_components
    }
    declared_components: list[ProjectComponent] = []
    declared_paths: set[str] = set()
    declared = value.get("components", [])
    if not isinstance(declared, list) or len(declared) > 250:
        raise ValueError("辅助项目描述中的 components 必须是有界数组")
    for index, item in enumerate(declared):
        if not isinstance(item, dict):
            raise ValueError("辅助项目描述中的组件必须是对象")
        allowed = {"id", "type", "name", "role", "source_paths", "consumes", "produces"}
        if set(item) - allowed:
            raise ValueError("辅助项目描述中的组件包含未知字段")
        alias = str(item.get("id") or f"component-{index + 1}").strip()
        if not alias or alias in aliases:
            raise ValueError("辅助项目描述中的组件 id 为空或重复")
        paths = _string_tuple(item.get("source_paths"), "component source_paths")
        if not paths or any(path not in tracked_paths for path in paths):
            raise ValueError("辅助项目描述引用了当前版本中不存在的组件文件")
        if declared_paths.intersection(paths):
            raise ValueError("辅助项目描述不能把同一文件分配给多个组件")
        kind = ComponentKind(str(item.get("type", ComponentKind.UNKNOWN)))
        name = str(item.get("name") or PurePosixPath(paths[0]).stem).strip()
        role = str(item.get("role") or "由项目辅助描述声明的组件").strip()
        consumes = _string_tuple(item.get("consumes", []), "component consumes")
        produces = _string_tuple(item.get("produces", []), "component produces")
        replacement = ProjectComponent(
            component_id=_identifier("COMP", manifest_path, alias),
            kind=kind,
            name=name,
            role=role,
            source_paths=paths,
            consumes=consumes,
            produces=produces,
            confidence=1.0,
            evidence=(f"declared in {manifest_path}",),
            inferred_by="confirmed",
        )
        declared_components.append(replacement)
        declared_paths.update(paths)
        aliases[alias] = replacement.component_id

    components: list[ProjectComponent] = []
    removed_component_ids: set[str] = set()
    for component in scanner_components:
        remaining_paths = tuple(
            path for path in component.source_paths if path not in declared_paths
        )
        if not remaining_paths:
            removed_component_ids.add(component.component_id)
            continue
        if remaining_paths != component.source_paths:
            component = component.model_copy(
                update={
                    "source_paths": remaining_paths,
                    "confidence": min(component.confidence, 0.5),
                    "evidence": (
                        *component.evidence[:19],
                        f"partially superseded by {manifest_path}",
                    ),
                }
            )
        components.append(component)
    components.extend(declared_components)
    aliases = {
        alias: component_id
        for alias, component_id in aliases.items()
        if component_id not in removed_component_ids
    }

    assets_by_path = {item.path: item.asset_id for item in project.assets}
    node_aliases = {**aliases, **assets_by_path}
    relations = [
        item
        for item in project.relations
        if item.source_id not in removed_component_ids
        and item.target_id not in removed_component_ids
    ]
    relation_keys = {
        (item.source_id, item.kind, item.target_id, item.signal) for item in relations
    }
    declared_relations = value.get("relations", [])
    if not isinstance(declared_relations, list) or len(declared_relations) > 500:
        raise ValueError("辅助项目描述中的 relations 必须是有界数组")
    for item in declared_relations:
        if not isinstance(item, dict) or set(item) - {"source", "type", "target", "signal"}:
            raise ValueError("辅助项目描述中的关系字段无效")
        source_alias = str(item.get("source", ""))
        target_alias = str(item.get("target", ""))
        if source_alias not in node_aliases or target_alias not in node_aliases:
            raise ValueError("辅助项目描述中的关系引用了未知组件或资产")
        source_id = node_aliases[source_alias]
        target_id = node_aliases[target_alias]
        kind = RelationKind(str(item.get("type", RelationKind.DATA_FLOW)))
        signal = str(item.get("signal", "")).strip()
        key = (source_id, kind, target_id, signal)
        if key in relation_keys:
            continue
        relation_keys.add(key)
        relations.append(
            ProjectRelation(
                relation_id=_identifier("REL", manifest_path, *map(str, key)),
                source_id=source_id,
                kind=kind,
                target_id=target_id,
                signal=signal,
                confidence=1.0,
                evidence=(f"declared in {manifest_path}",),
                inferred_by="confirmed",
            )
        )

    runtime = project.runtime
    runtime_value = value.get("runtime", {})
    if not isinstance(runtime_value, dict) or set(runtime_value) - {"entrypoints"}:
        raise ValueError("辅助项目描述中的 runtime 字段无效")
    if "entrypoints" in runtime_value:
        entrypoints = _existing_paths(
            runtime_value["entrypoints"], tracked_paths, "runtime entrypoints"
        )
        runtime = runtime.model_copy(update={"entrypoints": entrypoints})

    validation = project.validation
    validation_value = value.get("validation", {})
    if not isinstance(validation_value, dict) or set(validation_value) - {
        "runner",
        "test_runner",
        "datasets",
    }:
        raise ValueError("辅助项目描述中的 validation 字段无效")
    updates: dict[str, Any] = {}
    if validation_value.get("runner"):
        runner = str(validation_value["runner"])
        if runner not in tracked_paths:
            raise ValueError("辅助项目描述的运行入口不存在")
        if PurePosixPath(runner).suffix.lower() != ".py":
            raise ValueError("VisionDoctor 自动执行入口目前必须是 Python 脚本")
        updates["selected_runner"] = runner
    if validation_value.get("test_runner"):
        test_runner = str(validation_value["test_runner"])
        if test_runner not in tracked_paths:
            raise ValueError("辅助项目描述的测试入口不存在")
        if PurePosixPath(test_runner).suffix.lower() != ".py":
            raise ValueError("VisionDoctor 自动测试入口目前必须是 Python 脚本")
        updates["selected_test_runner"] = test_runner
    if "datasets" in validation_value:
        updates["dataset_candidates"] = _existing_paths(
            validation_value["datasets"], tracked_paths, "validation datasets"
        )
    validation = validation.model_copy(update=updates)

    facts = dict(project.knowledge.confirmed_facts)
    confirmations = value.get("confirmations", {})
    if not isinstance(confirmations, dict):
        raise ValueError("辅助项目描述中的 confirmations 必须是对象")
    ambiguities = []
    for ambiguity in project.ambiguities:
        selection = confirmations.get(ambiguity.key)
        if selection is None and ambiguity.key == "runner_entrypoint":
            selection = updates.get("selected_runner")
        if selection is None and ambiguity.key == "test_entrypoint":
            selection = updates.get("selected_test_runner")
        option_lookup = {option.label: option.option_id for option in ambiguity.options}
        option_ids = {option.option_id for option in ambiguity.options}
        selected = option_lookup.get(str(selection), str(selection)) if selection else None
        if selected is not None:
            if selected not in option_ids:
                raise ValueError("辅助项目描述确认了不属于待确认项的答案")
            ambiguity = ambiguity.model_copy(
                update={
                    "selected_option_id": selected,
                    "status": AmbiguityStatus.CONFIRMED,
                }
            )
            facts[ambiguity.key] = selected
        ambiguities.append(ambiguity)

    project_value = value.get("project", {})
    if not isinstance(project_value, dict) or set(project_value) - {"name"}:
        raise ValueError("辅助项目描述中的 project 字段无效")
    name = str(project_value.get("name") or project.name).strip()
    pending = any(item.status is AmbiguityStatus.PENDING for item in ambiguities)
    inventory = dict(project.inventory)
    inventory["components"] = len(components)
    inventory["relations"] = len(relations)
    return project.model_copy(
        update={
            "name": name,
            "inventory": inventory,
            "components": tuple(components),
            "relations": tuple(relations),
            "runtime": runtime,
            "validation": validation,
            "ambiguities": tuple(ambiguities),
            "knowledge": project.knowledge.model_copy(
                update={"confirmed_facts": facts}
            ),
            "understanding_status": (
                UnderstandingStatus.NEEDS_CONFIRMATION
                if pending
                else UnderstandingStatus.DISCOVERED
            ),
            "revision": project.revision + 1,
            "updated_at": datetime.now(UTC),
        }
    )


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > 100:
        raise ValueError(f"{label} 必须是有界字符串数组")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"{label} 不能包含空值")
    return result


def _existing_paths(value: Any, tracked_paths: set[str], label: str) -> tuple[str, ...]:
    paths = _string_tuple(value, label)
    if any(path not in tracked_paths for path in paths):
        raise ValueError(f"{label} 引用了当前版本中不存在的文件")
    return paths
