from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from visiondoctor.llm import ModelGateway
from visiondoctor.projects.ingestion import ProjectIngestionService
from visiondoctor.projects.models import (
    AmbiguityStatus,
    UnderstandingStatus,
    VisionProject,
)
from visiondoctor.projects.resolver import ProjectSemanticResolver


class ProjectCatalog:
    """Persistent canonical projects shared by multiple diagnosis sessions."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def ingest(self, repository: Path) -> VisionProject:
        discovered = ProjectIngestionService().discover(repository)
        with self._lock:
            path = self._path(discovered.project_id)
            if path.is_file():
                previous = self.get(discovered.project_id)
                if previous.source.head_commit == discovered.source.head_commit:
                    return previous
                discovered = self._carry_forward_confirmations(previous, discovered)
            self._save(discovered)
            return discovered

    def understand(self, project_id: str, gateway: ModelGateway) -> VisionProject:
        project = self.get(project_id)
        discovered = ProjectIngestionService().discover(
            Path(project.source.repository_path)
        )
        if (
            discovered.project_id != project.project_id
            or discovered.source.head_commit != project.source.head_commit
        ):
            raise ValueError(
                "椤圭洰婧愮爜鐗堟湰宸叉敼鍙橈紝璇峰厛閲嶆柊杩炴帴椤圭洰"
            )
        semantic_base = self._semantic_base(project, discovered)
        resolved = ProjectSemanticResolver(gateway).understand(semantic_base)
        with self._lock:
            current = self.get(project_id)
            if current.revision != project.revision:
                raise ValueError("项目在理解过程中已更新，请重新执行项目理解")
            self._save(resolved)
            return resolved

    @classmethod
    def _semantic_base(
        cls, previous: VisionProject, discovered: VisionProject
    ) -> VisionProject:
        """Rebuild the deterministic layer and retain only valid human knowledge."""

        base = cls._carry_forward_confirmations(previous, discovered)
        node_ids = {
            *(item.component_id for item in base.components),
            *(item.asset_id for item in base.assets),
        }
        retained_model_confirmations = []
        facts = dict(base.knowledge.confirmed_facts)
        existing_ambiguity_ids = {item.ambiguity_id for item in base.ambiguities}
        for ambiguity in previous.ambiguities:
            if (
                ambiguity.inferred_by != "model"
                or ambiguity.status is not AmbiguityStatus.CONFIRMED
                or not ambiguity.selected_option_id
                or ambiguity.selected_option_id not in node_ids
                or any(option.option_id not in node_ids for option in ambiguity.options)
                or ambiguity.ambiguity_id in existing_ambiguity_ids
            ):
                continue
            retained_model_confirmations.append(ambiguity)
            existing_ambiguity_ids.add(ambiguity.ambiguity_id)
            facts[ambiguity.key] = ambiguity.selected_option_id
        ambiguities = (*base.ambiguities, *retained_model_confirmations)
        pending = any(item.status is AmbiguityStatus.PENDING for item in ambiguities)
        return base.model_copy(
            update={
                "ambiguities": ambiguities,
                "knowledge": base.knowledge.model_copy(
                    update={"confirmed_facts": facts, "summary": "", "model": ""}
                ),
                "understanding_status": (
                    UnderstandingStatus.NEEDS_CONFIRMATION
                    if pending
                    else UnderstandingStatus.DISCOVERED
                ),
                "revision": previous.revision,
            }
        )

    def confirm(self, project_id: str, ambiguity_id: str, option_id: str) -> VisionProject:
        with self._lock:
            project = self.get(project_id)
            ambiguities = []
            selected_key = ""
            found = False
            for ambiguity in project.ambiguities:
                if ambiguity.ambiguity_id != ambiguity_id:
                    ambiguities.append(ambiguity)
                    continue
                option_ids = {item.option_id for item in ambiguity.options}
                if option_id not in option_ids:
                    raise ValueError("所选答案不属于这个待确认问题")
                found = True
                selected_key = ambiguity.key
                ambiguities.append(
                    ambiguity.model_copy(
                        update={
                            "selected_option_id": option_id,
                            "status": AmbiguityStatus.CONFIRMED,
                        }
                    )
                )
            if not found:
                raise KeyError(ambiguity_id)
            validation = project.validation
            if selected_key == "runner_entrypoint":
                validation = validation.model_copy(update={"selected_runner": option_id})
            elif selected_key == "test_entrypoint":
                validation = validation.model_copy(update={"selected_test_runner": option_id})
            facts = dict(project.knowledge.confirmed_facts)
            facts[selected_key] = option_id
            pending = any(item.status is AmbiguityStatus.PENDING for item in ambiguities)
            updated = project.model_copy(
                update={
                    "ambiguities": tuple(ambiguities),
                    "validation": validation,
                    "knowledge": project.knowledge.model_copy(
                        update={"confirmed_facts": facts}
                    ),
                    "understanding_status": (
                        UnderstandingStatus.NEEDS_CONFIRMATION
                        if pending
                        else UnderstandingStatus.UNDERSTOOD
                    ),
                    "revision": project.revision + 1,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._save(updated)
            return updated

    def record_incident(self, project_id: str, incident_id: str) -> VisionProject:
        with self._lock:
            project = self.get(project_id)
            incident_ids = tuple(dict.fromkeys((*project.incident_ids, incident_id)))
            updated = project.model_copy(
                update={
                    "incident_ids": incident_ids,
                    "revision": project.revision + 1,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._save(updated)
            return updated

    def get(self, project_id: str) -> VisionProject:
        path = self._path(project_id)
        if not path.is_file():
            raise KeyError(project_id)
        return VisionProject.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> tuple[VisionProject, ...]:
        values: list[VisionProject] = []
        for path in self.root.glob("PROJECT-*/project.json"):
            try:
                values.append(VisionProject.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        values.sort(key=lambda item: item.updated_at, reverse=True)
        return tuple(values)

    def _path(self, project_id: str) -> Path:
        import re

        if not re.fullmatch(r"PROJECT-[a-f0-9]{12}", project_id):
            raise KeyError(project_id)
        return self.root / project_id / "project.json"

    @staticmethod
    def _carry_forward_confirmations(
        previous: VisionProject, discovered: VisionProject
    ) -> VisionProject:
        """Keep human facts only when they remain valid in the new commit graph."""

        previous_by_key = {item.key: item for item in previous.ambiguities}
        ambiguities = []
        facts = dict(discovered.knowledge.confirmed_facts)
        validation = discovered.validation
        for ambiguity in discovered.ambiguities:
            if ambiguity.status is AmbiguityStatus.CONFIRMED:
                selected = ambiguity.selected_option_id
            else:
                old = previous_by_key.get(ambiguity.key)
                selected = (
                    old.selected_option_id
                    if old is not None and old.status is AmbiguityStatus.CONFIRMED
                    else None
                )
                option_ids = {item.option_id for item in ambiguity.options}
                if selected not in option_ids:
                    selected = None
                if selected is not None:
                    ambiguity = ambiguity.model_copy(
                        update={
                            "selected_option_id": selected,
                            "status": AmbiguityStatus.CONFIRMED,
                        }
                    )
            if selected is not None:
                facts[ambiguity.key] = selected
                if ambiguity.key == "runner_entrypoint":
                    validation = validation.model_copy(update={"selected_runner": selected})
                elif ambiguity.key == "test_entrypoint":
                    validation = validation.model_copy(
                        update={"selected_test_runner": selected}
                    )
            ambiguities.append(ambiguity)
        pending = any(item.status is AmbiguityStatus.PENDING for item in ambiguities)
        return discovered.model_copy(
            update={
                "created_at": previous.created_at,
                "incident_ids": previous.incident_ids,
                "revision": previous.revision + 1,
                "ambiguities": tuple(ambiguities),
                "validation": validation,
                "knowledge": discovered.knowledge.model_copy(
                    update={"confirmed_facts": facts}
                ),
                "understanding_status": (
                    UnderstandingStatus.NEEDS_CONFIRMATION
                    if pending
                    else UnderstandingStatus.DISCOVERED
                ),
            }
        )

    def _save(self, project: VisionProject) -> None:
        path = self._path(project.project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        payload = json.dumps(
            project.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True
        )
        with self._lock:
            temporary.write_text(payload + "\n", encoding="utf-8")
            temporary.replace(path)
