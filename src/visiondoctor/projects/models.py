from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _now() -> datetime:
    return datetime.now(UTC)


class ProjectModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentKind(StrEnum):
    SENSOR = "sensor"
    PERCEPTION_PROCESSOR = "perception_processor"
    GEOMETRY_PROCESSOR = "geometry_processor"
    MODEL = "model"
    ROBOT_INTERFACE = "robot_interface"
    TASK_EXECUTOR = "task_executor"
    DATA_SOURCE = "data_source"
    VISUALIZATION = "visualization"
    STORAGE = "storage"
    EXTERNAL_SERVICE = "external_service"
    UNKNOWN = "unknown"


class AssetKind(StrEnum):
    CONFIGURATION = "configuration"
    CALIBRATION = "calibration"
    MODEL = "model"
    DATASET = "dataset"
    RECORDING = "recording"
    ROBOT_DESCRIPTION = "robot_description"
    DEPLOYMENT = "deployment"
    DOCUMENTATION = "documentation"


class RelationKind(StrEnum):
    DATA_FLOW = "data_flow"
    USES = "uses"
    CONFIGURED_BY = "configured_by"
    CALIBRATED_BY = "calibrated_by"
    TRANSFORM_TO = "transform_to"
    VALIDATED_BY = "validated_by"


class UnderstandingStatus(StrEnum):
    DISCOVERED = "discovered"
    UNDERSTOOD = "understood"
    NEEDS_CONFIRMATION = "needs_confirmation"


class AmbiguityStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"


class SourceProject(ProjectModel):
    repository_path: str = Field(min_length=1)
    head_commit: str = Field(min_length=7)
    branch: str = Field(min_length=1)
    remote_url: str | None = None


class ProjectComponent(ProjectModel):
    component_id: str = Field(pattern=r"^COMP-[a-f0-9]{12}$")
    kind: ComponentKind
    name: str = Field(min_length=1, max_length=200)
    role: str = Field(default="", max_length=1000)
    source_paths: tuple[str, ...] = Field(min_length=1, max_length=50)
    runtime: dict[str, str] = Field(default_factory=dict)
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    confidence: float = Field(ge=0, le=1)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=30)
    inferred_by: Literal["scanner", "model", "confirmed"] = "scanner"


class ProjectAsset(ProjectModel):
    asset_id: str = Field(pattern=r"^ASSET-[a-f0-9]{12}$")
    kind: AssetKind
    path: str = Field(min_length=1, max_length=1000)
    size_bytes: int = Field(ge=0)
    fingerprint: str = Field(min_length=7, max_length=128)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=20)


class ProjectRelation(ProjectModel):
    relation_id: str = Field(pattern=r"^REL-[a-f0-9]{12}$")
    source_id: str = Field(min_length=1)
    kind: RelationKind
    target_id: str = Field(min_length=1)
    signal: str = Field(default="", max_length=200)
    confidence: float = Field(ge=0, le=1)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=20)
    inferred_by: Literal["scanner", "model", "confirmed"] = "scanner"


class AmbiguityOption(ProjectModel):
    option_id: str = Field(min_length=1, max_length=200)
    label: str = Field(min_length=1, max_length=500)
    evidence: tuple[str, ...] = Field(default=(), max_length=20)


class ProjectAmbiguity(ProjectModel):
    ambiguity_id: str = Field(pattern=r"^AMB-[a-f0-9]{12}$")
    key: str = Field(min_length=1, max_length=300)
    question: str = Field(min_length=1, max_length=1200)
    options: tuple[AmbiguityOption, ...] = Field(min_length=2, max_length=30)
    recommended_option_id: str | None = None
    selected_option_id: str | None = None
    status: AmbiguityStatus = AmbiguityStatus.PENDING
    evidence: tuple[str, ...] = Field(default=(), max_length=30)
    inferred_by: Literal["scanner", "model"] = "scanner"

    @model_validator(mode="after")
    def selected_option_exists(self) -> ProjectAmbiguity:
        option_ids = {item.option_id for item in self.options}
        if self.recommended_option_id and self.recommended_option_id not in option_ids:
            raise ValueError("recommended ambiguity option is unavailable")
        if self.selected_option_id and self.selected_option_id not in option_ids:
            raise ValueError("selected ambiguity option is unavailable")
        if self.status is AmbiguityStatus.CONFIRMED and not self.selected_option_id:
            raise ValueError("confirmed ambiguity requires a selection")
        return self


class RuntimeProfile(ProjectModel):
    frameworks: tuple[str, ...] = ()
    ros_packages: tuple[str, ...] = ()
    entrypoints: tuple[str, ...] = ()
    launch_files: tuple[str, ...] = ()
    containers: tuple[str, ...] = ()
    evidence: dict[str, tuple[str, ...]] = Field(default_factory=dict)


class ValidationProfile(ProjectModel):
    dataset_candidates: tuple[str, ...] = ()
    runner_candidates: tuple[str, ...] = ()
    test_candidates: tuple[str, ...] = ()
    selected_runner: str | None = None
    selected_test_runner: str | None = None


class ProjectKnowledge(ProjectModel):
    summary: str = Field(default="", max_length=6000)
    confirmed_facts: dict[str, str] = Field(default_factory=dict)
    model: str = Field(default="", max_length=200)


class VisionProject(ProjectModel):
    project_id: str = Field(pattern=r"^PROJECT-[a-f0-9]{12}$")
    name: str = Field(min_length=1, max_length=200)
    source: SourceProject
    inventory: dict[str, int] = Field(default_factory=dict)
    components: tuple[ProjectComponent, ...] = ()
    assets: tuple[ProjectAsset, ...] = ()
    relations: tuple[ProjectRelation, ...] = ()
    runtime: RuntimeProfile = Field(default_factory=RuntimeProfile)
    validation: ValidationProfile = Field(default_factory=ValidationProfile)
    ambiguities: tuple[ProjectAmbiguity, ...] = ()
    knowledge: ProjectKnowledge = Field(default_factory=ProjectKnowledge)
    understanding_status: UnderstandingStatus = UnderstandingStatus.DISCOVERED
    incident_ids: tuple[str, ...] = ()
    revision: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def graph_is_consistent(self) -> VisionProject:
        component_ids = [item.component_id for item in self.components]
        asset_ids = [item.asset_id for item in self.assets]
        node_ids = set(component_ids) | set(asset_ids)
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("project contains duplicate component ids")
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("project contains duplicate asset ids")
        relation_ids = [item.relation_id for item in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("project contains duplicate relation ids")
        for relation in self.relations:
            if relation.source_id not in node_ids or relation.target_id not in node_ids:
                raise ValueError("project relation references an unavailable node")
        ambiguity_ids = [item.ambiguity_id for item in self.ambiguities]
        if len(ambiguity_ids) != len(set(ambiguity_ids)):
            raise ValueError("project contains duplicate ambiguity ids")
        return self

    def model_context(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "head_commit": self.source.head_commit,
            "frameworks": self.runtime.frameworks,
            "components": [
                {
                    "component_id": item.component_id,
                    "kind": item.kind,
                    "name": item.name,
                    "role": item.role,
                    "source_paths": item.source_paths,
                    "consumes": item.consumes,
                    "produces": item.produces,
                    "confidence": item.confidence,
                    "evidence": item.evidence,
                }
                for item in self.components
            ],
            "assets": [
                {
                    "asset_id": item.asset_id,
                    "kind": item.kind,
                    "path": item.path,
                    "evidence": item.evidence,
                }
                for item in self.assets
            ],
            "relations": [item.model_dump(mode="json") for item in self.relations],
            "runtime": self.runtime.model_dump(mode="json"),
            "validation": self.validation.model_dump(mode="json"),
            "pending_ambiguities": [
                item.model_dump(mode="json")
                for item in self.ambiguities
                if item.status is AmbiguityStatus.PENDING
            ],
            "confirmed_ambiguities": [
                item.model_dump(mode="json")
                for item in self.ambiguities
                if item.status is AmbiguityStatus.CONFIRMED
            ],
            "confirmed_facts": self.knowledge.confirmed_facts,
            "summary": self.knowledge.summary,
        }
