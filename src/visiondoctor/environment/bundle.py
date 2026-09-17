"""One bounded observation of a machine-vision cell.

A bundle is what the product reads instead of the cell itself: a manifest, the
artifacts it lists, and enough time information to say whether two sources may
be compared at all.  Nothing here knows which kind of environment produced it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

MANIFEST_NAME = "bundle.json"
SCHEMA_VERSION = "observation-bundle/v1"


class BundleModel(BaseModel):
    """Manifest entries stay frozen but tolerate fields the product does not model."""

    model_config = ConfigDict(extra="allow", frozen=True)


class Clock(BundleModel):
    source: str
    quality: str
    alignment_error_ms: float | None = None


class Artifact(BundleModel):
    path: str
    media_type: str
    sha256: str
    captured_at: datetime
    clock_domain: str
    phase: str | None = None
    #: Declared by the exporter: ``software`` for what the running program read,
    #: wrote or logged; ``observation`` for sensors, cameras and motion records.
    layer: str | None = None

    @property
    def artifact_id(self) -> str:
        return self.path


class TimelineEvent(BundleModel):
    """A coarse marker on the bundle's own clock.

    Cross-source comparison rests on artifacts, which each name their domain;
    a timeline event that names none belongs to the bundle's clock.
    """

    at: datetime
    event: str
    clock_domain: str | None = None


class TaskResult(BundleModel):
    part_id: str
    classification: str
    success: bool


class ObservationBundle(BundleModel):
    schema_version: str
    run_id: str
    source: str
    created_at: datetime
    clock: Clock
    project_revision: dict[str, str]
    observation_started_at: datetime | None = None
    timeline: tuple[TimelineEvent, ...] = ()
    results: tuple[TaskResult, ...] = ()
    artifacts: tuple[Artifact, ...] = ()

    @property
    def clock_domains(self) -> frozenset[str]:
        return frozenset(item.clock_domain for item in self.artifacts)

    @property
    def cross_source_comparable(self) -> bool:
        """Whether artifacts from different sources may be placed on one axis.

        A shared host does not align simulation time and UTC. Distinct domains
        need an explicitly recorded alignment; pairwise tools still require
        actual timestamp mappings before subtracting across clock domains.
        """

        domains = self.clock_domains
        return (bool(domains) and len(domains) == 1 and "" not in domains) or (
            self.clock.alignment_error_ms is not None and self.clock.alignment_error_ms >= 0
        )

    @property
    def succeeded(self) -> bool:
        return bool(self.results) and all(item.success for item in self.results)


def load_manifest(directory: Path) -> ObservationBundle:
    payload: dict[str, Any] = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    return ObservationBundle.model_validate(payload)
