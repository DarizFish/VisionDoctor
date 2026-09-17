"""The case is the lifecycle root: what was observed, and what it supports."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from visiondoctor.environment import ObservationBundle
from visiondoctor.repair import ProjectBinding

from .chain import GUIDED_SEGMENTS, Hypothesis, Segment, SegmentFinding, SegmentStatus, chain_for
from .repair import RepairPlan

#: How deep an observation reaches.  The software layer is what the running
#: program read, wrote and logged; the source layer is how it is written.
Layer = Literal["observation", "software", "source"]
LAYER_ORDER: dict[str, int] = {"observation": 0, "software": 1, "source": 2}


class Evidence(BaseModel):
    """An immutable thing that was actually observed, and where it came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    bundle_id: str
    reference: str
    media_type: str
    captured_at: datetime
    clock_domain: str
    summary: str
    sha256: str | None = None
    # Small computed measurements remain inspectable after a case is reloaded.
    content: dict[str, Any] | None = None
    phase: str | None = None
    layer: Layer = "observation"


class Case:
    """One diagnosis, from first observation to a demarcation that holds up."""

    def __init__(self, case_id: str, title: str, *, guided_motion: bool = False) -> None:
        self.case_id = case_id
        self.title = title
        self.guided_motion = guided_motion
        #: The repository this case is about.  Source is read at its revision,
        #: never from a working tree that may have moved on.
        self.project: ProjectBinding | None = None
        self.observations: list[ObservationBundle] = []
        self.evidence: list[Evidence] = []
        self.findings: list[SegmentFinding] = []
        self.hypotheses: list[Hypothesis] = []
        self.repair_plans: list[RepairPlan] = []
        #: Evidence the host actually delivered.  A model claiming to have
        #: checked something it never asked for cannot cite it.
        self.examined: set[str] = set()
        #: What has been said so far, carried from one turn into the next so the
        #: case reads as one conversation.  The ledger above stays the authority
        #: on what counts as evidence; this is only memory of how it got there.
        self.transcript: list[dict] = []

    @property
    def segments(self) -> tuple[Segment, ...]:
        return chain_for(guided_motion=self.guided_motion)

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(item.evidence_id for item in self.evidence)

    def bind_project(self, binding: ProjectBinding) -> ProjectBinding:
        self.project = binding
        return binding

    def extend_for_guided_motion(self) -> tuple[Segment, ...]:
        """Vision output is consumed by an actuator, so three more segments exist."""

        self.guided_motion = True
        return GUIDED_SEGMENTS

    def admit(self, bundle: ObservationBundle) -> tuple[Evidence, ...]:
        """Take one observation into the case, artifact by artifact."""

        self.observations.append(bundle)
        admitted: list[Evidence] = []

        def take(evidence: Evidence) -> None:
            self.evidence.append(evidence)
            admitted.append(evidence)

        for artifact in bundle.artifacts:
            take(
                Evidence(
                    evidence_id=self._next_id(),
                    bundle_id=bundle.run_id,
                    reference=artifact.path,
                    media_type=artifact.media_type,
                    captured_at=artifact.captured_at,
                    clock_domain=artifact.clock_domain,
                    sha256=artifact.sha256,
                    phase=artifact.phase,
                    layer="software" if artifact.layer == "software" else "observation",
                    summary=f"{artifact.path} collected from {bundle.source}",
                )
            )
        for result in bundle.results:
            verdict = "succeeded" if result.success else f"failed as {result.classification}"
            take(
                Evidence(
                    evidence_id=self._next_id(),
                    bundle_id=bundle.run_id,
                    reference=f"results/{result.part_id}",
                    media_type="application/json",
                    captured_at=bundle.created_at,
                    clock_domain=bundle.clock.source,
                    summary=f"task {result.part_id} {verdict}",
                )
            )
        return tuple(admitted)

    def record(self, finding: SegmentFinding) -> SegmentFinding:
        """Revise one target; previous judgments remain in the turn history."""

        self.validate_finding(finding)
        key = finding.target_id or finding.segment.value
        self.findings = [
            item for item in self.findings if (item.target_id or item.segment.value) != key
        ]
        self.findings.append(finding)
        return finding

    def validate_finding(self, finding: SegmentFinding) -> None:
        self._require_known(finding.evidence_ids)
        self._require_target(finding.target_id, finding.segment)
        if finding.status is not SegmentStatus.UNTESTED and not self.running_evidence(
            finding.evidence_ids
        ):
            raise ValueError(
                f"{finding.status} on {finding.target_id or finding.segment.value} rests on "
                "source alone; source explains a located fault but never locates or clears one"
            )

    def layer_of(self, evidence_id: str) -> str:
        for item in self.evidence:
            if item.evidence_id == evidence_id:
                return item.layer
        raise KeyError(f"{evidence_id} is not in this case")

    def running_evidence(self, evidence_ids: tuple[str, ...]) -> tuple[str, ...]:
        """The citations that describe what actually ran, not how it is written."""

        return tuple(name for name in evidence_ids if self.layer_of(name) != "source")

    def validate_hypothesis(self, hypothesis: Hypothesis) -> None:
        self._require_known(hypothesis.evidence_ids + hypothesis.counter_evidence_ids)
        self._require_target(hypothesis.target_id, hypothesis.target_segment)

    def propose(self, hypothesis: Hypothesis) -> Hypothesis:
        self.validate_hypothesis(hypothesis)
        self.hypotheses = [
            item for item in self.hypotheses if item.hypothesis_id != hypothesis.hypothesis_id
        ]
        self.hypotheses.append(hypothesis)
        return hypothesis

    def demarcation(self) -> dict[Segment, SegmentStatus]:
        verdicts = dict.fromkeys(self.segments, SegmentStatus.UNTESTED)
        rank = {SegmentStatus.UNTESTED: 0, SegmentStatus.CLEARED: 1, SegmentStatus.SUSPECT: 2}
        for finding in self.findings:
            if rank[finding.status] > rank[verdicts.get(finding.segment, SegmentStatus.UNTESTED)]:
                verdicts[finding.segment] = finding.status
        return verdicts

    @staticmethod
    def _require_target(target_id: str | None, segment: Segment) -> None:
        if target_id:
            from .graph import target_segment

            if target_segment(target_id) is not segment:
                raise ValueError(f"graph target {target_id} does not belong to {segment.value}")

    def add_evidence(self, evidence: Evidence) -> Evidence:
        """Take in evidence a tool produced, and mark it as actually seen."""

        self.evidence.append(evidence)
        self.examined.add(evidence.evidence_id)
        return evidence

    def next_evidence_id(self) -> str:
        return self._next_id()

    def _require_known(self, evidence_ids: tuple[str, ...]) -> None:
        unknown = sorted(set(evidence_ids) - self.evidence_ids)
        if unknown:
            raise ValueError(f"case {self.case_id} holds no evidence {', '.join(unknown)}")
        unseen = sorted(set(evidence_ids) - self.examined)
        if unseen:
            raise ValueError(f"case {self.case_id} never delivered {', '.join(unseen)}")

    def _next_id(self) -> str:
        return f"EV-{len(self.evidence) + 1:03d}"
