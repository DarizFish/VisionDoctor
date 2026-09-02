"""The case is the lifecycle root: what was observed, and what it supports."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from visiondoctor.environment import ObservationBundle

from .chain import GUIDED_SEGMENTS, Hypothesis, Segment, SegmentFinding, SegmentStatus, chain_for


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


class Case:
    """One diagnosis, from first observation to a demarcation that holds up."""

    def __init__(self, case_id: str, title: str, *, guided_motion: bool = False) -> None:
        self.case_id = case_id
        self.title = title
        self.guided_motion = guided_motion
        self.observations: list[ObservationBundle] = []
        self.evidence: list[Evidence] = []
        self.findings: list[SegmentFinding] = []
        self.hypotheses: list[Hypothesis] = []
        #: Evidence the host actually delivered.  A model claiming to have
        #: checked something it never asked for cannot cite it.
        self.examined: set[str] = set()

    @property
    def segments(self) -> tuple[Segment, ...]:
        return chain_for(guided_motion=self.guided_motion)

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(item.evidence_id for item in self.evidence)

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
        self._require_known(finding.evidence_ids)
        self.findings = [item for item in self.findings if item.segment is not finding.segment]
        self.findings.append(finding)
        return finding

    def propose(self, hypothesis: Hypothesis) -> Hypothesis:
        self._require_known(hypothesis.evidence_ids)
        self.hypotheses.append(hypothesis)
        return hypothesis

    def demarcation(self) -> dict[Segment, SegmentStatus]:
        verdicts = dict.fromkeys(self.segments, SegmentStatus.UNTESTED)
        for finding in self.findings:
            verdicts[finding.segment] = finding.status
        return verdicts

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
