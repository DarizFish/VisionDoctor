"""The read-only surface an investigation may touch.

Every call is executed by the host, which records it and hands back what it
actually read.  A model that says it examined something without asking for it
has no evidence to cite: only what passed through here is admitted.

Nothing on this surface can move the robot, change a camera, write a
configuration, or run a command.  Source code is not here either; it becomes
visible only after the diagnosis gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from visiondoctor.case import Case, Evidence
from visiondoctor.environment import FileBundleAdapter, ObservationBundle
from visiondoctor.multimodal import VisionGateway

from .transform_check import check_transform_chain

TEXT_LIMIT = 20_000
BATCH_LIMIT = 8


class Toolbox:
    """What one investigation may ask of one collected observation."""

    def __init__(
        self,
        case: Case,
        adapter: FileBundleAdapter,
        bundle: ObservationBundle,
        vision: VisionGateway | None = None,
    ) -> None:
        self.case = case
        self.adapter = adapter
        self.bundle = bundle
        self.vision = vision

    def list_evidence(self) -> list[dict[str, Any]]:
        """The catalogue: what exists, when it was taken, on which clock."""

        return [
            {
                "evidence_id": item.evidence_id,
                "reference": item.reference,
                "media_type": item.media_type,
                "captured_at": item.captured_at.isoformat(),
                "clock_domain": item.clock_domain,
                "summary": item.summary,
            }
            for item in self.case.evidence
        ]

    def read_evidence(self, evidence_ids: list[str]) -> list[dict[str, Any]]:
        """Deliver artifacts.  Images come back as observations, not pixels."""

        if len(evidence_ids) > BATCH_LIMIT:
            raise ValueError(f"一次最多读取 {BATCH_LIMIT} 件证据")
        delivered: list[dict[str, Any]] = []
        for evidence_id in evidence_ids:
            item = self._evidence(evidence_id)
            payload = self.adapter.read_artifact(item.reference)
            self.case.examined.add(evidence_id)
            delivered.append(
                {
                    "evidence_id": evidence_id,
                    "reference": item.reference,
                    "media_type": item.media_type,
                    **self._content(item, payload),
                }
            )
        return delivered

    def check_transform_chain(
        self,
        *,
        detection_evidence_id: str,
        calibration_evidence_id: str,
        tool_evidence_id: str,
        command_evidence_id: str,
        motion_evidence_id: str | None = None,
    ) -> dict[str, Any]:
        """Recompute the declared chain from evidence the caller names.

        Poses are read by their conventional hand-eye names -- ``camera_to_base``,
        ``pick_offset_from_part``, ``tool0_to_tcp``, ``detected_part_in_camera``,
        ``commanded_flange_base``, ``actual_flange_base``.  The result is a
        measurement; it names no segment and no cause.
        """

        detection = self._pose(detection_evidence_id, "detected_part_in_camera")
        calibration_source = self._parsed(calibration_evidence_id)
        tool = self._pose(tool_evidence_id, "tool0_to_tcp")
        command = self._pose(command_evidence_id, "commanded_flange_base")
        measured = (
            self._pose(motion_evidence_id, "actual_flange_base", optional=True)
            if motion_evidence_id
            else None
        )
        result = check_transform_chain(
            detection=detection,
            camera_to_base=calibration_source["camera_to_base"],
            pick_offset=calibration_source["pick_offset_from_part"],
            tool0_to_tcp=tool,
            commanded_flange=command,
            measured_flange=measured,
        )
        derived = result.as_dict()
        sources = [
            detection_evidence_id,
            calibration_evidence_id,
            tool_evidence_id,
            command_evidence_id,
            *([motion_evidence_id] if motion_evidence_id else []),
        ]
        evidence = self.case.add_evidence(
            Evidence(
                evidence_id=self.case.next_evidence_id(),
                bundle_id=self.bundle.run_id,
                reference="derived/transform-chain",
                media_type="application/json",
                captured_at=self.bundle.created_at,
                clock_domain=self.bundle.clock.source,
                summary="变换链核算，来自 " + "、".join(sources),
            )
        )
        return {"evidence_id": evidence.evidence_id, "sources": sources, **derived}

    def _evidence(self, evidence_id: str) -> Evidence:
        for item in self.case.evidence:
            if item.evidence_id == evidence_id:
                return item
        raise KeyError(f"{evidence_id} is not in this case")

    def _parsed(self, evidence_id: str) -> dict[str, Any]:
        item = self._evidence(evidence_id)
        payload = self.adapter.read_artifact(item.reference)
        self.case.examined.add(evidence_id)
        text = payload.decode("utf-8")
        if item.reference.endswith((".yaml", ".yml")):
            return yaml.safe_load(text)
        return json.loads(text)

    def _pose(self, evidence_id: str, key: str, *, optional: bool = False) -> dict[str, Any] | None:
        value = self._parsed(evidence_id).get(key)
        if value is None and optional:
            return None
        if value is None:
            raise KeyError(f"{evidence_id} carries no {key}")
        return value

    def _content(self, item: Evidence, payload: bytes) -> dict[str, Any]:
        if item.media_type.startswith("image/"):
            return self._observe(item, payload)
        if item.reference.endswith(".npy"):
            return self._depth_summary(payload, item.reference)
        text = payload.decode("utf-8", errors="replace")
        truncated = len(text) > TEXT_LIMIT
        return {"text": text[:TEXT_LIMIT], "truncated": truncated}

    def _observe(self, item: Evidence, payload: bytes) -> dict[str, Any]:
        if self.vision is None:
            return {"unavailable": "没有配置视觉模型，这张图无法被观察"}
        del payload
        assessment = self.vision.assess(
            self.adapter.root / item.reference,
            attachment_id=item.evidence_id,
            visible_name=item.reference,
            user_context="这是一次抓取节拍的现场画面，只描述看得见的内容。",
        )
        return {"observation": assessment}

    @staticmethod
    def _depth_summary(payload: bytes, reference: str) -> dict[str, Any]:
        import io

        depth = np.load(io.BytesIO(payload))
        finite = depth[np.isfinite(depth) & (depth > 0)]
        return {
            "summary": {
                "reference": reference,
                "shape": list(depth.shape),
                "valid_ratio": round(float(finite.size) / float(depth.size), 6),
                "min_m": round(float(finite.min()), 4) if finite.size else None,
                "max_m": round(float(finite.max()), 4) if finite.size else None,
                "median_m": round(float(np.median(finite)), 4) if finite.size else None,
            }
        }


def artifact_path(adapter: FileBundleAdapter, reference: str) -> Path:
    return adapter.root / reference
