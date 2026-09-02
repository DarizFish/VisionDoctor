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
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from visiondoctor.case import Case, Evidence, RepairPlan, Segment, diagnosis_gate
from visiondoctor.environment import FileBundleAdapter, ObservationBundle
from visiondoctor.multimodal import VisionGateway
from visiondoctor.repair import replay

from .transform_check import check_transform_chain

TEXT_LIMIT = 20_000
BATCH_LIMIT = 8


def _now() -> datetime:
    return datetime.now(UTC)


class Toolbox:
    """What one investigation may ask of one collected observation."""

    def __init__(
        self,
        case: Case,
        adapter: FileBundleAdapter | None,
        bundle: ObservationBundle | None,
        vision: VisionGateway | None = None,
        sandbox_root: Path | None = None,
        uploads: dict[str, Path] | None = None,
    ) -> None:
        self.case = case
        self.adapter = adapter
        self.bundle = bundle
        self.vision = vision
        #: Material a person handed in directly, by evidence id.
        self.uploads = uploads or {}
        self.sandbox_root = sandbox_root or Path(".runtime/vd-sandbox")

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
            payload = self._payload(item)
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
                bundle_id=self.bundle.run_id if self.bundle else "-",
                reference="derived/transform-chain",
                media_type="application/json",
                captured_at=_now(),
                clock_domain="host",
                summary="变换链核算，来自 " + "、".join(sources),
            )
        )
        return {"evidence_id": evidence.evidence_id, "sources": sources, **derived}

    def _evidence(self, evidence_id: str) -> Evidence:
        for item in self.case.evidence:
            if item.evidence_id == evidence_id:
                return item
        raise KeyError(f"{evidence_id} is not in this case")

    def _payload(self, item: Evidence) -> bytes:
        """Bundle artifacts come through the adapter; handed-in files from disk."""

        if item.evidence_id in self.uploads:
            return self.uploads[item.evidence_id].read_bytes()
        if self.adapter is None:
            raise ValueError(f"{item.evidence_id} 没有可读取的来源")
        return self.adapter.read_artifact(item.reference)

    def _local_path(self, item: Evidence) -> Path:
        if item.evidence_id in self.uploads:
            return self.uploads[item.evidence_id]
        return self.adapter.root / item.reference

    def _parsed(self, evidence_id: str) -> dict[str, Any]:
        item = self._evidence(evidence_id)
        payload = self._payload(item)
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
            self._local_path(item),
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

    # ---- Behind the diagnosis gate -------------------------------------------------

    def list_source(self) -> dict[str, Any]:
        """File names at the bound revision.  Available once the gate is passed."""

        binding = self._unlocked_project()
        listing = self._git(binding, "ls-tree", "-r", "--name-only", binding.revision)
        return {"revision": binding.revision, "paths": listing.split()}

    def read_source(self, path: str) -> dict[str, Any]:
        """One file as it stood at the bound revision, not as it stands now."""

        binding = self._unlocked_project()
        text = self._git(binding, "show", f"{binding.revision}:{path}")
        return {"revision": binding.revision, "path": path, "text": text[:TEXT_LIMIT]}

    def propose_repair(
        self,
        *,
        target_segment: str,
        hypothesis_id: str,
        path: str,
        new_text: str,
        rationale: str,
    ) -> dict[str, Any]:
        """Freeze a candidate and run it, in isolation, on this run's own inputs.

        The bound repository is not touched.  A passing replay says the code now
        computes what the chain intended -- never that the cell has recovered.
        """

        binding = self._unlocked_project()
        inputs = {
            artifact.path.replace("/", "-"): self.adapter.read_artifact(artifact.path)
            for artifact in (self.bundle.artifacts if self.bundle else ())
            if artifact.path.endswith("input.json")
        }
        try:
            outcome = replay(
                binding=binding,
                candidate_id=f"PLAN-{len(self.case.repair_plans) + 1}",
                sandbox_root=self.sandbox_root,
                inputs=inputs,
                edits={path: new_text},
            )
        except Exception as exc:  # noqa: BLE001 - the model needs the reason back
            return {"error": f"{type(exc).__name__}: {exc}"}
        plan = RepairPlan(
            plan_id=f"PLAN-{len(self.case.repair_plans) + 1}",
            case_id=self.case.case_id,
            target_segment=Segment(target_segment),
            hypothesis_id=hypothesis_id,
            project_revision=binding.revision,
            diff=outcome.diff,
        )
        self.case.repair_plans.append(plan)
        evidence = self.case.add_evidence(
            Evidence(
                evidence_id=self.case.next_evidence_id(),
                bundle_id=self.bundle.run_id,
                reference=f"derived/replay/{plan.plan_id}",
                media_type="application/json",
                captured_at=_now(),
                clock_domain="host",
                summary=f"{plan.plan_id} 的隔离复现，理由：{rationale}",
            )
        )
        return {
            "plan_id": plan.plan_id,
            "frozen_hash": plan.frozen_hash,
            "evidence_id": evidence.evidence_id,
            **outcome.as_dict(),
        }

    def _unlocked_project(self):
        if self.case.project is None:
            raise ValueError("这个案件还没有绑定项目仓库")
        verdict = diagnosis_gate(self.case)
        if not verdict.passed:
            raise PermissionError("诊断门未通过，源码不可见：" + "；".join(verdict.reasons))
        return self.case.project

    @staticmethod
    def _git(binding, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(binding.repository), *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )
        if result.returncode != 0:
            raise ValueError(result.stderr.strip() or "git 命令失败")
        return result.stdout
