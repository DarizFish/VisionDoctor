"""The read-only surface an investigation may touch.

Every call is executed by the host, which records it and hands back what it
actually read.  A model that says it examined something without asking for it
has no evidence to cite: only what passed through here is admitted.

Two layers are kept apart.  The software layer -- what the running program read,
wrote, logged and does again on recorded inputs -- is always open and is where a
fault is located.  The source layer opens only beneath a located software fault,
at the version that ran; it explains the mechanism and carries the patch.  Real
equipment is never controlled here.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from dataclasses import replace as _replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from visiondoctor.case import (
    LAYER_ORDER,
    Case,
    Evidence,
    RepairPlan,
    repair_gate,
    software_localizations,
    source_layer_gate,
)
from visiondoctor.case.graph import inspect_target
from visiondoctor.case.isolation import REFERENCE as STRUCTURAL_REFERENCE
from visiondoctor.case.isolation import structural_results
from visiondoctor.case.templates import TEMPLATES
from visiondoctor.environment import FileBundleAdapter, ObservationBundle
from visiondoctor.knowledge import read_knowledge
from visiondoctor.multimodal import VisionGateway
from visiondoctor.repair import replay

from .measurements import check_alignment, depth_statistics, measure_region
from .residuals import evaluate
from .run_compare import differences, parse
from .structure import ASSUMPTIONS, Model, gaps, isolate, suggestions
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
        observation_dirs: tuple[Path, ...] = (),
    ) -> None:
        self.case = case
        self.adapter = adapter
        self.bundle = bundle
        self.vision = vision
        #: Material a person handed in directly, by evidence id.
        self.uploads = uploads or {}
        self.sandbox_root = sandbox_root or Path(".runtime/vd-sandbox")
        self.observation_adapters = {}
        for directory in observation_dirs:
            source = FileBundleAdapter(directory)
            self.observation_adapters[source.collect().run_id] = source
        if adapter is not None and bundle is not None:
            self.observation_adapters.setdefault(bundle.run_id, adapter)

    @staticmethod
    def inspect_grasp_graph(target_id: str) -> dict[str, Any]:
        return inspect_target(target_id)

    @staticmethod
    def read_domain_knowledge(knowledge_ids: list[str]) -> dict[str, Any]:
        return read_knowledge(knowledge_ids)

    def list_evidence(self) -> list[dict[str, Any]]:
        """The catalogue: what exists, when it was taken, on which clock."""

        return [
            {
                "evidence_id": item.evidence_id,
                "bundle_id": item.bundle_id,
                "reference": item.reference,
                "media_type": item.media_type,
                "captured_at": item.captured_at.isoformat(),
                "clock_domain": item.clock_domain,
                "phase": item.phase,
                "layer": item.layer,
                "summary": item.summary,
            }
            for item in self.case.evidence
        ]

    def read_evidence(
        self, evidence_ids: list[str], question: str = "",
    ) -> list[dict[str, Any]]:
        """Deliver artifacts.  Images come back as observations, not pixels."""

        if len(evidence_ids) > BATCH_LIMIT:
            raise ValueError(f"一次最多读取 {BATCH_LIMIT} 件证据")
        delivered: list[dict[str, Any]] = []
        for evidence_id in evidence_ids:
            item = self._evidence(evidence_id)
            payload = self._payload(item)
            content = self._content(item, payload, question)
            if not content.get("unavailable"):
                self.case.examined.add(evidence_id)
            delivered.append(
                {
                    "evidence_id": evidence_id,
                    "bundle_id": item.bundle_id,
                    "reference": item.reference,
                    "media_type": item.media_type,
                    "phase": item.phase,
                    "layer": item.layer,
                    **content,
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
        return self._derived("transform-chain", sources, derived)

    def measure_rgbd_region(
        self, *, rgb_evidence_id: str, depth_evidence_id: str,
        camera_evidence_id: str, roi_xyxy: list[int],
        detection_evidence_id: str | None = None,
    ) -> dict[str, Any]:
        sources = [rgb_evidence_id, depth_evidence_id, camera_evidence_id]
        depth_item = self._evidence(depth_evidence_id)
        depth_payload = self._payload(depth_item)
        if not depth_payload.startswith(b"\x93NUMPY"):
            raise ValueError(
                f"{depth_evidence_id} ({depth_item.reference}) is not a NumPy depth array. "
                "Choose a depth .npy artifact from the evidence catalogue. If none was "
                "supplied, depth cannot be measured; this is not evidence of sensor failure "
                "or corrupt depth data."
            )
        rgb = np.asarray(Image.open(io.BytesIO(self._payload(self._evidence(rgb_evidence_id))))
                         .convert("RGB"))
        depth = np.load(io.BytesIO(depth_payload), allow_pickle=False)
        camera = self._parsed(camera_evidence_id)
        detection = self._parsed(detection_evidence_id) if detection_evidence_id else None
        if detection_evidence_id:
            sources.append(detection_evidence_id)
        result = measure_region(rgb, depth, camera, roi_xyxy, detection)
        self.case.examined.update(sources)
        result["correspondence_scope"] = (
            "Pixel/geometry statistics only. Use capture records to check identities and timing."
        )
        return self._derived("rgbd-region", sources, result)

    def check_capture_alignment(
        self, *, capture_evidence_id: str, detection_evidence_id: str,
        command_evidence_id: str | None = None,
    ) -> dict[str, Any]:
        sources = [capture_evidence_id, detection_evidence_id]
        capture = self._parsed(capture_evidence_id)
        detection = self._parsed(detection_evidence_id)
        command = self._parsed(command_evidence_id) if command_evidence_id else None
        if command_evidence_id:
            sources.append(command_evidence_id)
        return self._derived(
            "capture-alignment", sources, check_alignment(capture, detection, command)
        )

    # ---- Structural diagnosis: what the bound records can detect and tell apart ------

    def structural_diagnose(
        self, *, template_ids: list[str], part_id: str, bindings: dict[str, str],
    ) -> dict[str, Any]:
        """Structure first, then the admitted tests, then isolation -- all computed by the host.

        The caller maps this project's records onto a template's measurements;
        which faults the records can detect or tell apart, which tests may run
        and what they are sensitive to all follow from the template's structure.
        Bindings from an earlier call on the same run and part are carried forward.
        """

        unknown = [name for name in template_ids if name not in TEMPLATES]
        if unknown:
            raise KeyError(f"没有模板 {'、'.join(unknown)}；可用：{'、'.join(TEMPLATES)}")
        references = {
            item.id for template in TEMPLATES.values() for item in template.measurements
            if item.reference
        }
        runs = {self._evidence(evidence_id).bundle_id
                for name, evidence_id in bindings.items() if name not in references}
        if len(runs) > 1:
            raise ValueError("除参考测量外，绑定的证据必须来自同一次运行")
        previous_run = next((
            item.content["run_id"] for item in structural_results(self.case)
            if item.content["part_id"] == part_id and not runs
        ), None)
        if not runs and previous_run is None:
            raise ValueError("至少绑定一项本次运行的测量")
        run_id = runs.pop() if runs else previous_run
        for evidence_id in bindings.values():
            reference = self._evidence(evidence_id).reference
            if reference.startswith("parts/") and not reference.startswith(f"parts/{part_id}/"):
                raise ValueError(f"{evidence_id}（{reference}）不属于工件 {part_id}")
        previous = next((
            item.content for item in structural_results(self.case)
            if item.content["run_id"] == run_id and item.content["part_id"] == part_id
        ), None)
        chosen = list(dict.fromkeys([*(previous or {}).get("templates", []), *template_ids]))
        bound = {**(previous or {}).get("bindings", {}), **bindings}
        templates = [TEMPLATES[name] for name in chosen]
        offered = {item.id for template in templates for item in template.measurements}
        stray = sorted(set(bound) - offered)
        if stray:
            raise ValueError(
                f"所选模板没有测量 {'、'.join(stray)}；可用：{'、'.join(sorted(offered))}"
            )

        documents = {name: self._record(evidence_id) for name, evidence_id in bound.items()}
        # A record that is bound but does not carry a variable does not measure it.
        unusable: dict[str, list[str]] = {}
        missing: set[str] = set()
        for template in templates:
            for measurement in template.measurements:
                document = documents.get(measurement.id)
                if document is None:
                    continue
                for variable, fields in measurement.provides.items():
                    absent = [key for key in fields
                              if not isinstance(document, dict) or document.get(key) is None]
                    if not fields and not isinstance(document, np.ndarray):
                        absent = ["深度数组"]
                    if absent:
                        missing.add(variable)
                        unusable.setdefault(measurement.id, []).append(
                            f"{variable}（缺 {'、'.join(absent)}）"
                        )
        model = Model(templates, bound, missing)
        tests, outcomes, sensitivity = [], {}, {}
        for test in model.tests:
            row: dict[str, Any] = {
                "test": test.id, "name": test.name, "equations": list(test.equations),
                "measurements": list(test.measurements),
                "sensitive_to": sorted(model.sensitivity(test)),
                "threshold": test.threshold, "threshold_source": test.threshold_source,
            }
            refused = model.admissible(test)
            if refused is None:
                try:
                    row.update(evaluate(test.residual, test.threshold, documents))
                except (KeyError, ValueError, TypeError) as exc:
                    refused = f"记录不足以计算：{exc}"
            if refused is not None:
                row.update(status="unevaluated", reason=refused)
            else:
                outcomes[test.id] = row["status"]
                sensitivity[test.id] = model.sensitivity(test)
            tests.append(row)
        isolation = isolate(model, outcomes, sensitivity)
        result = {
            "run_id": run_id,
            "part_id": part_id,
            "templates": chosen,
            "bindings": bound,
            "unusable_bindings": unusable,
            "assumptions": list(ASSUMPTIONS),
            "structure": model.summary(),
            "tests": tests,
            "isolation": isolation,
            "gaps": gaps(model, isolation["candidates"], sensitivity),
            "suggestions": suggestions(model, isolation["candidates"]),
            "scope": ("核算只覆盖所选模板里的节点与方程；未检验不等于正常。"
                      "模板外的原因（如遮挡、成像质量）需要其他证据。"),
        }
        return self._derived(
            STRUCTURAL_REFERENCE.removeprefix("derived/"), list(bound.values()), result
        )

    # ---- Software layer: what ran, compared and re-run ------------------------------

    def compare_runs(self, *, baseline_run_id: str, run_id: str) -> dict[str, Any]:
        """Version and software-layer records of two runs, compared by content."""

        runs = {bundle.run_id: bundle for bundle in self.case.observations}
        for name in (baseline_run_id, run_id):
            if name not in runs:
                raise KeyError(f"本案没有观察 {name}；可用：{'、'.join(runs)}")
        held = {
            name: {
                item.reference: item for item in self.case.evidence
                if item.bundle_id == name and item.layer == "software"
            }
            for name in (baseline_run_id, run_id)
        }
        baseline, candidate = held[baseline_run_id], held[run_id]
        records, sources = [], []
        for reference in sorted(set(baseline) & set(candidate)):
            left, right = baseline[reference], candidate[reference]
            parsed = [parse(reference, self._payload(item)) for item in (left, right)]
            sources += [left.evidence_id, right.evidence_id]
            row: dict[str, Any] = {
                "reference": reference,
                "byte_identical": left.sha256 == right.sha256,
            }
            if parsed[0] is None or parsed[1] is None:
                row["compared"] = "not a single structured document; read it directly"
            else:
                row.update(differences(parsed[0], parsed[1]))
            records.append(row)
        self.case.examined.update(sources)
        revisions = [
            str(runs[name].project_revision.get("commit") or "")
            for name in (baseline_run_id, run_id)
        ]
        result = {
            "baseline_run_id": baseline_run_id,
            "run_id": run_id,
            "revision": {"baseline": revisions[0], "run": revisions[1],
                         "same": revisions[0] == revisions[1] != ""},
            "records": records,
            "only_in_baseline": sorted(set(baseline) - set(candidate)),
            "only_in_run": sorted(set(candidate) - set(baseline)),
            "scope": ("Software-layer records only, compared by parsed content. Identity and "
                      "time fields are omitted. A difference locates a change; it does not "
                      "say which side is correct or why it changed."),
        }
        return self._derived("run-comparison", sources, result)

    def replay_running_version(self) -> dict[str, Any]:
        """Re-run the version that ran on its recorded inputs; no source is read."""

        binding = self.case.project
        if binding is None or not binding.runnable:
            raise ValueError("本案没有可复跑的程序，无法在记录版本上复跑")
        inputs, sources, outputs = self._recorded_inputs()
        outcome = replay(
            binding=_replace(binding, test_command=None),
            candidate_id="RUNNING-VERSION",
            sandbox_root=self.sandbox_root,
            inputs=inputs,
        )
        rows = []
        for item in outcome.replays:
            recorded = outputs.get(item["input"])
            row: dict[str, Any] = {"input": item["input"], "exit_code": item["run"]["exit_code"]}
            if recorded is not None and item["output"] is not None:
                sources.append(recorded.evidence_id)
                compared = differences(self._parsed(recorded.evidence_id), item["output"])
                row.update(reproduced=item["run"]["succeeded"] and compared["same"], **compared)
            else:
                row.update(reproduced=False, output_tail=item["run"]["output_tail"][-400:])
            rows.append(row)
        result = {
            "scope": ("The recorded version re-run on recorded inputs. Reproducing the recorded "
                      "output shows the result follows from this version and these inputs; it "
                      "does not say where in the implementation the cause lies."),
            "revision": outcome.base_revision,
            "replays": rows,
        }
        return self._derived("replay-running-version", sources, result, layer="software")

    def _recorded_inputs(self) -> tuple[dict[str, bytes], list[str], dict[str, Evidence]]:
        """Program inputs of the observation under diagnosis, and their recorded outputs."""

        if self.bundle is None or self.adapter is None:
            raise ValueError("没有可复跑的观察输入")
        inputs: dict[str, bytes] = {}
        sources: list[str] = []
        outputs: dict[str, Evidence] = {}
        held = {
            item.reference: item for item in self.case.evidence
            if item.bundle_id == self.bundle.run_id
        }
        for artifact in self.bundle.artifacts:
            if not artifact.path.endswith("input.json"):
                continue
            name = artifact.path.replace("/", "-")
            inputs[name] = self.adapter.read_artifact(artifact.path)
            if artifact.path in held:
                sources.append(held[artifact.path].evidence_id)
            recorded = held.get(artifact.path[: -len("input.json")] + "output.json")
            if recorded is not None:
                outputs[name] = recorded
        self.case.examined.update(sources)
        return inputs, sources, outputs

    def _derived(
        self, name: str, sources: list[str], result: dict[str, Any], *, layer: str | None = None,
    ) -> dict[str, Any]:
        content = {"sources": sources, **result}
        payload = json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")
        source_bundles = {self._evidence(source).bundle_id for source in sources}
        bundle_id = next(iter(source_bundles)) if len(source_bundles) == 1 else (
            f"case:{self.case.case_id}" if source_bundles else (
                self.bundle.run_id if self.bundle else "-"
            )
        )
        evidence = self.case.add_evidence(
            Evidence(
                evidence_id=self.case.next_evidence_id(),
                bundle_id=bundle_id,
                reference=f"derived/{name}",
                media_type="application/json",
                captured_at=_now(),
                clock_domain="host",
                summary=f"{name}，来自 " + "、".join(sources),
                sha256=hashlib.sha256(payload).hexdigest(),
                content=content,
                layer=layer or max(
                    (self.case.layer_of(source) for source in sources),
                    key=LAYER_ORDER.__getitem__, default="observation",
                ),
            )
        )
        return {"evidence_id": evidence.evidence_id, "layer": evidence.layer, **content}

    def _evidence(self, evidence_id: str) -> Evidence:
        for item in self.case.evidence:
            if item.evidence_id == evidence_id:
                return item
        raise KeyError(f"{evidence_id} is not in this case")

    def _payload(self, item: Evidence) -> bytes:
        """Bundle artifacts come through the adapter; handed-in files from disk."""

        if item.content is not None:
            payload = json.dumps(item.content, ensure_ascii=False, sort_keys=True).encode("utf-8")
            if hashlib.sha256(payload).hexdigest() != item.sha256:
                raise ValueError("computed evidence content no longer matches its hash")
            return payload
        if item.evidence_id in self.uploads:
            return self.uploads[item.evidence_id].read_bytes()
        if item.reference.startswith("source/git/"):
            _, _, revision, path = item.reference.split("/", maxsplit=3)
            payload = self._git(self._bound_project(), "show", f"{revision}:{path}").encode("utf-8")
            if hashlib.sha256(payload).hexdigest() != item.sha256:
                raise ValueError("source evidence hash no longer matches its pinned content")
            return payload
        if item.reference.startswith("results/"):
            return self._task_result(item.bundle_id, item.reference.split("/", maxsplit=1)[1])
        payload = self._observation_adapter(item).read_artifact(item.reference)
        if item.sha256 and hashlib.sha256(payload).hexdigest() != item.sha256:
            raise ValueError("observation artifact no longer matches its admitted evidence hash")
        return payload

    def _observation_adapter(self, item: Evidence) -> FileBundleAdapter:
        source = self.observation_adapters.get(item.bundle_id)
        if source is None:
            raise ValueError(f"missing observation source for {item.bundle_id}: {item.evidence_id}")
        return source

    def _task_result(self, bundle_id: str, part_id: str) -> bytes:
        """A task result is evidence with no file behind it: serve the manifest's own."""

        for bundle in self.case.observations:
            if bundle.run_id == bundle_id:
                for result in bundle.results:
                    if result.part_id == part_id:
                        return json.dumps(
                            result.model_dump(mode="json"), ensure_ascii=False
                        ).encode("utf-8")
        raise KeyError(f"观察 {bundle_id} 没有 {part_id} 的任务结果")

    def _local_path(self, item: Evidence) -> Path:
        if item.evidence_id in self.uploads:
            return self.uploads[item.evidence_id]
        return self._observation_adapter(item).root / item.reference

    def _parsed(self, evidence_id: str) -> dict[str, Any]:
        item = self._evidence(evidence_id)
        payload = self._payload(item)
        self.case.examined.add(evidence_id)
        text = payload.decode("utf-8")
        if item.reference.endswith((".yaml", ".yml")):
            return yaml.safe_load(text)
        return json.loads(text)

    def _record(self, evidence_id: str) -> Any:
        """A structured record, or a NumPy array for a ``.npy`` artifact."""

        item = self._evidence(evidence_id)
        if not item.reference.endswith(".npy"):
            return self._parsed(evidence_id)
        array = np.load(io.BytesIO(self._payload(item)), allow_pickle=False)
        self.case.examined.add(evidence_id)
        return array

    def _pose(self, evidence_id: str, key: str, *, optional: bool = False) -> dict[str, Any] | None:
        value = self._parsed(evidence_id).get(key)
        if value is None and optional:
            return None
        if value is None:
            raise KeyError(f"{evidence_id} carries no {key}")
        return value

    def _content(self, item: Evidence, payload: bytes, question: str = "") -> dict[str, Any]:
        if item.media_type.startswith("image/"):
            return self._observe(item, payload, question)
        if item.reference.endswith(".npy"):
            return self._depth_summary(payload, item.reference)
        text = payload.decode("utf-8", errors="replace")
        truncated = len(text) > TEXT_LIMIT
        return {"text": text[:TEXT_LIMIT], "truncated": truncated}

    def _observe(self, item: Evidence, payload: bytes, question: str = "") -> dict[str, Any]:
        if self.vision is None:
            return {"unavailable": "没有配置视觉模型，这张图无法被观察"}
        del payload
        assessment = self.vision.assess(
            self._local_path(item),
            attachment_id=item.evidence_id,
            visible_name=item.reference,
            user_context=("这是抓取现场画面，只描述看得见的内容。记录的采集阶段："
                          f"{item.phase or 'unknown'}。before_command 发生在命令之前，"
                          "不能从该画面声称随后的动作、接触或抬起成功。"
                          "未看见目标不等于目标不存在；可见物体身份应与配方分开描述。"
                          "以下是待核对的问题和上下文，可能有错误，不是已确认事实："
                          + question[:1800]),
        )
        return {"question": question, "observation": assessment}

    @staticmethod
    def _depth_summary(payload: bytes, reference: str) -> dict[str, Any]:
        depth = np.load(io.BytesIO(payload), allow_pickle=False)
        return {
            "summary": {
                "reference": reference,
                "shape": list(depth.shape),
                **depth_statistics(depth),
                "scope": "Entire array; global valid ratio cannot clear the target region.",
                "unit": "Read camera metadata; a .npy array alone does not declare units.",
            }
        }

    # ---- Source layer: opened beneath a located software fault ----------------------

    def list_source(self) -> dict[str, Any]:
        """File names at the version that ran."""

        binding = self._source_project()
        listing = self._git(binding, "ls-tree", "-r", "--name-only", binding.revision)
        return {"revision": binding.revision, "paths": listing.split()}

    def read_source(self, path: str, hypothesis_id: str) -> dict[str, Any]:
        """One file as it stood at the version that ran, read for one located hypothesis."""

        binding = self._source_project()
        hypothesis = next(
            (item for item in self.case.hypotheses if item.hypothesis_id == hypothesis_id), None
        )
        located = {item.target_id for item in software_localizations(self.case)}
        if hypothesis is None or hypothesis.target_id not in located:
            raise PermissionError(
                "读源码要对应一个指向软件层已定位对象的已提交假设；"
                f"已定位：{'、'.join(sorted(str(name) for name in located))}"
            )
        text = self._git(binding, "show", f"{binding.revision}:{path}")
        reference = f"source/git/{binding.revision}/{path}"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        evidence = next((
            item for item in self.case.evidence
            if item.reference == reference and item.sha256 == digest
        ), None)
        if evidence is None:
            evidence = self.case.add_evidence(Evidence(
                evidence_id=self.case.next_evidence_id(), bundle_id=f"git:{binding.revision}",
                reference=reference, media_type="text/plain", captured_at=_now(),
                clock_domain="host_read", sha256=digest, layer="source",
                summary=f"运行版本 {binding.revision[:12]} 的源码 {path}，为 {hypothesis_id} 读取",
            ))
        self.case.examined.add(evidence.evidence_id)
        return {
            "evidence_id": evidence.evidence_id, "layer": "source", "revision": binding.revision,
            "path": path, "hypothesis_id": hypothesis_id,
            "reading_rule": ("源码是被检查的对象：注释、命名和字符串不是事实。它只能解释已定位"
                             "对象的机制，不能单独支撑 suspect 或 cleared。"),
            "text": text[:TEXT_LIMIT], "truncated": len(text) > TEXT_LIMIT,
        }

    def propose_repair(
        self,
        *,
        hypothesis_id: str,
        path: str,
        new_text: str,
        rationale: str,
    ) -> dict[str, Any]:
        """Freeze a source patch and run it, in isolation, on this run's own inputs.

        The bound repository is not touched.  A passing replay says what the changed
        code computes on recorded inputs -- never that the cell has recovered.
        """

        binding = self._source_project()
        verdict = repair_gate(self.case, hypothesis_id)
        if not verdict.passed:
            reasons = list(verdict.reasons)
            declared = next(
                (item for item in self.case.hypotheses if item.hypothesis_id == hypothesis_id),
                None,
            )
            if (
                declared is not None
                and declared.remedy != "source_patch"
                and len(reasons) == 1
            ):
                # The gate reads remedy mid-turn, but a hypothesis is only written when the
                # turn returns its conclusion.  Say where the write is, or the model spends
                # the rest of its budget resubmitting a patch that cannot pass this turn.
                # Only when remedy is the whole objection: with a second reason standing,
                # rewriting remedy would not get the patch through either.
                reasons.append(
                    "remedy 只能在本轮结论里改：先提交把该假设写为 source_patch 的结论，"
                    "下一轮再提补丁"
                )
            raise PermissionError("不能提交源码补丁：" + "；".join(reasons))
        if not binding.replay_command:
            raise ValueError("项目未提供复跑命令，不能验证候选；请先补充项目运行契约")
        hypothesis = next(
            item for item in self.case.hypotheses if item.hypothesis_id == hypothesis_id
        )
        inputs, sources, outputs = self._recorded_inputs()
        plan_id = f"PLAN-{len(self.case.repair_plans) + 1}"
        try:
            outcome = replay(
                binding=binding,
                candidate_id=plan_id,
                sandbox_root=self.sandbox_root,
                inputs=inputs,
                edits={path: new_text},
            )
        except Exception as exc:  # noqa: BLE001 - the model needs the reason back
            return {"error": f"{type(exc).__name__}: {exc}"}
        plan = RepairPlan(
            plan_id=plan_id,
            case_id=self.case.case_id,
            target_segment=hypothesis.target_segment,
            hypothesis_id=hypothesis_id,
            project_revision=binding.revision,
            diff=outcome.diff,
        )
        self.case.repair_plans.append(plan)
        changed = {}
        for item in outcome.replays:
            recorded = outputs.get(item["input"])
            if recorded is not None and item["output"] is not None:
                changed[item["input"]] = differences(
                    self._parsed(recorded.evidence_id), item["output"]
                )
        result = {
            "plan_id": plan.plan_id,
            "frozen_hash": plan.frozen_hash,
            "rationale": rationale,
            **outcome.as_dict(),
            "changed_from_recorded_output": changed,
        }
        return self._derived(f"replay/{plan.plan_id}", sources, result, layer="source")

    def _source_project(self):
        verdict = source_layer_gate(self.case)
        if not verdict.passed:
            raise PermissionError("源码层未开放：" + "；".join(verdict.reasons))
        return self.case.project

    def _bound_project(self):
        if self.case.project is None:
            raise ValueError("这个案件还没有绑定项目仓库")
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
