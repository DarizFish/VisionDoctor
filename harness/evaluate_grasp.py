"""Run a real, read-only diagnosis on a recorded grasp observation.

This captures behavior and failures; it does not grade its own root cause. Fault
labels and subsequent review belong outside the model's inputs.

``--access`` sets what the case can reach: ``observation`` (records only),
``runnable`` (the program can be re-run but its source is never shown, as when a
site holds only a release) or ``source`` (source opens beneath a located fault).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from visiondoctor.case import CaseService, software_localizations, source_layer_gate
from visiondoctor.case.service import _call_summary, _model_gateway, _vision_gateway
from visiondoctor.investigation import investigate

REPAIR_PROMPT = (
    "根据上一轮的软件层定位继续。若源码层已开放且证据支持实现缺陷，读取与已定位对象相关的"
    "源码解释机制，把对应假设的 remedy 写为 source_patch 后提交结论；若本轮已有这样的已提交假设，"
    "提出最小补丁并调用 propose_repair 隔离复跑，核对复跑输出相对记录输出的变化。"
    "若源码层未开放、证据不支持软件实现缺陷或应外部处理，保留软件层结论与交接出口。"
    "不要通过改变观测、评分或放宽容差伪装修复。本轮不批准应用，不声称现场恢复。"
)


def run(
    bundle: Path, repository: Path | None, prompt: str, output: Path, vision: bool,
    attempt_repair: bool = False, access: str = "source",
    references: tuple[Path, ...] = (), repair_turns: int = 1,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    implementation = hashlib.sha256()
    for path in sorted((project_root / "src/visiondoctor").rglob("*.py")):
        implementation.update(path.relative_to(project_root).as_posix().encode())
        implementation.update(path.read_bytes())
    service = CaseService(output / "cases")
    case_id = service.create(prompt[:40])
    service.attach_observation(case_id, bundle.resolve())
    for reference in references:
        service.attach_observation(case_id, reference.resolve())
    if repository is not None and access != "observation":
        service.bind_project(
            case_id, repository.resolve(),
            replay_command=(sys.executable, "-m", "pick_demo.replay", "--input", "{input}",
                            "--output", "{output}", "--log", "{log}"),
            test_command=(sys.executable, "-m", "pytest", "-q") if attempt_repair else None,
            source_readable=access == "source",
        )
    record = service.record(case_id)
    gateway = _model_gateway()
    started = time.monotonic()
    calls = []

    def observed(call) -> None:
        calls.append(_call_summary(call))
        print(json.dumps({
            "elapsed_s": round(time.monotonic() - started, 1),
            "tool": call.name, "delivered": call.delivered, "failure": call.failure,
        }, ensure_ascii=False), flush=True)

    prompts = [prompt] + ([REPAIR_PROMPT] * repair_turns if attempt_repair else [])
    gates = []
    error = None
    try:
        for phase_prompt in prompts:
            record.messages.append({"role": "user", "content": phase_prompt})
            gates.append(source_layer_gate(record.case).model_dump(mode="json"))
            turn = investigate(
                case=record.case, adapter=record.adapter, bundle=record.bundle,
                prompt=phase_prompt, gateway=gateway, vision=_vision_gateway() if vision else None,
                sandbox_root=output / "sandbox", observer=observed,
                observation_dirs=tuple(Path(path) for path in record.observation_dirs),
            )
            record.turns.append(turn)
            record.messages.append({
                "role": "assistant", "content": turn.next_step,
                "turn_id": turn.turn_id, "calls": [_call_summary(c) for c in turn.calls],
            })
            service.keep(case_id)
    except Exception as exc:  # noqa: BLE001 - preserve a failed real experiment
        error = f"{type(exc).__name__}: {exc}"
        record.messages.append({"role": "assistant", "content": error, "calls": calls})
    service.keep(case_id)
    view = service.view(case_id)
    (output / "case-view.json").write_text(
        json.dumps(view, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "implementation_sha256": implementation.hexdigest(),
        "prompt": prompt,
        "model": gateway.model,
        "bundle": str(bundle), "case_id": case_id,
        "access": access, "references": [str(item) for item in references],
        "source_layer_before_each_turn": gates,
        "source_layer_now": source_layer_gate(record.case).model_dump(mode="json"),
        "software_localizations": [
            item.model_dump(mode="json") for item in software_localizations(record.case)
        ],
        "source_calls": [
            call["name"] for call in calls
            if call["name"] in {"list_source", "read_source", "propose_repair"}
        ],
        "elapsed_s": round(time.monotonic() - started, 2),
        "error": error, "calls": calls,
        "knowledge_calls": sum(call["name"] == "read_domain_knowledge" for call in calls),
        "graph_calls": sum(call["name"] == "inspect_grasp_graph" for call in calls),
        "findings": [item.model_dump(mode="json") for item in record.case.findings],
        "hypotheses": view["hypotheses"],
        "next_step": record.messages[-1]["content"],
        "plans": view["plans"],
        "scope": "Real model behavior on a recorded bundle; accuracy requires external review.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "completed": error is None, "elapsed_s": summary["elapsed_s"],
        "output": str(output.resolve()),
    }, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repository", type=Path)
    parser.add_argument(
        "--prompt", default="工位抓取位置不对，分析原因和下一步。先诊断，不执行修复。",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument(
        "--repair", action="store_true", help="request follow-up turns beneath the software layer",
    )
    parser.add_argument("--repair-turns", type=int, default=1)
    parser.add_argument("--access", choices=("observation", "runnable", "source"),
                        default="source")
    parser.add_argument("--reference", type=Path, action="append", default=[],
                        help="another observation, such as a normal run, attached for comparison")
    args = parser.parse_args()
    result = run(
        args.bundle, args.repository, args.prompt, args.output, not args.no_vision, args.repair,
        args.access, tuple(args.reference), args.repair_turns,
    )
    if result["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
