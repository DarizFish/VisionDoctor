from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_agent_events(run_root: Path) -> list[dict[str, Any]]:
    """Return a safe, user-facing timeline from append-only execution audit files."""

    root = run_root.resolve()
    events: list[dict[str, Any]] = []
    for item in _read_jsonl(root / "trace.jsonl"):
        event = str(item.get("event", "workflow_event"))
        timestamp = str(item.get("timestamp", ""))
        if event == "state_transition":
            current = str(item.get("current", "UNKNOWN"))
            events.append(
                _event(
                    timestamp,
                    actor="Orchestrator",
                    kind="workflow",
                    title=f"进入 {current}",
                    detail=f"{item.get('previous', 'NEW')} → {current}",
                    status="running" if current not in _TERMINAL_STATES else "completed",
                    proof={"state": current},
                )
            )
        elif event == "execution_started":
            events.append(
                _event(
                    timestamp,
                    actor="Sandbox",
                    kind="execution",
                    title=f"隔离执行 {item.get('candidate_id', 'candidate')}",
                    detail="创建独立 Git worktree，候选代码仅进入无网络 Docker 沙箱。",
                    status="running",
                    proof={
                        "candidate_id": item.get("candidate_id"),
                        "base_commit": item.get("base_commit"),
                    },
                )
            )
        elif event == "execution_completed":
            succeeded = bool(item.get("tests_passed"))
            events.append(
                _event(
                    timestamp,
                    actor="Deterministic QA",
                    kind="validation",
                    title=f"完成 {item.get('candidate_id', 'candidate')} 执行",
                    detail=(
                        f"执行状态 {item.get('status')}; "
                        f"单元测试 {'通过' if succeeded else '失败'}。"
                    ),
                    status="completed" if succeeded else "failed",
                    proof={
                        "candidate_id": item.get("candidate_id"),
                        "execution_status": item.get("status"),
                        "tests_passed": succeeded,
                    },
                )
            )
        elif event == "candidate_rejected":
            events.append(
                _event(
                    timestamp,
                    actor="QA Agent",
                    kind="validation",
                    title=f"拒绝 {item.get('candidate_id', 'candidate')}",
                    detail="确定性门禁未通过，候选已回滚并把失败证据交给 Patch Agent。",
                    status="failed",
                    proof={
                        "candidate_id": item.get("candidate_id"),
                        "rollback_succeeded": item.get("rollback_succeeded"),
                    },
                )
            )
        elif event == "external_gate_attempt":
            passed = bool(item.get("passed"))
            retryable = bool(item.get("retryable"))
            category = str(item.get("failure_category") or "none")
            events.append(
                _event(
                    timestamp,
                    actor="Simulation Adapter",
                    kind="validation",
                    title=f"机器人仿真检查第 {item.get('attempt', '?')} 次",
                    detail=(
                        "仿真检查完成。"
                        if passed
                        else (
                            "检测到可重试的仿真或规划波动；候选代码不会因此被退回。"
                            if retryable
                            else "仿真检查未通过，已按明确故障类别停止。"
                        )
                    ),
                    status="completed" if passed else "failed",
                    proof={
                        "candidate_id": item.get("candidate_id"),
                        "attempt": item.get("attempt"),
                        "failure_category": category,
                        "retryable": retryable,
                    },
                )
            )
        else:
            events.append(
                _event(
                    timestamp,
                    actor="Orchestrator",
                    kind="workflow",
                    title=event.replace("_", " ").title(),
                    detail="工作流审计事件。",
                    status="completed",
                    proof={key: value for key, value in item.items() if key != "timestamp"},
                )
            )

    for item in _read_jsonl(root / "model_audit.jsonl"):
        timestamp = str(item.get("created_at", ""))
        succeeded = item.get("status") == "succeeded"
        tools = tuple(str(name) for name in item.get("tool_names") or ())
        total_tokens = item.get("total_tokens")
        detail = (
            f"真实模型调用耗时 {item.get('duration_s', '?')} s"
            + (f"，共 {total_tokens} tokens" if total_tokens is not None else "")
            + (f"；返回工具：{', '.join(tools)}" if tools else "。")
        )
        events.append(
            _event(
                timestamp,
                actor=str(item.get("model", "Model Agent")),
                kind="model",
                title="模型推理与工具选择",
                detail=detail,
                status="completed" if succeeded else "failed",
                proof={
                    "request_id": item.get("request_id"),
                    "request_sha256": item.get("request_sha256"),
                    "response_sha256": item.get("response_sha256"),
                    "finish_reason": item.get("finish_reason"),
                    "tool_names": list(tools),
                    "prompt_tokens": item.get("prompt_tokens"),
                    "completion_tokens": item.get("completion_tokens"),
                },
            )
        )
        for name in tools:
            actor = "Diagnosis Agent" if name == "submit_diagnosis" else "Patch Agent"
            if name not in {"submit_diagnosis", "submit_patch"}:
                actor = "Repository Inspector"
            events.append(
                _event(
                    timestamp,
                    actor=actor,
                    kind="tool",
                    title=f"工具调用 · {name}",
                    detail=_TOOL_DETAILS.get(name, "模型调用了受限工具。"),
                    status="completed" if succeeded else "failed",
                    proof={"request_id": item.get("request_id"), "tool": name},
                )
            )

    events.sort(key=lambda item: (item["timestamp"], _KIND_ORDER.get(item["kind"], 99)))
    return events


def find_job_run_root(workspace_path: Path, attempt_count: int) -> Path | None:
    """Locate a job's run root without trusting incident-controlled path input."""

    workspace = workspace_path.resolve()
    if attempt_count < 1:
        return None
    runs = workspace / f"attempt-{attempt_count:03d}" / "runs"
    if not runs.is_dir():
        return None
    candidates = sorted(path.resolve() for path in runs.iterdir() if path.is_dir())
    for candidate in candidates:
        if runs.resolve() in candidate.parents:
            return candidate
    return None


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return ()
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                entries.append(value)
    return tuple(entries)


def _event(
    timestamp: str,
    *,
    actor: str,
    kind: str,
    title: str,
    detail: str,
    status: str,
    proof: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "actor": actor,
        "kind": kind,
        "title": title,
        "detail": detail,
        "status": status,
        "proof": proof,
    }


_TERMINAL_STATES = {
    "AWAITING_HUMAN_APPROVAL",
    "AWAITING_TECHNICAL_REVIEW",
    "REJECTED_BY_HUMAN",
    "PR_READY",
    "INFRA_ERROR",
}
_KIND_ORDER = {"workflow": 0, "model": 1, "tool": 2, "execution": 3, "validation": 4}
_TOOL_DETAILS = {
    "list_repository_files": "列出故障 Commit 中的仓库文件；只读且固定在指定 Commit。",
    "read_repository_file": "读取故障 Commit 的源代码；没有宿主文件系统或 QA 真值权限。",
    "search_repository": "在固定 Commit 中搜索代码证据。",
    "inspect_commit_diff": "比较正常 Commit 与故障 Commit。",
    "submit_diagnosis": "提交结构化根因、观察和建议；结束诊断阶段。",
    "submit_patch": "提交完整文件内容；可信主机据此构造 Git Diff。",
}
