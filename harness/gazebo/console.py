"""Streamlit operator console for the local PICK-A17 demonstration only."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import streamlit as st

from harness.gazebo.controller import PickCellController

WORKSPACE_INPUT_KEY = "pick_cell_workspace"


@st.cache_resource
def _controller() -> PickCellController:
    """One controller for every viewer: there is one cell, and its warm processes are shared."""

    return PickCellController()


@st.cache_resource
def _warmed() -> dict[str, float]:
    return {}


def _prewarm(controller: PickCellController, workspace: str) -> None:
    """Bring up the cell agent and the project's interpreter before the first click."""

    if workspace in _warmed() or _running_job():
        return
    _warmed()[workspace] = time.monotonic()

    def work() -> None:
        try:
            controller._warm_cell(verify=True)
            controller.warm_project(Path(workspace))
        except Exception:  # noqa: BLE001 - the click path warms again and reports failures
            _warmed().pop(workspace, None)

    threading.Thread(target=work, name="pick-cell-prewarm", daemon=True).start()


STAGE_LABEL = {
    "prepare": "准备：确认 MoveIt 与工位代理",
    "capture": "采集相机：RGB-D 与旁观视频",
    "perception": "视觉定位：运行项目感知程序",
    "command": "计算指令：运行项目指令程序",
    "motion": "机器人运动",
    "bundle": "生成观察证据包",
}
STEP_LABEL = {
    "APPROACH": "确认安全位",
    "PREGRASP": "移动到预抓点",
    "PICK": "下压到抓取位",
    "RETRACT_TO_PREGRASP": "抬回预抓点",
    "RETREAT": "返回安全位",
}


@st.cache_resource
def _jobs() -> dict[str, dict[str, Any]]:
    """Cycle jobs outlive a page refresh; the cell itself is shared by every viewer."""

    return {}


def _running_job() -> dict[str, Any] | None:
    job = _jobs().get("cycle")
    return job if job and job["status"] == "running" else None


def _start_cycle(controller: PickCellController, workspace: str) -> None:
    if _running_job():
        return
    job: dict[str, Any] = {
        "status": "running", "workspace": workspace, "started_at": time.monotonic(),
        "events": [], "result": None, "error": None, "refreshed": False,
    }
    _jobs()["cycle"] = job

    def work() -> None:
        try:
            job["result"] = controller.run_cycle(Path(workspace), progress=job["events"].append)
            job["status"] = "done"
        except Exception as exc:  # noqa: BLE001 - shown to the operator as it happened
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["status"] = "failed"
        job["finished_at"] = time.monotonic()

    threading.Thread(target=work, name="pick-cell-cycle", daemon=True).start()


def _rows(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Fold progress events into one row per stage, each with its own timing."""

    rows: list[dict[str, Any]] = []
    for event in list(job["events"]):
        stage = event["stage"]
        if stage in STAGE_LABEL:
            if rows and rows[-1]["end"] is None:
                rows[-1]["end"] = event["at"]
            rows.append({"stage": stage, "part": event["part"], "start": event["at"],
                         "end": None, "step": None, "moving": False})
        elif stage == "agent_step" and rows:
            rows[-1]["step"], rows[-1]["moving"] = event["detail"], False
        elif stage == "agent_moving" and rows:
            rows[-1]["moving"] = True
        elif stage in {"part_done", "no_command", "done"} and rows:
            if rows[-1]["end"] is None:
                rows[-1]["end"] = event["at"]
            if stage != "done":
                rows[-1]["outcome"] = event["detail"]
    return rows


@st.fragment(run_every=1.0)
def _show_cycle_progress() -> None:
    job = _jobs().get("cycle")
    if not job:
        return
    now = job.get("finished_at") or time.monotonic()
    elapsed = now - job["started_at"]
    events = list(job["events"])
    moving = next((item for item in events if item["stage"] == "agent_moving"), None)
    st.subheader("节拍进度")
    first, total, state = st.columns(3)
    first.metric(
        "点击到机器人开动",
        f"{moving['at'] - job['started_at']:.1f} s" if moving else "等待中…",
    )
    total.metric("已用时间" if job["status"] == "running" else "总用时", f"{elapsed:.1f} s")
    state.metric("状态", {"running": "进行中", "done": "完成", "failed": "失败"}[job["status"]])
    expected = 10  # prepare, four stages for each part, bundle
    done_rows = [row for row in _rows(job) if row["end"] is not None]
    st.progress(min(len(done_rows) / expected, 1.0))
    lines = []
    for row in _rows(job):
        finished = row["end"] is not None
        duration = (row["end"] or now) - row["start"]
        label = STAGE_LABEL[row["stage"]]
        if row["stage"] == "motion" and row["step"] and not finished:
            step = STEP_LABEL.get(row["step"], row["step"])
            label += f" · {step}" + ("（运动中）" if row["moving"] and not finished else "")
        if row.get("outcome"):
            label += f" · 结果 {row['outcome']}"
        mark = "✅" if finished else "⏳"
        part = f"工件 {row['part']} · " if row["part"] else ""
        lines.append(f"{mark} {part}{label} — {duration:.1f} s")
    st.markdown("\n".join(f"- {line}" for line in lines) or "- 正在启动…")
    if job["status"] == "failed":
        st.error(f"节拍失败：{job['error']}")
    if job["status"] != "running" and not job["refreshed"]:
        job["refreshed"] = True
        st.rerun(scope="app")


def _action(label: str, callback) -> None:
    try:
        with st.spinner(label):
            value = callback()
        st.session_state.pick_cell_last_action = value
        st.success(label + "完成")
    except Exception as exc:
        st.error(f"{label}失败：{exc}")


def _show_latest_run(controller: PickCellController, latest: dict[str, object] | None) -> None:
    if not latest:
        st.info("尚未运行完整 A/B 节拍。")
        return
    st.subheader("最近一次完整节拍")
    st.json({key: value for key, value in latest.items() if key != "bundle_path"})
    bundle_path = latest.get("bundle_path")
    if isinstance(bundle_path, str):
        run_root = Path(bundle_path).parent
        for part_id in ("A", "B"):
            part = run_root / "parts" / part_id
            with st.expander(f"工件 {part_id}", expanded=True):
                result = controller._read_json(part / "result.json")
                if result:
                    st.json(result)
                image = part / "capture" / "rgb.png"
                if image.is_file():
                    st.image(str(image), caption="RGB-D 相机原图")
                observer_clip = part / "capture" / "observer.gif"
                if observer_clip.is_file():
                    st.image(str(observer_clip), caption="观察相机短片（真实相机帧）")
                motion = controller._read_json(part / "motion.json")
                if motion:
                    st.caption("机器人轨迹摘要")
                    st.json(
                        {
                            "success": motion.get("success"),
                            "steps": [step.get("name") for step in motion.get("steps", [])],
                            "sample_count": len(motion.get("joint_trajectory", [])),
                        }
                    )
                log_path = part / "algorithm" / "application.jsonl"
                if log_path.is_file():
                    st.caption("算法结构化日志")
                    st.code(log_path.read_text(encoding="utf-8"), language="json")


def _remember_workspace_choice(controller: PickCellController) -> None:
    controller.remember_workspace(Path(str(st.session_state[WORKSPACE_INPUT_KEY])))


def main() -> None:
    st.set_page_config(page_title="PICK-A17 Gazebo 操作台", layout="wide")
    st.title("PICK-A17 独立 Gazebo 抓取操作台")
    st.caption("本地 harness：真实 Gazebo Qt 窗口 + 场景相机证据；不连接产品或 Agent。")
    st.caption("台装 UR5e、低矮双工位；每次抓取先到工件正上方，再自上而下下压和抬起。")
    controller = _controller()
    snapshot = st.session_state.get("pick_cell_status")
    try:
        # A running cycle owns the cell; probing Docker and the GUI window on every
        # rerun would only compete with it, so the last snapshot stands in.
        status = snapshot if snapshot and _running_job() else controller.status()
    except Exception as exc:
        st.error(f"无法读取 Docker/Gazebo 状态：{exc}")
        return
    st.session_state["pick_cell_status"] = status

    busy = _running_job() is not None
    if status["running"] and not busy:
        _prewarm(controller, str(st.session_state.get(WORKSPACE_INPUT_KEY)
                                 or controller.selected_workspace))
    if WORKSPACE_INPUT_KEY not in st.session_state:
        st.session_state[WORKSPACE_INPUT_KEY] = str(controller.selected_workspace)

    with st.sidebar:
        st.header("项目工作区")
        workspace = st.text_input(
            "重放工作区",
            key=WORKSPACE_INPUT_KEY,
            on_change=_remember_workspace_choice,
            args=(controller,),
        )
        st.caption("可填写人工或后续修复后的本地 PICK-A17 项目副本。")
        if st.button("初始化故障项目与 Git bundle", use_container_width=True):
            _action("初始化演示项目", controller.bootstrap_project)
        st.divider()
        if st.button(
            "启动工位并打开 Gazebo GUI", type="primary", use_container_width=True, disabled=busy
        ):
            _action("启动工位", controller.start_cell)
        if st.button("关闭工位", use_container_width=True, disabled=busy):
            _action("关闭工位", controller.stop_cell)

    status_left, status_right = st.columns((1, 2))
    with status_left:
        st.subheader("工位状态")
        st.json(
            {
                "image_ready": status["image_ready"],
                "running": status["running"],
                "gazebo_gui": status["gazebo_gui_running"],
                "gazebo_window": status["gazebo_window"],
                "camera_bridge": status["camera_bridge_running"],
                "move_group": status["move_group_running"],
            }
        )
    with status_right:
        st.subheader("操作")
        capture_col, run_col = st.columns(2)
        with capture_col:
            if st.button(
                "拍照 / 采集 RGB-D", use_container_width=True,
                disabled=not status["running"] or busy,
            ):
                _action("场景拍照", controller.capture)
        with run_col:
            st.button(
                "节拍进行中…" if busy else "触发完整 A/B 机器视觉抓取",
                type="primary",
                use_container_width=True,
                disabled=not status["running"] or busy,
                on_click=_start_cycle,
                args=(controller, workspace),
            )
        st.caption(
            "完整节拍依次采集相机、运行所选项目的算法、到达预抓点、下压/抬起，"
            "记录轨迹并生成脱敏 bundle。点击后立即在下方显示各阶段进度。"
        )
        _show_cycle_progress()

    if action := st.session_state.get("pick_cell_last_action"):
        with st.expander("最近操作返回", expanded=False):
            st.json(action)
    _show_latest_run(controller, status.get("latest_run"))

    latest = status.get("latest_run")
    if isinstance(latest, dict) and isinstance(latest.get("run_id"), str):
        st.subheader("导出观察证据包")
        export_root = st.text_input("导出目录", value=str(controller.runtime_root / "exports"))
        if st.button("导出本轮 ObservationBundle"):
            _action(
                "导出证据包",
                lambda: controller.export_bundle(str(latest["run_id"]), Path(export_root)),
            )


if __name__ == "__main__":
    main()
