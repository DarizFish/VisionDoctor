"""Streamlit operator console for the local PICK-A17 demonstration only."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import streamlit as st

from harness.gazebo.controller import PickCellController


def _controller() -> PickCellController:
    if "pick_cell_controller" not in st.session_state:
        st.session_state.pick_cell_controller = PickCellController()
    return st.session_state.pick_cell_controller


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
                overlay = part / "detection-overlay.png"
                if image.is_file() and overlay.is_file():
                    left, right = st.columns(2)
                    left.image(str(image), caption="RGB-D 相机原图")
                    right.image(str(overlay), caption="检测叠加")
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


def main() -> None:
    st.set_page_config(page_title="PICK-A17 Gazebo 操作台", layout="wide")
    st.title("PICK-A17 独立 Gazebo 抓取操作台")
    st.caption("本地 harness：真实 Gazebo Qt 窗口 + 场景相机证据；不连接产品或 Agent。")
    st.caption("台装 UR5e、低矮双工位；每次抓取先到工件正上方，再自上而下下压和抬起。")
    controller = _controller()
    try:
        status = controller.status()
    except Exception as exc:
        st.error(f"无法读取 Docker/Gazebo 状态：{exc}")
        return

    with st.sidebar:
        st.header("项目工作区")
        default = str(controller.default_workspace)
        workspace = st.text_input("重放工作区", value=st.session_state.get("workspace", default))
        st.session_state.workspace = workspace
        st.caption("可填写人工或后续修复后的本地 PICK-A17 项目副本。")
        if st.button("初始化故障项目与 Git bundle", use_container_width=True):
            _action("初始化演示项目", controller.bootstrap_project)
        st.divider()
        if st.button("启动工位并打开 Gazebo GUI", type="primary", use_container_width=True):
            _action("启动工位", controller.start_cell)
        if st.button("关闭工位", use_container_width=True):
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
                "拍照 / 采集 RGB-D", use_container_width=True, disabled=not status["running"]
            ):
                _action("场景拍照", controller.capture)
        with run_col:
            if st.button(
                "触发完整 A/B 机器视觉抓取",
                type="primary",
                use_container_width=True,
                disabled=not status["running"],
            ):
                _action("运行完整节拍", lambda: controller.run_cycle(Path(workspace)))
        st.caption(
            "完整节拍依次采集相机、运行所选项目的算法、到达预抓点、下压/抬起，"
            "记录轨迹并生成脱敏 bundle。"
        )

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
