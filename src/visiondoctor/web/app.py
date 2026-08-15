from __future__ import annotations

import base64
import html
import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any

import streamlit as st


def _api(path: str, *, method: str = "GET", payload: dict | None = None):
    base = os.getenv("VISIONDOCTOR_API_URL", "http://127.0.0.1:8000").rstrip("/")
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        base + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            content_type = response.headers.get_content_type()
            data = response.read()
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("detail", str(exc))
        except (json.JSONDecodeError, AttributeError):
            detail = str(exc)
        if isinstance(detail, dict):
            detail = json.dumps(detail, ensure_ascii=False)
        raise RuntimeError(str(detail)) from exc
    if content_type == "application/json":
        return json.loads(data)
    return data


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return _api(path, method="POST", payload=payload)
    except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
        st.error(f"操作没有完成：{exc}")
        return None


def _delete(path: str) -> dict[str, Any] | None:
    try:
        return _api(path, method="DELETE")
    except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
        st.error(f"删除没有完成：{exc}")
        return None


def _base_url() -> str:
    return os.getenv("VISIONDOCTOR_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _time_label(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone().strftime("%m-%d %H:%M")


def _clock_label(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone().strftime("%H:%M:%S")


def _phase_label(phase: str) -> str:
    return {
        "collecting": "正在了解问题",
        "running": "正在深入诊断",
        "needs_attention": "需要补充信息",
        "review": "等待你的确认",
        "completed": "本轮已完成",
    }.get(phase, "诊断进行中")


def _apply_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; max-width: 1500px;}
        [data-testid="stSidebar"] {border-right: 1px solid rgba(120,120,120,.16);}
        [data-testid="stChatMessage"] {border-radius: 16px; padding: .45rem .8rem;}
        .vd-kicker {display: block; min-height: 1.4rem; overflow: visible;
                    font-size: .78rem; line-height: 1.45; letter-spacing: .08em;
                    color: #6b7280; padding-top: .12rem; margin: 0 0 .2rem;}
        .vd-title {font-size: 2rem; font-weight: 720; line-height: 1.2; margin-bottom: .25rem;}
        .vd-subtle {color: #6b7280; font-size: .92rem;}
        .vd-source {padding: .8rem .9rem; border: 1px solid rgba(120,120,120,.18);
                    border-radius: 12px; margin-bottom: .55rem; overflow-wrap: anywhere;}
        .vd-source-ready {border-left: 4px solid #22a06b;}
        .vd-source-wait {border-left: 4px solid #d39b21;}
        .vd-activity {padding: .55rem .75rem; border-left: 2px solid rgba(76,110,245,.45);
                      margin-left: .35rem; color: #374151;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(
    sessions: list[dict[str, Any]],
    projects: list[dict[str, Any]],
    health: dict[str, Any],
) -> str | None:
    with st.sidebar:
        st.markdown("## VisionDoctor")
        st.caption("机器视觉诊断助手")
        if st.button("＋ 新建诊断", type="primary", use_container_width=True):
            created = _post("/api/v1/sessions", {})
            if created:
                st.session_state.selected_session = created["session_id"]
                st.rerun()
        st.markdown("#### 项目与诊断")
        selected = st.session_state.get("selected_session")
        available_ids = [item["session_id"] for item in sessions]
        if selected not in available_ids:
            selected = available_ids[0] if available_ids else None

        def render_session(item: dict[str, Any]) -> None:
            marker = "●" if item["session_id"] == selected else "○"
            label = f"{marker} {item['title']}\n\n{_phase_label(item['phase'])}"
            if st.button(
                label,
                key=f"session-{item['session_id']}",
                use_container_width=True,
            ):
                st.session_state.selected_session = item["session_id"]
                st.rerun()

        rendered_sessions: set[str] = set()
        for project in projects:
            project_id = str(project["project_id"])
            project_sessions = [
                item
                for item in sessions
                if (item.get("project") or {}).get("project_id") == project_id
            ]
            with st.container(border=True):
                st.markdown(f"**{project.get('name', '机器视觉项目')}**")
                st.caption(
                    f"{project.get('component_count', 0)} 个组件 · "
                    f"{len(project_sessions)} 个诊断会话"
                )
                for item in project_sessions:
                    rendered_sessions.add(str(item["session_id"]))
                    render_session(item)
                if st.button(
                    "＋ 在此项目中新建诊断",
                    key=f"new-session-{project_id}",
                    use_container_width=True,
                ):
                    repository_path = str(
                        (project.get("source") or {}).get("repository_path") or ""
                    )
                    created = _post(
                        "/api/v1/sessions",
                        {"repository_path": repository_path},
                    )
                    if created:
                        st.session_state.selected_session = created["session_id"]
                        st.rerun()
        unassigned = [
            item for item in sessions if str(item["session_id"]) not in rendered_sessions
        ]
        if unassigned:
            st.caption("尚未连接项目")
            for item in unassigned:
                render_session(item)
        if not projects and not sessions:
            st.caption("还没有项目或诊断会话")
        st.divider()
        ready = bool(
            health.get("model", {}).get("configured") and health.get("docker", {}).get("available")
        )
        st.caption("🟢 诊断服务已就绪" if ready else "🟠 诊断服务需要检查")
        if health.get("gazebo", {}).get("available"):
            st.caption("🟢 仿真环境可用")
        else:
            st.caption("⚪ 仿真环境未就绪")
        vision = health.get("vision_model", {})
        if vision.get("available") and vision.get("model_ready"):
            st.caption("🟢 图片理解已就绪")
        else:
            st.caption("🟠 图片理解需要启动")
    return selected


def _render_attachment(session_id: str, attachment: dict[str, Any]) -> None:
    url = f"{_base_url()}/api/v1/sessions/{session_id}/attachments/{attachment['attachment_id']}"
    if str(attachment.get("media_type", "")).startswith("image/"):
        st.image(url, caption=attachment.get("name"), width=360)
    else:
        size_kb = float(attachment.get("size_bytes", 0)) / 1024
        st.markdown(f"📎 [{attachment.get('name', '附件')}]({url}) · {size_kb:.1f} KB")
        analysis = attachment.get("content_analysis") or {}
        if analysis:
            icon = "✓" if analysis.get("status") == "ready" else "◐"
            st.caption(f"{icon} {analysis.get('description', '附件文字已加入诊断')}")
            limitations = analysis.get("limitations") or []
            if limitations:
                st.caption("读取范围说明：" + "；".join(str(item) for item in limitations))


def _render_image_assessments(assessments: list[dict[str, Any]]) -> None:
    if not assessments:
        return
    st.markdown("#### 从图片中观察到")
    for assessment in assessments:
        confidence = float(assessment.get("confidence", 0))
        confidence_label = (
            "较高" if confidence >= 0.75 else "中等" if confidence >= 0.45 else "较低"
        )
        with st.container(border=True):
            st.markdown(f"**{assessment.get('visible_name', '图片')}**")
            for observation in assessment.get("observations") or ():
                st.markdown(f"- {observation}")
            relevance = str(assessment.get("diagnostic_relevance", "")).strip()
            if relevance:
                st.markdown(f"**与当前问题的关系：** {relevance}")
            limitations = assessment.get("limitations") or []
            if limitations:
                st.caption("仅凭这张图仍不能确认：" + "；".join(limitations))
            st.caption(f"对画面观察的把握：{confidence_label}")
    st.info("图片观察用于提供诊断线索；修复是否有效仍由独立的效果检查判定。")


def _render_attachment_assessments(assessments: list[dict[str, Any]]) -> None:
    if not assessments:
        return
    st.markdown("#### 从附件中分析到")
    for assessment in assessments:
        with st.container(border=True):
            st.markdown(f"**{assessment.get('visible_name', '附件')}**")
            for observation in assessment.get("observations") or ():
                st.markdown(f"- {observation}")
            relevance = str(assessment.get("diagnostic_relevance", "")).strip()
            if relevance:
                st.markdown(f"**与当前问题的关系：** {relevance}")
            limitations = assessment.get("limitations") or []
            if limitations:
                st.caption("读取与分析范围：" + "；".join(str(item) for item in limitations))
    st.info("附件分析用于提供诊断线索；修复是否有效仍由独立的效果检查判定。")


def _render_messages(session: dict[str, Any]) -> None:
    messages = session.get("messages") or []
    if not messages:
        st.info("先说说发生了什么。你不需要整理成表单，也不需要判断是哪次代码改动出了问题。")
        st.markdown(
            "例如：升级视觉节点后，机器人每次都会偏到目标右侧；也可以直接上传现场截图，"
            "或直接使用右侧仿真现场采集相机画面。"
        )
        return
    for message in messages:
        if message["role"] == "source":
            st.caption(f"📥 资料更新 · {message['content']}")
            for attachment in message.get("attachments") or ():
                _render_attachment(session["session_id"], attachment)
            _render_image_assessments(list(message.get("image_assessments") or ()))
            continue
        with st.chat_message(message["role"]):
            st.markdown(str(message.get("content", "")))
            for attachment in message.get("attachments") or ():
                _render_attachment(session["session_id"], attachment)
            if message["role"] == "assistant":
                _render_image_assessments(list(message.get("image_assessments") or ()))
                _render_attachment_assessments(
                    list(message.get("attachment_assessments") or ())
                )
            if message["role"] == "assistant" and message.get("next_actions"):
                with st.expander("接下来可以做什么"):
                    for action in message["next_actions"]:
                        st.markdown(f"- {action}")


def _encode_uploads(files: list[Any]) -> list[dict[str, str]]:
    known_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".txt": "text/plain",
        ".log": "text/plain",
        ".json": "application/json",
        ".pdf": "application/pdf",
        ".zip": "application/zip",
    }
    values: list[dict[str, str]] = []
    for uploaded in files:
        suffix = os.path.splitext(uploaded.name)[1].lower()
        media_type = known_types.get(suffix) or uploaded.type or mimetypes.guess_type(
            uploaded.name
        )[0]
        if not media_type:
            media_type = "text/plain"
        values.append(
            {
                "name": uploaded.name,
                "media_type": media_type,
                "content_base64": base64.b64encode(uploaded.getvalue()).decode("ascii"),
            }
        )
    return values


def _render_composer(session_id: str | None, health: dict[str, Any]) -> None:
    nonce = int(st.session_state.get("attachment_nonce", 0))
    files = st.file_uploader(
        "补充图片或文件",
        type=("png", "jpg", "jpeg", "webp", "txt", "log", "json", "pdf", "zip"),
        accept_multiple_files=True,
        max_upload_size=10,
        key=f"conversation-attachments-{nonce}",
        help=(
            "图片会由视觉模型直接查看；TXT/LOG 会读取文字，JSON 会先校验，PDF 会"
            "提取可搜索文字，ZIP 只读取受支持的文本文件。无法安全解析时本轮会明确失败。"
        ),
    )
    st.caption(
        "支持图片、TXT、LOG、JSON、PDF、ZIP；每次最多 8 个，单个附件不超过 10 MB，"
        "本轮合计不超过 20 MB。提取的文字会发送给诊断模型，请勿上传密钥或凭据。"
    )
    prompt = st.chat_input(
        "描述现象、回答问题，或告诉诊断助手接下来要检查什么……",
        disabled=not health.get("model", {}).get("configured", False),
    )
    if prompt is None:
        return
    uploads = list(files or ())
    encoded_uploads = _encode_uploads(uploads)
    has_images = any(
        str(item["media_type"]).startswith("image/") for item in encoded_uploads
    )
    vision = health.get("vision_model", {})
    if has_images and not (vision.get("available") and vision.get("model_ready")):
        st.error("图片理解服务尚未就绪，图片不会被降级处理。请先启动视觉模型后重试。")
        return
    if session_id is None:
        created = _post("/api/v1/sessions", {})
        if not created:
            return
        session_id = str(created["session_id"])
        st.session_state.selected_session = session_id
    payload = {"message": prompt, "attachments": encoded_uploads}
    has_documents = any(
        not str(item["media_type"]).startswith("image/") for item in encoded_uploads
    )
    if has_images and has_documents:
        spinner = "正在查看图片并逐份分析附件内容……"
    elif has_images:
        spinner = "视觉模型正在逐张查看图片……"
    elif has_documents:
        spinner = "正在安全读取并逐份分析附件内容……"
    else:
        spinner = "诊断助手正在查看你提供的信息……"
    with st.spinner(spinner):
        result = _post(f"/api/v1/sessions/{session_id}/turn", payload)
    if result:
        st.session_state.attachment_nonce = nonce + 1
        st.rerun()


def _render_connections(session: dict[str, Any]) -> None:
    st.markdown("### 当前连接")
    repository = session.get("repository") or {}
    repository_path = str(repository.get("path") or "")
    if repository.get("available"):
        st.markdown(
            '<div class="vd-source vd-source-ready"><b>项目路径</b><br>'
            f'<span class="vd-subtle">{html.escape(repository_path)}</span></div>',
            unsafe_allow_html=True,
        )
        with st.expander("更换项目", expanded=False):
            _render_repository_connector(session, repository_path, "重新连接")
    else:
        with st.container(border=True):
            st.markdown("**项目路径**")
            st.caption("连接项目后，诊断助手会在会话中动态理解代码与运行关系。")
            _render_repository_connector(session, "", "连接项目")
    _render_simulation(session)


def _render_repository_connector(
    session: dict[str, Any], current_path: str, button_label: str
) -> None:
    repository_path = st.text_input(
        "项目所在文件夹",
        value=current_path,
        placeholder=r"C:\projects\vision-node",
        key=f"repository-{session['session_id']}",
    )
    if st.button(
        button_label,
        key=f"connect-{session['session_id']}",
        disabled=not repository_path.strip(),
        use_container_width=True,
    ):
        with st.spinner("正在识别项目……"):
            result = _post(
                f"/api/v1/sessions/{session['session_id']}/repository",
                {"repository_path": repository_path, "semantic_understanding": True},
            )
        if result:
            st.rerun()


def _activity_label(event: dict[str, Any]) -> str:
    kind = event.get("kind")
    if kind == "model":
        return "诊断助手完成了一轮分析"
    if kind == "tool":
        return "诊断助手查看了相关代码"
    if kind == "execution":
        return "在独立副本中尝试并运行了修复"
    if kind == "validation":
        return "检查了修复后的实际效果"
    state = str(event.get("proof", {}).get("state", ""))
    return {
        "REPRODUCING": "正在复现问题",
        "DIAGNOSING": "正在定位原因",
        "ROOT_CAUSE_CONFIRMED": "已经找到有证据支持的原因",
        "PATCH_GENERATING": "正在准备代码改动",
        "VERIFYING": "正在验证改动",
        "PATCH_REJECTED": "一个无效方案已被退回",
        "AWAITING_HUMAN_APPROVAL": "修复结果等待你的确认",
        "AWAITING_TECHNICAL_REVIEW": "仿真验证需要人工复核，候选修复已保留",
    }.get(state, "诊断流程向前推进了一步")


def _render_activity(events: list[dict[str, Any]]) -> None:
    if not events:
        st.caption("任务已进入队列，活动记录会在开始工作后出现。")
        return
    seen: list[str] = []
    for event in events:
        label = _activity_label(event)
        if seen and seen[-1] == label:
            continue
        seen.append(label)
    for label in seen[-8:]:
        st.markdown(f'<div class="vd-activity">{label}</div>', unsafe_allow_html=True)


def _render_activity_detail(events: list[dict[str, Any]], key: str) -> None:
    """Full engineering record, collapsed by default so the product surface stays plain."""
    if not events:
        return
    with st.expander(f"技术细节：完整工作记录（{len(events)} 条）", expanded=False):
        st.caption(
            "按发生顺序列出诊断助手的每一轮分析、每一次被调用的只读工具、每一次隔离执行和"
            "效果检查。上面的进度说明只保留结论，这份记录用于技术复核。"
        )
        rows = ["| 时间 | 执行方 | 环节 | 说明 |", "|---|---|---|---|"]
        for event in events:
            detail = str(event.get("detail") or "").replace("|", "／").replace("\n", " ")
            rows.append(
                f"| {_clock_label(event.get('timestamp'))} "
                f"| {html.escape(str(event.get('actor') or ''))} "
                f"| {html.escape(str(event.get('title') or ''))} "
                f"| {html.escape(detail)} |"
            )
        st.markdown("\n".join(rows))
        with_proof = [event for event in events if event.get("proof")]
        if with_proof and st.checkbox("显示每一步的证据字段", key=f"activity-proof-{key}"):
            for event in with_proof:
                st.markdown(f"**{event.get('title') or ''}** · {event.get('actor') or ''}")
                st.code(
                    json.dumps(event["proof"], ensure_ascii=False, indent=2),
                    language="json",
                )


_METRIC_LABELS = {
    "unit_tests": "代码自身检查",
    "translation_rmse": "目标位置误差",
    "mean_rotation_error": "目标方向误差",
    "mean_reprojection_error": "图像对齐误差",
    "scene_pass_rate": "场景通过率",
    "case_pass_rate": "案例通过率",
    "structured_output_contract": "任务输出检查",
    "task_output_contract": "任务输出格式",
    "detection_precision": "检测准确率",
    "detection_recall": "检测召回率",
    "detection_mean_iou": "检测框重合度",
    "ocr_character_error_rate": "文字字符错误率",
    "ocr_word_error_rate": "文字词语错误率",
    "segmentation_mean_iou": "分割区域重合度",
    "segmentation_pixel_accuracy": "分割像素准确率",
    "segmentation_boundary_f1": "分割边界准确度",
    "task_adapter_contract": "任务类型一致性",
    "evidence_hash_integrity": "输入数据完整性",
    "reference_provenance_integrity": "参考结果完整性",
    "fixed_robot_motion": "机器人固定动作",
    "tcp_validation_rmse": "机器人到位误差",
    "latency_growth": "运行速度变化",
}


def _metric_value(value: Any, unit: str | None) -> str:
    if isinstance(value, bool):
        return "通过" if value else "未通过"
    if unit == "m":
        return f"{float(value) * 1000:.2f} mm"
    if unit == "rad":
        return f"{float(value) * 57.2958:.2f}°"
    if unit == "ratio":
        return f"{float(value) * 100:.1f}%"
    if unit == "px":
        return f"{float(value):.2f} px"
    return str(value)


def _render_approval(run: dict[str, Any], session_id: str) -> None:
    if run.get("state") != "AWAITING_HUMAN_APPROVAL":
        return
    st.markdown("### 需要你的决定")
    st.write("检查已经完成。系统不会自行改动原仓库，请你决定如何处理这个修复建议。")
    note = st.text_area("备注（可选）", key=f"approval-note-{run['run_id']}")
    columns = st.columns(3)
    action = None
    if columns[0].button("接受修复", type="primary", use_container_width=True):
        action = "approve"
    if columns[1].button("继续补充验证", use_container_width=True):
        action = "additional_testing"
    if columns[2].button("退回这个方案", use_container_width=True):
        action = "reject"
    if action:
        outcome = _post(
            f"/api/v1/runs/{run['run_id']}/approval",
            {"action": action, "actor": "工作台用户", "note": note},
        )
        if outcome:
            if action in {"reject", "additional_testing"}:
                prefix = "我退回了上一轮方案" if action == "reject" else "我希望增加验证"
                feedback = f"{prefix}。" + (f"备注：{note}" if note.strip() else "请继续调查。")
                _post(
                    f"/api/v1/sessions/{session_id}/feedback",
                    {"message": feedback},
                )
            st.rerun()


def _render_run_result(run_id: str, session_id: str) -> None:
    try:
        run = _api(f"/api/v1/runs/{run_id}")
        validation = _api(f"/api/v1/runs/{run_id}/validation")
    except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
        st.warning(f"诊断已结束，但结果暂时无法读取：{exc}")
        return
    passed = validation.get("decision") == "PASS"
    technical_review = run.get("state") == "AWAITING_TECHNICAL_REVIEW"
    if technical_review:
        st.warning(
            "候选修复已通过代码、视觉和几何检查，但机器人仿真连续出现规划或执行波动。"
            "系统已保留候选修复并停止自动改代码，等待技术复核。"
        )
    elif run.get("state") == "REJECTED_BY_HUMAN":
        st.warning("这份方案虽然通过了效果检查，但已经被你退回，不会进入原仓库。")
    elif run.get("state") == "ADDITIONAL_TESTING":
        st.info("你要求增加验证。当前结果保留，诊断助手会在下一轮继续检查。")
    elif passed:
        st.success("本轮修复建议已经通过复现、代码检查和效果验证。")
    else:
        st.warning("本轮方案没有通过全部检查，诊断助手可以继续尝试。")
    overview, evidence_tab, changes = st.tabs(("验证结果", "输入证据", "代码改动"))
    with overview:
        cards = st.columns(3)
        cards[0].metric("测试案例", int(validation.get("case_count", 0)))
        cards[1].metric("通过案例", len(validation.get("passed_cases") or ()))
        cards[2].metric(
            "结论",
            (
                "等待技术复核"
                if technical_review
                else "可以交给你确认" if passed else "需要继续诊断"
            ),
        )
        rows = []
        for item in validation.get("metric_results") or ():
            name = str(item.get("name"))
            if name not in _METRIC_LABELS:
                continue
            rows.append(
                {
                    "检查内容": _METRIC_LABELS[name],
                    "实际结果": _metric_value(item.get("value"), item.get("unit")),
                    "要求": _metric_value(item.get("threshold"), item.get("unit")),
                    "是否符合": "✓" if item.get("passed") else "✕",
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    with evidence_tab:
        evidence = _api(f"/api/v1/runs/{run_id}/evidence")
        if not evidence:
            st.info("本轮没有可显示的输入证据。")
        else:
            case_id = st.selectbox(
                "选择一个测试案例",
                [item["case_id"] for item in evidence],
                key=f"scene-{run_id}",
            )
            selected = next(item for item in evidence if item["case_id"] == case_id)
            if selected.get("task_kind") != "rgbd_pose":
                st.markdown("#### 本次输入")
                structured = selected.get("structured_input") or {}
                st.json(structured.get("value"))
                for artifact in selected.get("input_artifacts") or ():
                    artifact_id = urllib.parse.quote(str(artifact["artifact_id"]), safe="")
                    artifact_url = (
                        f"{_base_url()}/api/v1/runs/{run_id}/evidence/"
                        f"{urllib.parse.quote(case_id, safe='')}/artifacts/{artifact_id}"
                    )
                    if str(artifact.get("media_type", "")).startswith("image/"):
                        st.image(artifact_url, caption="导致本案例结果的视觉输入")
                    else:
                        st.link_button("查看输入文件", artifact_url)
                st.caption("这里显示的是候选程序收到的输入；独立参考结果不会在此处公开。")
            else:
                columns = st.columns(2)
                columns[0].image(
                    f"{_base_url()}/api/v1/runs/{run_id}/evidence/{case_id}/rgb",
                    caption="相机看到的彩色画面",
                )
                columns[1].image(
                    f"{_base_url()}/api/v1/runs/{run_id}/evidence/{case_id}/depth.png",
                    caption="距离分布；亮度变化表示测得的远近",
                )
                st.caption(
                    "这两幅图来自同一次采集：左侧用于观察目标，右侧用于计算三维位置。"
                )
    with changes:
        diff = _api(f"/api/v1/runs/{run_id}/diff").decode("utf-8")
        st.write("以下内容是系统建议的代码改动，尚未合并到原仓库。")
        st.code(diff, language="diff")
    _render_approval(run, session_id)
    if run.get("state") in {"REJECTED_BY_HUMAN", "ADDITIONAL_TESTING"} and st.button(
        "根据反馈继续诊断", type="primary", key=f"continue-{run_id}"
    ):
        created = _post(
            f"/api/v1/sessions/{session_id}/runs",
            {
                "validation_plan_confirmed": True,
                "safe_change_scope_confirmed": True,
            },
        )
        if created:
            st.rerun()


def _render_job(session: dict[str, Any]) -> None:
    latest = session.get("latest_job")
    if not latest:
        if session["readiness"].get("can_start"):
            st.success("信息已经足够，可以开始深入诊断。")
            st.caption(
                "开始后，系统会在独立副本中复现问题并尝试修复；最多改动 3 个文件，"
                "原仓库不会被直接修改。输出正确性和性能会由当前任务的固定标准检查。"
            )
            if st.button(
                "同意范围并开始诊断",
                type="primary",
                key=f"start-run-{session['session_id']}",
            ):
                created = _post(
                    f"/api/v1/sessions/{session['session_id']}/runs",
                    {
                        "validation_plan_confirmed": True,
                        "safe_change_scope_confirmed": True,
                    },
                )
                if created:
                    st.rerun()
        return
    status = latest.get("status")
    if status == "QUEUED":
        st.info("本次诊断正在等待可用资源。你可以切换到其他会话继续交流。")
    elif status == "RUNNING":
        st.info("诊断助手正在复现问题、检查代码并验证候选方案。你仍然可以继续补充信息。")
    elif status == "FAILED":
        st.error("本次诊断没有完成。你可以继续对话补充信息，然后重新发起。")
        with st.expander("查看失败原因"):
            error = str(latest.get("error") or "")
            if "without calling required terminal tool" in error:
                st.write("诊断助手没有在本轮给出完整的修复方案，系统已停止且没有采用不完整结果。")
            else:
                st.write(error or "没有返回具体原因")
        if session["readiness"].get("can_start") and st.button(
            "重新开始深入诊断",
            type="primary",
            key=f"retry-run-{latest['job_id']}",
        ):
            created = _post(
                f"/api/v1/sessions/{session['session_id']}/runs",
                {
                    "validation_plan_confirmed": True,
                    "safe_change_scope_confirmed": True,
                },
            )
            if created:
                st.rerun()
    events = list(latest.get("events") or [])
    if status in {"QUEUED", "RUNNING"}:
        left, right = st.columns([4, 1])
        with left:
            _render_activity(events)
        if right.button("刷新进度", key=f"refresh-{latest['job_id']}", use_container_width=True):
            st.rerun()
    elif status == "SUCCEEDED":
        _render_run_result(str(latest["run_id"]), str(session["session_id"]))
    _render_activity_detail(events, str(latest["job_id"]))


def _render_conversation(session_id: str | None, health: dict[str, Any]) -> None:
    if session_id is None:
        st.markdown('<div class="vd-kicker">新的诊断</div>', unsafe_allow_html=True)
        st.markdown('<div class="vd-title">今天遇到了什么问题？</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="vd-subtle">直接描述现象。代码、图片和仿真信息都可以稍后补充。</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        _render_messages({"session_id": "", "messages": []})
        _render_composer(None, health)
        return
    session = _api(f"/api/v1/sessions/{session_id}")
    st.markdown('<div class="vd-kicker">诊断会话</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="vd-title">{session["title"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="vd-subtle">{_phase_label(session["phase"])} · '
        f"更新于 {_time_label(session['updated_at'])}</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    conversation, sources = st.columns([1.75, 0.7], gap="large")
    with conversation:
        _render_messages(session)
        _render_job(session)
        _render_composer(session_id, health)
    with sources:
        _render_connections(session)
        st.write("")
        with st.expander("会话管理", expanded=False):
            latest = session.get("latest_job") or {}
            running = latest.get("status") in {"QUEUED", "RUNNING"}
            confirm = st.checkbox(
                "我确认删除这个会话",
                key=f"confirm-delete-{session_id}",
                disabled=running,
            )
            if running:
                st.caption("当前诊断结束后才能删除会话。")
            elif st.button(
                "删除会话",
                key=f"delete-session-{session_id}",
                disabled=not confirm,
                use_container_width=True,
            ):
                deleted = _delete(f"/api/v1/sessions/{session_id}")
                if deleted:
                    st.session_state.selected_session = None
                    st.toast("会话已移到本机回收区")
                    st.rerun()


def _simulation_action(action: str, session_id: str | None) -> None:
    suffix = session_id[-6:] if session_id else "public"
    created = _post(
        "/api/v1/simulation/actions",
        {
            "action": action,
            "case_id": f"scene-{suffix}",
            "session_id": session_id,
        },
    )
    if created:
        st.rerun()


@st.fragment(run_every="3s")
def _render_simulation(session: dict[str, Any]) -> None:
    session_id = str(session["session_id"])
    status = _api("/api/v1/simulation")
    visual = status["visual"]
    active = status.get("active_operation")
    connected = bool(visual["gazebo_gui_running"] or visual["gazebo_server_running"])
    st.markdown(
        '<div class="vd-source '
        + ("vd-source-ready" if connected else "vd-source-wait")
        + '"><b>仿真</b><br><span class="vd-subtle">'
        + ("Gazebo 现场已连接" if connected else "Gazebo 现场未启动")
        + "</span></div>",
        unsafe_allow_html=True,
    )
    with st.expander("仿真现场", expanded=bool(active)):
        st.caption("Gazebo 官方 3D 窗口；状态和采集结果会同步到当前会话。")
        controls = st.columns(2)
        if controls[0].button(
        "打开 3D 现场",
        disabled=visual["gazebo_gui_running"] or active is not None,
        key=f"simulation-open-{session_id}",
        use_container_width=True,
        ):
            _simulation_action("start_gui", session_id)
        if controls[1].button(
        "让机器人走一遍",
        disabled=not visual["gazebo_gui_running"] or active is not None,
        key=f"simulation-motion-{session_id}",
        type="primary",
        use_container_width=True,
        ):
            _simulation_action("run_motion", session_id)
        has_capture = bool(session.get("simulation_capture_operation_ids"))
        if st.button(
            "用当前项目观测机器人",
            disabled=(
                not visual["gazebo_gui_running"]
                or active is not None
                or not has_capture
                or not bool((session.get("repository") or {}).get("available"))
            ),
            key=f"simulation-project-motion-{session_id}",
            type="primary",
            use_container_width=True,
        ):
            _simulation_action("run_project_observation", session_id)
        st.caption("项目观测会把本会话刚采集的 RGB-D 输入当前代码，再让机器人走向代码给出的目标。")
        if controls[0].button(
        "采集相机与距离",
        disabled=active is not None,
        key=f"simulation-capture-{session_id}",
        use_container_width=True,
        ):
            _simulation_action("capture_rgbd", session_id)
        if controls[1].button(
        "关闭现场",
        disabled=not visual["gazebo_gui_running"],
        key=f"simulation-stop-{session_id}",
        use_container_width=True,
        ):
            _simulation_action("stop", session_id)
        if st.button("同步状态", key=f"simulation-refresh-{session_id}"):
            st.rerun()
        latest = status.get("latest_operation") or {}
        if active:
            st.info("仿真正在执行；刷新后会显示最新结果。")
        elif latest.get("status") == "FAILED":
            st.error("最近一次仿真操作没有完成。")
            st.caption(str(latest.get("error") or "未返回具体原因"))
        elif latest.get("action") == "run_motion" and latest.get("result"):
            result = latest["result"]
            st.success("机器人固定动作已完成。")
            st.caption(
                f"到位偏差 {float(result.get('tcp_translation_error_m', 0)) * 1000:.2f} mm · "
                f"方向偏差 {float(result.get('tcp_rotation_error_rad', 0)) * 57.2958:.2f}°"
            )
        elif latest.get("action") == "run_project_observation" and latest.get("result"):
            result = latest["result"]
            st.success("机器人已走到当前项目计算出的目标位置。")
            target = result.get("project_target_tcp") or {}
            position = target.get("position") or (0.0, 0.0, 0.0)
            st.caption(
                "项目目标位置 "
                f"x={float(position[0]):.3f} m · y={float(position[1]):.3f} m · "
                f"z={float(position[2]):.3f} m"
            )
        elif latest.get("action") == "capture_rgbd" and latest.get("result"):
            operation_id = str(latest.get("operation_id") or "")
            attached_ids = set(session.get("simulation_capture_operation_ids") or ())
            belongs_to_session = latest.get("case_id") == f"scene-{session_id[-6:]}"
            if belongs_to_session and operation_id and operation_id not in attached_ids:
                with st.spinner("正在把现场采集同步到当前会话……"):
                    attached = _post(
                        f"/api/v1/sessions/{session_id}/simulation-capture", {}
                    )
                if attached:
                    st.rerun()
            if belongs_to_session:
                st.success("相机采集已同步到当前会话。")
            else:
                st.caption("最近的采集属于另一诊断会话。")
            result = latest["result"]
            st.caption(
                f"有效距离像素 {float(result.get('depth_valid_ratio', 0)) * 100:.1f}% · "
                f"目标位置偏差 "
                f"{float(result.get('rgbd_translation_error_m', 0)) * 1000:.2f} mm"
            )


def main() -> None:
    st.set_page_config(page_title="VisionDoctor", page_icon="🩺", layout="wide")
    _apply_style()
    try:
        health = _api("/health")
        sessions = _api("/api/v1/sessions")
        projects = _api("/api/v1/projects")
    except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
        st.error(f"VisionDoctor 服务暂时不可用：{exc}")
        st.stop()
    selected_session = _render_sidebar(sessions, projects, health)
    if not health.get("model", {}).get("configured"):
        st.error("诊断助手尚未配置模型服务，请先完成服务配置。")
    _render_conversation(selected_session, health)


if __name__ == "__main__":
    main()
