"""The diagnosis workbench.

A sidebar of cases grouped by the project they are about, a conversation in the
middle that opens up to show what each turn actually did, and on the right the
things the conversation is arguing over: what the case can reach, the graph a
demarcation is written on, and -- beneath a located software fault -- the source.
"""

from __future__ import annotations

import contextlib
import html
import json
import os
import urllib.error
import urllib.request
from typing import Any

import streamlit as st

STATUS_MARK = {"untested": "·", "cleared": "○", "suspect": "●"}
STATUS_LABEL = {"untested": "未测", "cleared": "已排除", "suspect": "可疑"}
LAYER_LABEL = {"observation": "现场观测", "software": "软件层", "source": "源码层"}
REMEDY_LABEL = {
    "configuration": "改运行配置", "handoff": "交开发方", "source_patch": "源码补丁",
    "physical": "现场处理", "more_evidence": "补充取证",
}


def _api(path: str, *, method: str = "GET", payload: dict | None = None) -> Any:
    base = os.getenv("VISIONDOCTOR_API_URL", "http://127.0.0.1:8000").rstrip("/")
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        base + path, data=body, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        with contextlib.suppress(json.JSONDecodeError):
            detail = json.loads(detail).get("detail", detail)
        raise RuntimeError(detail) from exc


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return _api(path, method="POST", payload=payload)
    except (RuntimeError, urllib.error.URLError) as exc:
        st.error(str(exc))
        return None


def _apply_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 0; max-width: 1600px;}
        /* Both columns end at the window's bottom, so the page itself never scrolls. */
        [data-testid="stMain"] {overflow: hidden;}
        [data-testid="stSidebar"] {border-right: 1px solid rgba(120,120,120,.16);}
        [data-testid="stChatMessage"] {border-radius: 16px; padding: .45rem .8rem;}
        .vd-kicker {display: block; min-height: 1.4rem; font-size: .78rem; line-height: 1.45;
                    letter-spacing: .08em; color: #6b7280; margin: 0 0 .2rem;}
        .vd-title {font-size: 2rem; font-weight: 720; line-height: 1.2; margin-bottom: .25rem;}
        .vd-subtle {color: #6b7280; font-size: .92rem;}
        .vd-source {padding: .8rem .9rem; border: 1px solid rgba(120,120,120,.18);
                    border-radius: 12px; margin-bottom: .55rem; overflow-wrap: anywhere;}
        .vd-source-ready {border-left: 4px solid #22a06b;}
        .vd-seg {padding: .5rem .7rem; border: 1px solid rgba(120,120,120,.18);
                 border-left-width: 3px; border-radius: 10px; margin-bottom: .35rem;}
        .vd-seg-suspect {border-left-color: #d94f70; background: rgba(217,79,112,.06);}
        .vd-seg-cleared {border-left-color: #22a06b;}
        .vd-seg-untested {border-left-color: rgba(120,120,120,.3);}
        .vd-scope {color: #6b7280; font-size: .78rem; line-height: 1.5;}
        /* Where the case stands stays put while the panel under it scrolls. */
        .vd-status {padding: .5rem .75rem; border: 1px solid rgba(120,120,120,.22);
                    border-radius: 12px; background: var(--background-color, #fff);
                    box-shadow: 0 2px 8px rgba(0,0,0,.06);}
        /* Streamlit wraps every element in its own div; the wrapper is what can stick,
           because only it has room to travel inside the panel's scroller. */
        [data-testid="stElementContainer"]:has(.vd-status)
            {position: sticky; top: 0; z-index: 101; padding-bottom: .4rem;
             background: var(--background-color, #fff);}
        /* The panel runs to the bottom of the window instead of stopping at a fixed
           height.  Its own element height stays as the fallback where :has() is
           unavailable; the wrapper carries it too, since the panel is a flex child. */
        [data-testid="stLayoutWrapper"]:has(
            > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] .vd-status)
            {flex: 0 0 calc(100vh - 52px) !important;}
        [data-testid="stLayoutWrapper"]:has(
            > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] .vd-status),
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vd-status)
            {height: calc(100vh - 52px) !important; min-height: 24rem;}
        .vd-access {display: flex; gap: .4rem; flex-wrap: wrap; margin: .2rem 0 .5rem;}
        .vd-chip {padding: .15rem .55rem; border-radius: 999px; font-size: .8rem;
                  border: 1px solid rgba(120,120,120,.25);}
        .vd-chip-on {border-color: #22a06b; color: #17784c; background: rgba(34,160,107,.08);}
        .vd-chip-off {color: #6b7280;}
        .vd-layer {padding: .55rem .75rem; border-radius: 10px; margin-bottom: .6rem;
                   border: 1px solid rgba(120,120,120,.18);}
        .vd-layer-open {border-left: 4px solid #4474b8; background: rgba(68,116,184,.06);}
        .vd-layer-closed {border-left: 4px solid rgba(120,120,120,.35);}
        /* The conversation scrolls in its own box, with its input right under it, so
           nothing spans the page over the panel.  The column is given the panel's
           height and the box takes what the title and the input leave, whatever
           height those turn out to be. */
        [data-testid="stColumn"]:has(.vd-chat) > [data-testid="stVerticalBlock"]
            {height: calc(100vh - 52px) !important;}
        [data-testid="stLayoutWrapper"]:has(
            > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] .vd-chat)
            {flex: 1 1 auto !important; min-height: 8rem !important;}
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vd-chat)
            {height: 100% !important; min-height: 0 !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---- sidebar -----------------------------------------------------------------------


def _render_sidebar(cases: list[dict[str, Any]]) -> str | None:
    with st.sidebar:
        st.markdown("## VisionDoctor")
        st.caption("机器视觉诊断助手")
        if st.button("＋ 新建诊断", type="primary", use_container_width=True):
            created = _post("/api/v1/cases", {"title": "新的诊断"})
            if created:
                st.session_state.selected_case = created["case_id"]
                st.rerun()
        st.markdown("#### 项目与诊断")
        selected = st.session_state.get("selected_case")
        available = [item["case_id"] for item in cases]
        if selected not in available:
            selected = available[0] if available else None

        def render_case(item: dict[str, Any]) -> None:
            marker = "●" if item["case_id"] == selected else "○"
            state = "源码层已开放" if item["gate"] else f"{item['evidence']} 条证据"
            if st.button(
                f"{marker} {item['title']}\n\n{state}",
                key=f"case-{item['case_id']}",
                use_container_width=True,
            ):
                st.session_state.selected_case = item["case_id"]
                st.rerun()

        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in cases:
            grouped.setdefault(str(item.get("project") or ""), []).append(item)
        for project in sorted(grouped, key=lambda name: name == ""):
            members = grouped[project]
            if project:
                with st.container(border=True):
                    st.markdown(f"**{project.replace(chr(92), '/').rsplit('/', maxsplit=1)[-1]}**")
                    st.caption(f"{len(members)} 个诊断会话")
                    for item in members:
                        render_case(item)
            else:
                st.caption("尚未连接项目")
                for item in members:
                    render_case(item)
        if not cases:
            st.caption("还没有诊断会话")
        st.session_state.selected_case = selected
        st.divider()
        try:
            health = _api("/health")
            st.caption(f"🟢 诊断服务已就绪 · {health['cases']} 个案件")
        except (RuntimeError, urllib.error.URLError) as exc:
            st.caption(f"🟠 诊断服务需要检查：{exc}")
    return selected


# ---- conversation ------------------------------------------------------------------


def _describe_call(call: dict[str, Any], names: dict[str, str]) -> tuple[str, str]:
    """Say what the host did, the way a person would say it."""

    headline, detail = _describe_intent(call, names)
    if call.get("failure"):
        # The success wording is past tense; a refused call never happened.
        return f"⚠️ 没能{headline.replace('了', '', 1)}", _reason(str(call["failure"]))
    return headline, detail


#: Raw exception text is for the model; a person gets the sentence behind it.
REASONS = (
    ("neither on disk nor in the index", "绑定的提交里没有这个文件"),
    ("does not exist in", "绑定的提交里没有这个文件"),
    ("Not a valid object name", "绑定的提交无效，源码读不出来"),
    ("is not listed in bundle.json", "观察包里没有这份材料"),
    ("is not in this case", "这个案件里没有这条证据"),
    ("never delivered", "引用了宿主没有交付过的证据"),
    ("not on the investigation surface", "这件工具当前不开放"),
)


def _reason(failure: str) -> str:
    for mark, said in REASONS:
        if mark in failure:
            return said
    return failure.split(": ", maxsplit=1)[-1]


def _describe_intent(call: dict[str, Any], names: dict[str, str]) -> tuple[str, str]:
    arguments = call.get("arguments") or {}
    listed = "、".join(
        f"{item} {names.get(item, '')}".strip() for item in (call.get("requested") or ())
    )
    if call["name"] == "read_evidence":
        return f"读取了 {listed or '若干证据'}", ""
    if call["name"] == "inspect_grasp_graph":
        return "核对抓取系统的依赖与交接", str(arguments.get("target_id") or "")
    if call["name"] == "read_domain_knowledge":
        return "参考相关领域的检查方法", "、".join(arguments.get("knowledge_ids") or [])
    if call["name"] == "check_transform_chain":
        return "重算了声明的变换链", f"用到 {listed}" if listed else ""
    if call["name"] == "measure_rgbd_region":
        return "测量目标区域的图像与深度", f"像素范围 {arguments.get('roi_xyxy')}；{listed}"
    if call["name"] == "check_capture_alignment":
        return "核对采集、检测与指令的对应关系", listed
    if call["name"] == "compare_runs":
        return (
            "按内容对比了两次运行的版本与软件层记录",
            f"{arguments.get('baseline_run_id', '')} → {arguments.get('run_id', '')}",
        )
    if call["name"] == "replay_running_version":
        return "用记录输入复跑了运行版本", "没有读取源码"
    if call["name"] == "list_source":
        return "列出了运行版本的源码文件", "源码层"
    if call["name"] == "read_source":
        return (
            f"读取源码 {arguments.get('path', '')}",
            f"源码层 · 为 {arguments.get('hypothesis_id', '')} 读取运行版本的那一份",
        )
    if call["name"] == "propose_repair":
        return (
            f"提交候选修复 {arguments.get('path', '')}",
            str(arguments.get("rationale") or ""),
        )
    return call["name"], listed


def _render_calls(calls: list[dict[str, Any]], names: dict[str, str]) -> None:
    if not calls:
        return
    with st.expander(f"这一轮做了什么 · {len(calls)} 次工具调用", expanded=False):
        for index, call in enumerate(calls):
            headline, detail = _describe_call(call, names)
            st.markdown(f"**{index + 1}.** {headline}")
            first = "、".join(call["delivered"])
            parts = [detail, f"本轮首次读到 {first}" if first else ""]
            note = "；".join(part for part in parts if part)
            if note:
                st.caption(note)
        st.caption("工具由宿主执行并记账。模型自称检查过而没有调用的，不能作为证据引用。")

@st.cache_data(show_spinner=False)
def _evidence_bytes(case_id: str, evidence_id: str) -> bytes | None:
    import base64

    try:
        payload = _api(f"/api/v1/cases/{case_id}/evidence/{evidence_id}")
    except (RuntimeError, urllib.error.URLError):
        return None
    return base64.b64decode(payload["content_base64"])


def _render_files(case_id: str, files: list[dict[str, Any]]) -> None:
    pictures = [item for item in files if str(item.get("media_type", "")).startswith("image/")]
    others = [item for item in files if item not in pictures]
    if pictures:
        columns = st.columns(min(len(pictures), 3))
        for column, item in zip(columns, pictures, strict=False):
            payload = _evidence_bytes(case_id, item["evidence_id"])
            if payload is not None:
                column.image(payload, caption=f"{item['name']} · {item['evidence_id']}")
    for item in others:
        st.caption(f"　📄 {item['name']} · {item['evidence_id']}")


@st.fragment(run_every="1.5s")
def _render_running(case_id: str) -> None:
    try:
        view = _api(f"/api/v1/cases/{case_id}")
    except (RuntimeError, urllib.error.URLError):
        return
    running = view.get("running")
    if not running:
        st.rerun()
        return
    names = {
        item["evidence_id"]: item["reference"].rsplit("/", maxsplit=1)[-1]
        for item in view.get("evidence") or ()
    }
    with st.chat_message("assistant"), st.status("正在查看你提供的材料……", expanded=True):
        st.caption("工具调用由宿主执行并记账。")
        for index, call in enumerate(running["calls"]):
            headline, detail = _describe_call(call, names)
            st.markdown(f"**{index + 1}.** {headline}")
            if detail:
                st.caption(detail)


def _render_messages(case_id: str, view: dict[str, Any]) -> None:
    names = {
        item["evidence_id"]: item["reference"].rsplit("/", maxsplit=1)[-1]
        for item in view.get("evidence") or ()
    }
    messages = view.get("messages") or []
    if not messages:
        st.info("先说说发生了什么。你不需要整理成表单，也不需要判断是哪次代码改动出了问题。")
        st.markdown(
            "例如：这两个件都抓偏了，同一套程序上周还是好的；把一次运行的观察证据包附上来，"
            "诊断助手会自己决定要看哪些材料。"
        )
        return
    for message in messages:
        if message["role"] == "source":
            st.caption(f"📥 {message['content']} · {message.get('detail', '')}")
            _render_files(case_id, message.get("files") or [])
            continue
        with st.chat_message(message["role"]):
            st.markdown(str(message.get("content", "")))
            if message["role"] != "assistant":
                continue
            if message.get("findings"):
                st.caption("本轮判定：" + "　".join(message["findings"]))
            for statement in message.get("hypotheses") or ():
                st.markdown(f"- {statement}")
            _render_calls(list(message.get("calls") or ()), names)


MEDIA_TYPES = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp",
    "txt": "text/plain", "log": "text/plain", "json": "application/json",
    "csv": "text/csv", "yaml": "text/yaml", "yml": "text/yaml", "pdf": "application/pdf",
}


def _encode(files: list[Any]) -> list[dict[str, str]]:
    import base64

    encoded = []
    for item in files:
        suffix = item.name.rsplit(".", maxsplit=1)[-1].lower()
        encoded.append(
            {
                "name": item.name,
                "media_type": MEDIA_TYPES.get(suffix, "application/octet-stream"),
                "content_base64": base64.b64encode(item.getvalue()).decode("ascii"),
            }
        )
    return encoded


def _submit(case_id: str, prompt: str, files: list[Any]) -> None:
    uploads = _encode(files)
    if uploads and not _post(f"/api/v1/cases/{case_id}/attachments", {"files": uploads}):
        return
    if _post(f"/api/v1/cases/{case_id}/turns", {"prompt": prompt}):
        st.rerun()


# ---- right column ------------------------------------------------------------------


def _render_connections(case_id: str, view: dict[str, Any]) -> None:
    observation = view.get("observation")
    label = "当前连接" if (observation or view.get("project")) else "连接项目仓库"
    expander = st.expander(label, expanded=not (observation or view.get("project")))
    with expander:
        _connections_body(case_id, view, observation)


def _connections_body(
    case_id: str, view: dict[str, Any], observation: dict[str, Any] | None
) -> None:
    if observation:
        results = "　".join(
            f"{item['part_id']}：{'成功' if item['success'] else item['classification']}"
            for item in observation["results"]
        )
        st.markdown(
            '<div class="vd-source vd-source-ready"><b>观察证据包</b><br>'
            f'<span class="vd-subtle">{html.escape(observation["run_id"])}<br>'
            f'{observation["artifacts"]} 件工件 · 代码 '
            f'{html.escape(str(observation["revision"].get("commit", ""))[:8])}<br>'
            f"{html.escape(results)}</span></div>",
            unsafe_allow_html=True,
        )
    else:
        with st.container(border=True):
            st.markdown("**观察证据包**")
            st.caption("一次运行导出的目录。单张图片和日志直接从输入框的 ＋ 交上来即可。")
            directory = st.text_input(
                "证据包所在目录",
                key="bundle-dir",
                placeholder=".runtime/gazebo-pick-cell/exports/run-…",
            )
            attach = st.button("接入证据包", use_container_width=True) and directory.strip()
            if attach and _post(
                f"/api/v1/cases/{case_id}/observations", {"directory": directory}
            ):
                st.rerun()
    project = view.get("project")
    if project:
        st.markdown(
            '<div class="vd-source vd-source-ready"><b>项目仓库</b><br>'
            f'<span class="vd-subtle">{html.escape(project["repository"])}<br>'
            f'提交 {html.escape(project["revision"][:12])} · '
            f'{"可读源码" if project.get("source_readable") else "只有可运行程序，不读源码"}'
            "</span></div>",
            unsafe_allow_html=True,
        )
        return
    with st.container(border=True):
        st.markdown("**项目仓库**")
        st.caption("程序可按记录输入复跑；源码只在软件层定位之后、按运行版本开放。")
        repository = st.text_input("仓库所在文件夹", key="repo-path")
        commit = (observation or {}).get("revision", {}).get("commit", "")
        st.caption(
            f"提交由观察包指定：`{commit[:12]}`　源码按这个提交读取"
            if commit
            else "观察包没有指定提交，将按仓库当前 HEAD 读取"
        )
        command = st.text_input(
            "项目自己的复跑命令",
            value="python -m pick_demo.replay --input {input} --output {output} --log {log}",
            key="repo-cmd",
        )
        readable = st.checkbox(
            "允许读取源码", value=True, key="repo-source",
            help="取消时按现场只有发布产物处理：可以复跑，诊断停在软件层。",
        )
        if st.button("连接项目", use_container_width=True) and repository.strip():
            posted = _post(
                f"/api/v1/cases/{case_id}/project",
                {"repository": repository, "replay_command": command.split(),
                 "source_readable": readable},
            )
            if posted:
                st.rerun()


RECHECK_SCOPE = {
    "site_recovered": "现场已恢复",
    "site_still_failing": "现场仍未通过",
    "not_a_recheck": "复核不成立：采集时刻或任务范围不匹配",
}


def _headline(view: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """What holds right now, the host's arithmetic first and the model's claim only after."""

    recheck = view.get("recheck")
    if recheck:
        return RECHECK_SCOPE.get(recheck["scope"], recheck["scope"])
    if rows:
        labels = sorted({CONCLUSION_LABEL[row["isolation"]["conclusion"]] for row in rows})
        return "核算：" + "；".join(labels) + f"　剩余候选 {_remaining(view) or '未核算'}"
    suspects = [node["name"] for node in (view.get("graph") or {}).get("nodes", [])
                if node.get("status") == "suspect"]
    if suspects:
        return "模型可疑：" + "、".join(suspects) + "（未经核算）"
    return "尚无结论"


def _stage(view: dict[str, Any]) -> tuple[str, str]:
    """The stage this case is in, and whose move it is next."""

    rows = [row for row in (view.get("isolation") or []) if row.get("diagnosed_run")]
    conclusions = {row["isolation"]["conclusion"] for row in rows}
    plans = view.get("plans") or []
    if view.get("recheck"):
        return "已复核", "复核记录已经入案；现场若仍未通过，回到会话继续诊断。"
    if view.get("applied_at"):
        return "待现场复核", "在『宿主核算』页填入应用之后新采集的证据包目录，重算同一组检验。"
    if any(plan["approved"] for plan in plans):
        return "待应用", "把批准的改动应用到现场，再回来做现场复核。"
    if any(plan["approved"] is None for plan in plans):
        return "待审批", "在『模型主张』页审阅补丁 diff，批准或退回。"
    if "isolated" in conclusions:
        if ((view.get("gates") or {}).get("source_layer") or {}).get("passed"):
            return "已定位 · 源码层已开放", "让模型在源码层解释机制并提出最小补丁。"
        return "已定位 · 停在软件层", "出口是改配置或交开发方；接入源码之后才能解释代码层机制。"
    if "ambiguous" in conclusions:
        return "候选不止一个", "按『宿主核算』页的区分建议补测量，再核算一次。"
    if rows:
        return "核算未发现违反", "通过不等于正常：看未检验与不可检测的节点，换检验或补测量。"
    if view.get("observation"):
        return "调查中", "在会话里推进；几何、帧对应或深度问题可以先做结构核算。"
    return "取证", "在『宿主核算』页的当前连接里接入观察证据包。"


def _render_status(view: dict[str, Any]) -> None:
    """Where the case stands, so the panel below has somewhere to start."""

    stage, step = _stage(view)
    rows = [row for row in (view.get("isolation") or []) if row.get("diagnosed_run")]
    st.markdown(
        f'<div class="vd-status"><b>{html.escape(stage)}</b>　'
        f'{html.escape(_headline(view, rows))}'
        f'<div class="vd-scope">下一步：{html.escape(step)}</div></div>',
        unsafe_allow_html=True,
    )


def _render_banner(view: dict[str, Any]) -> None:
    observation = view.get("observation")
    if not observation:
        st.caption("尚未接入观察证据包，诊断助手只能看你直接交上来的材料。")
        return
    results = "　".join(
        f"{item['part_id']} {'成功' if item['success'] else '失败'}"
        for item in observation["results"]
    )
    st.caption(
        f"观察 {observation['run_id'][:26]} · {observation['artifacts']} 件工件"
        + (f" · {results}" if results else "")
    )


def _render_access(view: dict[str, Any]) -> None:
    """What this case can reach, and whether the source layer is open beneath it."""

    access = view.get("access") or {}
    chips = (("运行证据", access.get("observation")), ("可复跑", access.get("runnable")),
             ("源码", access.get("source_readable")))
    st.markdown(
        '<div class="vd-access">'
        + "".join(
            f'<span class="vd-chip vd-chip-{"on" if on else "off"}">'
            f'{"✓" if on else "✗"} {label}</span>'
            for label, on in chips
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    gate = (view.get("gates") or {}).get("source_layer") or {}
    if gate.get("passed"):
        basis = "；".join(
            f'{item["target_id"]}（{"、".join(item["evidence_ids"])}）'
            for item in gate.get("basis", [])
        )
        body = f"<b>源码层已开放</b><br>依据软件层定位：{html.escape(basis)}"
        style = "open"
    else:
        body = "<b>诊断停在软件层</b><br>" + html.escape("；".join(gate.get("reasons", [])))
        style = "closed"
    st.markdown(
        f'<div class="vd-layer vd-layer-{style}"><span class="vd-scope">{body}</span></div>',
        unsafe_allow_html=True,
    )


def _render_graph(view: dict[str, Any]) -> None:
    """The map both columns argue on: the model draws the borders, the host fills them."""

    graph = view.get("graph")
    if not graph:
        return
    from visiondoctor.web.graph import graph_dot

    # Foldable: it is the tallest thing here, and comparing the two columns beneath it
    # is easier with the map out of the way.
    with st.expander("抓取系统图", expanded=True):
        st.caption("节点和连线分别核查；边框颜色是模型判定，底色是宿主核算，都不扩展到邻居。"
                   "双线框是承载软件的节点，源码层只能在其下展开。")
        st.graphviz_chart(
            graph_dot(graph, view.get("node_standing")), use_container_width=True
        )
        st.caption("可参考的领域知识："
                   + "、".join(item["title"] for item in view.get("knowledge_catalogue", []))
                   + "。参考结构和领域知识帮助选择检查；未采集的接触、保持等信号仍为未查。")


def _render_segments(view: dict[str, Any]) -> None:
    """The model's own verdicts: what it says it checked, cleared or still suspects."""

    graph = view.get("graph")
    if graph:
        with st.expander("查看检查依据与未决项"):
            for item in [*graph["nodes"], *graph["edges"]]:
                finding = item.get("finding")
                if finding is None:
                    continue
                st.markdown(f"**{item['name']} · {STATUS_LABEL[item['status']]}**")
                st.write(finding["note"])
                if finding.get("checked_scope"):
                    st.caption("检查范围：" + finding["checked_scope"])
                if finding.get("limitations"):
                    st.caption("仍未排除：" + finding["limitations"])
                layers = "、".join(
                    LAYER_LABEL.get(name, name) for name in finding.get("layers", [])
                )
                st.caption("证据：" + "、".join(finding["evidence_ids"])
                           + (f"　依据：{layers}" if layers else ""))
    st.markdown("#### 分组概览")
    st.caption("各组的排除只限于已检查对象与工况。")
    for row in view["chain"]:
        note = row["note"] or row["scope"]
        evidence = "　".join(row["evidence_ids"])
        st.markdown(
            f'<div class="vd-seg vd-seg-{row["status"]}">'
            f'<b>{STATUS_MARK[row["status"]]} {row.get("name") or row["segment"]}</b> '
            f'<span class="vd-scope">{STATUS_LABEL[row["status"]]}</span>'
            f'<div class="vd-scope">{html.escape(note)}</div>'
            + (f'<div class="vd-scope">证据 {html.escape(evidence)}</div>' if evidence else "")
            + "</div>",
            unsafe_allow_html=True,
        )


CONCLUSION_LABEL = {
    "isolated": "已隔离到唯一候选",
    "ambiguous": "剩余候选不止一个",
    "no_fault_detected": "可用检验全部通过",
    "outside_model": "没有候选能解释全部违反：单故障假设不成立，或原因在模板之外",
    "nothing_evaluated": "没有可运行的检验",
}
TEST_STATUS = {"pass": "通过", "fail": "违反", "unevaluated": "未核算"}


def _node_names(view: dict[str, Any]) -> dict[str, str]:
    return {node["id"]: node["name"] for node in (view.get("graph") or {}).get("nodes", [])}


def _ratio_text(reading: dict[str, Any], gap: float) -> str:
    """A valid-depth ratio says most when the two sides it compares are both shown."""

    raw = reading.get("raw_valid_ratio")
    for key, label in (("consumed_valid_ratio", "消费"), ("reference_valid_ratio", "参考")):
        other = reading.get(key)
        if raw is not None and other is not None:
            return f"有效比例 原始 {raw:.2f} · {label} {other:.2f}"
    return f"有效比例差 {gap:.2f}"


def _rule_text(reading: dict[str, Any], mismatch: float) -> str:
    said = "与声明规则不符" if mismatch else "按声明规则执行"
    ratio, minimum = reading.get("consumed_valid_ratio"), reading.get("declared_minimum")
    if ratio is None or minimum is None:
        return said
    return f"{said}：比例 {ratio:.2f} · 下限 {minimum:.2f}"


def _residual_text(test: dict[str, Any]) -> str:
    """What the check actually read, in units someone can argue with."""

    residual = test.get("residual") or {}
    reading = test.get("reading") or {}
    parts = []
    for key, value in residual.items():
        if key == "position_m":
            parts.append(f"位置 {value * 1000:.1f} mm")
        elif key == "rotation_rad":
            parts.append(f"姿态 {value:.3f} rad")
        elif key == "identity_mismatch":
            parts.append("采集标识不同" if value else "采集标识一致")
        elif key == "age_s":
            parts.append(f"时差 {value:.2f} s")
        elif key == "ratio_gap":
            parts.append(_ratio_text(reading, value))
        elif key == "rule_mismatch":
            parts.append(_rule_text(reading, value))
        else:  # a later evaluator's reading still shows up, rather than an empty cell
            parts.append(f"{key} {value:.3f}")
    return " · ".join(parts)


def _remaining(view: dict[str, Any]) -> str | None:
    """The host's remaining candidates for the run under diagnosis, for exit cards."""

    standing = view.get("node_standing") or {}
    if not standing:
        return None
    names = _node_names(view)
    left = [names.get(node, node) for node, state in standing.items() if state == "candidate"]
    return "、".join(left) or "无（可用检验未发现违反）"


def _render_isolation(view: dict[str, Any]) -> None:
    rows = view.get("isolation") or []
    if not rows:
        return
    names = _node_names(view)

    st.markdown("### 核算隔离")
    st.caption("宿主计算。结构分析决定哪些故障可检测、可区分、哪些检验可用；"
               "检验数值来自预先实现的残差计算器。单故障假设，未检验不等于正常。")
    for row in rows:
        isolation = row["isolation"]
        label = "本次运行" if row["diagnosed_run"] else (
            "复核运行" if row["run_id"] == (view.get("recheck") or {}).get("after_run_id")
            else "参考运行"
        )
        # Results written before fault modes carry node ids only; names fall back to the graph.
        faults = {key: value["name"]
                  for key, value in row["structure"].get("faults", {}).items()}

        def named(ids: list[str], faults: dict[str, str] = faults) -> str:
            return "、".join(faults.get(item) or names.get(item, item) for item in ids) or "无"

        with st.container(border=True):
            conclusion = CONCLUSION_LABEL[isolation["conclusion"]]
            if isolation.get("undetectable"):
                conclusion += f"；{len(isolation['undetectable'])} 个节点当前不可检测"
            st.markdown(f"**{label} · 工件 {row['part_id']}**　{conclusion}")
            st.dataframe(
                [
                    {
                        "检验": f"{test['test']} {test['name']}",
                        "结果": TEST_STATUS[test["status"]],
                        "残差": _residual_text(test),
                        "敏感节点": named(test["sensitive_to"]),
                        "说明": test.get("reason") or test["threshold_source"],
                    }
                    for test in row["tests"]
                ],
                hide_index=True, use_container_width=True,
            )
            st.markdown(f"剩余候选：**{named(isolation['candidates'])}**")
            st.caption("核算排除：" + named(isolation["exonerated"]))
            if isolation["unexamined"]:
                st.caption("未检验：" + named(isolation["unexamined"]))
            if isolation.get("undetectable"):
                st.caption("当前测量下不可检测（通过的检验对它们无话可说）："
                           + named(isolation["undetectable"]))
            for item in row["suggestions"]:
                st.write(f"区分建议：绑定 {'、'.join(item['bind']) or '已有记录'}，运行检验 "
                         f"{item['test']} {item['name']}，可把 {named(item['separates'])} "
                         f"与 {named(item['from'])} 分开")
            for group in row["gaps"]["missing_measurement"]:
                st.caption(f"缺测量：{named(group)} 用现有记录在结构上分不开")
            for pair in row["gaps"]["missing_test"]:
                st.caption(f"缺检验：{named(pair)} 结构上可分，但没有现成检验")
            with st.expander("结构分析：关联矩阵与可隔离性"):
                structure = row["structure"]
                st.caption(
                    f"冗余度 {structure['redundancy']} · 过约束方程 "
                    + "、".join(structure["overdetermined_equations"])
                    + " · 已知变量 " + "、".join(structure["known_variables"])
                )
                st.dataframe(
                    [{"方程": item["equation"], "关系": item["relation"],
                      "未知变量": "、".join(item["unknowns"]) or "（全部已测）"}
                     for item in structure["incidence"]],
                    hide_index=True, use_container_width=True,
                )
                detectable = structure["detectable"]
                st.caption("可隔离性：行能否与列区分（✓ 可以，✗ 不行）")
                st.dataframe(
                    [
                        {"故障": faults.get(node, node), **{
                            faults.get(other, other): (
                                "—" if other == node else
                                "✗" if other in structure["cannot_isolate_from"][node] else "✓"
                            )
                            for other in detectable
                        }}
                        for node in detectable
                    ],
                    hide_index=True, use_container_width=True,
                )
                if structure["undetectable"]:
                    st.caption("当前测量下不可检测：" + named(structure["undetectable"]))
    host_label = {"candidate": "剩余候选", "exonerated": "核算排除", "unexamined": "未检验"}
    for item in view.get("divergence") or []:
        st.warning(
            f"{names.get(item['target_id'], item['target_id'])}：模型判为"
            f"{STATUS_LABEL[item['model']]}，核算为{host_label[item['host']]}"
        )


def _render_hypotheses(view: dict[str, Any]) -> None:
    if not view["hypotheses"]:
        return
    st.markdown("### 当前假设")
    for item in view["hypotheses"]:
        with st.container(border=True):
            label = item.get("target_id") or item.get("segment_name") or item["target_segment"]
            remedy = REMEDY_LABEL.get(item.get("remedy"), item.get("remedy") or "")
            st.markdown(f"**{item['hypothesis_id']} · {label} · {remedy}**")
            st.markdown(item["statement"])
            st.caption("证据 " + "　".join(item["evidence_ids"]))
            if item.get("remedy") == "handoff":
                _render_handoff(item, view)
            if item.get("prediction"):
                st.write("预期现象：" + item["prediction"])
            if item.get("counter_evidence_ids"):
                st.caption("反证：" + "、".join(item["counter_evidence_ids"]))
            if item.get("next_check"):
                st.write("下一项鉴别：" + item["next_check"])


def _render_handoff(hypothesis: dict[str, Any], view: dict[str, Any]) -> None:
    """A located software fault handed to whoever owns the program, from the case itself."""

    evidence = {item["evidence_id"]: item for item in view["evidence"]}
    cited = [evidence[name] for name in hypothesis["evidence_ids"] if name in evidence]
    inputs = [item for item in view["evidence"]
              if item.get("layer") == "software" and item["reference"].endswith("input.json")
              and item.get("bundle_id") == (view.get("observation") or {}).get("run_id")]
    replays = [item for item in view["evidence"]
               if item["reference"] == "derived/replay-running-version"]
    lines = [
        f"定位对象：{hypothesis.get('target_id') or hypothesis['target_segment']}",
        "依据：" + "；".join(f"{item['evidence_id']} {item['reference']}" for item in cited),
        "复现输入：" + ("、".join(f"{item['evidence_id']} {item['reference']}" for item in inputs)
                     or "未登记"),
        "运行版本复跑：" + ("、".join(item["evidence_id"] for item in replays) or "未执行"),
        "核算剩余候选：" + (_remaining(view) or "未核算"),
    ]
    if hypothesis.get("prediction"):
        lines.append("违反的约定 / 预期现象：" + hypothesis["prediction"])
    with st.container(border=True):
        st.markdown("**交接说明**")
        for line in lines:
            st.caption(line)


def _render_evidence(view: dict[str, Any]) -> None:
    if not view["evidence"]:
        return
    seen = sum(1 for item in view["evidence"] if item["examined"])
    with st.expander(f"证据清单 · {len(view['evidence'])} 条，其中 {seen} 条被真的读取过"):
        order = {"software": 0, "observation": 1, "source": 2}
        st.dataframe(
            [
                {
                    "证据": item["evidence_id"],
                    "层": LAYER_LABEL.get(item.get("layer"), item.get("layer") or ""),
                    "运行": item.get("bundle_id", ""),
                    "来源": item["reference"],
                    "采集时间": item["captured_at"][:19],
                    "任务阶段": {
                        "before_command": "命令之前", "earlier_capture": "较早的采集",
                        "command_execution": "命令执行期间",
                    }.get(item.get("phase"), item.get("phase") or "未注明"),
                    "已读取": "✓" if item["examined"] else "",
                }
                for item in sorted(view["evidence"],
                                   key=lambda row: order.get(row.get("layer"), 1))
            ],
            use_container_width=True,
            hide_index=True,
            height=260,
        )


def _render_plans(case_id: str, view: dict[str, Any]) -> None:
    if not view["plans"]:
        return
    st.markdown("### 候选修复")
    for plan in view["plans"]:
        with st.container(border=True):
            st.markdown(f"**{plan['plan_id']} · {plan['target_segment']}**")
            st.caption(f"冻结 {plan['frozen_hash'][:24]}　批准之后再改动即失效")
            st.caption("核算剩余候选：" + (_remaining(view) or "未核算")
                       + "。候选不止一个时，由审批人判断补丁是否充分")
            st.code(plan["diff"] or "（无改动）", language="diff")
            if plan["approved"] is None:
                approver = st.text_input("审批人", key=f"who-{plan['plan_id']}")
                columns = st.columns(2)
                approve = columns[0].button(
                    "批准", key=f"ok-{plan['plan_id']}", type="primary", use_container_width=True
                )
                reject = columns[1].button(
                    "退回", key=f"no-{plan['plan_id']}", use_container_width=True
                )
                if (approve or reject) and not approver:
                    st.warning("请先填写审批人。")
                elif approve or reject:
                    posted = _post(
                        f"/api/v1/cases/{case_id}/approvals",
                        {
                            "plan_id": plan["plan_id"],
                            "approver": approver,
                            "approved": bool(approve),
                        },
                    )
                    if posted:
                        st.rerun()
            elif plan["approved"]:
                st.success(f"已批准 · {plan['approver']}")
                _render_landing(plan.get("landed"))
            else:
                st.warning(f"已退回 · {plan['approver']}")


def _render_landing(landed: dict[str, Any] | None) -> None:
    """Where the approved patch went, and what is still someone's to do."""

    if not landed:
        return
    if landed.get("error"):
        st.error(f"没能写入仓库：{landed['error']}")
        return
    st.markdown(
        f"补丁已提交到 `{landed['branch']}` · `{landed['commit'][:12]}`　"
        + "、".join(f"`{name}`" for name in landed["files"])
    )
    st.caption("改动已经在仓库的文件里了，没有推到远端。部署到工位仍然由你决定。")


def _parts_text(results: Any) -> str:
    """Per part, whether the cell reached the pick pose -- the scope this cell is built to."""

    if not isinstance(results, dict):
        return str(results)
    return " · ".join(
        f"{part} {'到位' if ok else '未到位'}" for part, ok in sorted(results.items())
    )


def _render_recheck(case_id: str, view: dict[str, Any]) -> None:
    if not any(plan["approved"] for plan in view["plans"]):
        return
    st.markdown("### 现场复核")
    st.caption("隔离复现只能证明技术回归。现场恢复只能由改动应用之后新采集的证据授予。")
    if not view["applied_at"]:
        if st.button("已按批准的方案应用到现场", use_container_width=True):
            _post(f"/api/v1/cases/{case_id}/applied", {})
            st.rerun()
        return
    st.caption(f"应用于 {view['applied_at'][:19]}")
    directory = st.text_input("应用后新采集的证据包目录", key="recheck-dir")
    review = st.button("复核", use_container_width=True) and directory.strip()
    if review and _post(f"/api/v1/cases/{case_id}/recheck", {"directory": directory}):
        st.rerun()
    outcome = view.get("recheck")
    if not outcome:
        return
    label = {
        "site_recovered": "本次任务结果通过复核",
        "site_still_failing": "本次任务结果仍未通过",
        "not_a_recheck": "这不构成复核：采集时刻或任务范围不匹配",
    }[outcome["scope"]]
    (st.success if outcome["scope"] == "site_recovered" else st.warning)(label)
    st.caption(f"改动前 {_parts_text(outcome['before'])}　改动后 {_parts_text(outcome['after'])}")
    rows = {(row["run_id"], row["part_id"], test["test"]): test
            for row in view.get("isolation") or [] for test in row["tests"]}
    compared = [
        {"工件": part, "检验": f"{test['test']} {test['name']}",
         "改动前": f"{TEST_STATUS[test['status']]} {_residual_text(test)}",
         "改动后": f"{TEST_STATUS[after['status']]} {_residual_text(after)}"}
        for (run, part, name), test in sorted(rows.items())
        if run == outcome["before_run_id"]
        and (after := rows.get((outcome["after_run_id"], part, name)))
    ]
    if compared:
        st.caption("同一组检验在复核运行的同名记录上重算：")
        st.dataframe(compared, hide_index=True, use_container_width=True)


# ---- page --------------------------------------------------------------------------


def _scroll_conversation_to_latest(signature: str) -> None:
    """A conversation opens on its newest turn, the way the page-wide input used to.

    Streamlit keeps an unchanged component mounted across reruns, so the signature
    (which case, how many turns) has to change for this to run again.
    """

    import streamlit.components.v1 as components

    components.html(
        f"""<!-- {html.escape(signature)} -->
        <script>
        // The box is still filling when this mounts, so keep pinning for a moment, and
        // stop as soon as the reader scrolls somewhere themselves.
        const find = () => window.parent.document.querySelector(
            '[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vd-chat)');
        let last = null;
        const pin = () => {{
            const box = find();
            if (!box) return false;
            if (last !== null && Math.abs(box.scrollTop - last) > 4) return true;
            box.scrollTop = box.scrollHeight;
            last = box.scrollTop;
            return false;
        }};
        const until = Date.now() + 6000;
        const timer = setInterval(() => {{
            if (pin() || Date.now() > until) clearInterval(timer);
        }}, 150);
        </script>""",
        height=0,
    )


def main() -> None:
    st.set_page_config(page_title="VisionDoctor", page_icon="🩺", layout="wide")
    _apply_style()
    try:
        cases = _api("/api/v1/cases")
    except (RuntimeError, urllib.error.URLError) as exc:
        st.error(f"VisionDoctor 服务暂时不可用：{exc}")
        st.stop()
    case_id = _render_sidebar(cases)
    if case_id is None:
        st.markdown('<div class="vd-kicker">新的诊断</div>', unsafe_allow_html=True)
        st.markdown('<div class="vd-title">今天遇到了什么问题？</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="vd-subtle">左侧新建一个诊断开始。观察证据包和项目仓库都可以稍后补充。'
            "</div>",
            unsafe_allow_html=True,
        )
        return
    view = _api(f"/api/v1/cases/{case_id}")
    conversation, side = st.columns([2, 1], gap="large")
    with conversation:
        st.markdown('<div class="vd-kicker">诊断会话</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="vd-title">{html.escape(view["title"])}</div>', unsafe_allow_html=True
        )
        st.markdown(
            '<div class="vd-subtle">一条消息推进一步。每一步做了什么，展开就能核。</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        # The conversation scrolls in its own box and the input sits under it, so the
        # input belongs to the conversation instead of spanning the page over the panel.
        with st.container(height=600, border=False):
            st.markdown('<div class="vd-chat"></div>', unsafe_allow_html=True)
            _render_messages(case_id, view)
            if view.get("running"):
                _render_running(case_id)
        _scroll_conversation_to_latest(
            f"{case_id}:{len(view.get('messages') or [])}:{view.get('running')}"
        )
        said = st.chat_input(
            "描述现象、回答问题，或告诉诊断助手接下来要检查什么……",
            accept_file="multiple",
            file_type=tuple(MEDIA_TYPES),
        )
    with side, st.container(height=700, border=False):
        # Pinned to the top of the panel's own scroller, so it survives a scroll.
        _render_status(view)
        _render_banner(view)
        _render_access(view)
        # The graph is the map both sides argue on; below it they are kept apart, because
        # what the host computed and what the model claims are not read the same way.
        _render_graph(view)
        host, claims = st.tabs(["宿主核算", "模型主张"])
        with host:
            st.caption("这一栏由宿主计算或记录，不经模型。")
            _render_isolation(view)
            _render_recheck(case_id, view)
            _render_evidence(view)
            _render_connections(case_id, view)
        with claims:
            st.caption("这一栏是模型的判断与提案；可信度取决于它引用的证据与核算。")
            _render_segments(view)
            _render_hypotheses(view)
            _render_plans(case_id, view)
    if said and (said.text or said.files):
        _submit(case_id, said.text.strip() or "先看看我交上来的材料。", list(said.files))


main()
