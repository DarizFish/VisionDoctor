"""The diagnosis workbench.

A sidebar of cases grouped by the project they are about, a conversation in the
middle that opens up to show what each turn actually did, and on the right the
things the conversation is arguing over: the connected repository and the chain
a demarcation is written on.
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
        .block-container {padding-top: 2rem; max-width: 1600px;}
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
        /* The pinned input belongs to the conversation, not to the whole page:
           hold it over the left two thirds so it stops running under the panel. */
        [data-testid="stBottomBlockContainer"] {max-width: 1600px; padding-bottom: 1rem;}
        [data-testid="stBottomBlockContainer"] > div {width: 65%; min-width: 22rem;}
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
            state = "已定界" if item["gate"] else f"{item['evidence']} 条证据"
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
    if call["name"] == "check_transform_chain":
        return "重算了声明的变换链", f"用到 {listed}" if listed else ""
    if call["name"] == "list_source":
        return "列出了项目在该提交下的源码文件", ""
    if call["name"] == "read_source":
        return f"读取源码 {arguments.get('path', '')}", "读的是绑定提交下的那一份"
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
            f'提交 {html.escape(project["revision"][:12])}</span></div>',
            unsafe_allow_html=True,
        )
        return
    with st.container(border=True):
        st.markdown("**项目仓库**")
        st.caption("连接之后，只有诊断门通过了才读得到该提交下的源码。")
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
        if st.button("连接项目", use_container_width=True) and repository.strip():
            posted = _post(
                f"/api/v1/cases/{case_id}/project",
                {"repository": repository, "replay_command": command.split()},
            )
            if posted:
                st.rerun()


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


def _render_chain(view: dict[str, Any]) -> None:
    st.markdown("### 诊断链路")
    gate = view["gates"]["diagnosis"]
    st.caption(
        "诊断门已通过，源码对诊断助手可见"
        if view["gates"]["source_visible"]
        else "诊断门未通过：" + ("；".join(gate["reasons"]) or "还没有开始调查")
    )
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


def _render_hypotheses(view: dict[str, Any]) -> None:
    if not view["hypotheses"]:
        return
    st.markdown("### 当前假设")
    for item in view["hypotheses"]:
        with st.container(border=True):
            label = item.get("segment_name") or item["target_segment"]
            st.markdown(f"**{item['hypothesis_id']} · {label}**")
            st.markdown(item["statement"])
            st.caption("证据 " + "　".join(item["evidence_ids"]))


def _render_evidence(view: dict[str, Any]) -> None:
    if not view["evidence"]:
        return
    seen = sum(1 for item in view["evidence"] if item["examined"])
    with st.expander(f"证据清单 · {len(view['evidence'])} 条，其中 {seen} 条被真的读取过"):
        st.dataframe(
            [
                {
                    "证据": item["evidence_id"],
                    "来源": item["reference"],
                    "采集时间": item["captured_at"][:19],
                    "已读取": "✓" if item["examined"] else "",
                }
                for item in view["evidence"]
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
            else:
                st.warning(f"已退回 · {plan['approver']}")


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
        "site_recovered": "现场已恢复",
        "site_still_failing": "现场仍然失败",
        "not_a_recheck": "这不构成复核：证据不是在应用之后采集的",
    }[outcome["scope"]]
    (st.success if outcome["scope"] == "site_recovered" else st.warning)(label)
    st.caption(f"改动前 {outcome['before']}　改动后 {outcome['after']}")


# ---- page --------------------------------------------------------------------------


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
        _render_messages(case_id, view)
        if view.get("running"):
            _render_running(case_id)
    with side, st.container(height=720, border=False):
        _render_banner(view)
        _render_chain(view)
        _render_hypotheses(view)
        _render_plans(case_id, view)
        _render_recheck(case_id, view)
        _render_evidence(view)
        _render_connections(case_id, view)
    said = st.chat_input(
        "描述现象、回答问题，或告诉诊断助手接下来要检查什么……",
        accept_file="multiple",
        file_type=tuple(MEDIA_TYPES),
    )
    if said and (said.text or said.files):
        _submit(case_id, said.text.strip() or "先看看我交上来的材料。", list(said.files))


main()
