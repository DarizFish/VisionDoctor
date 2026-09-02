"""The case console.

One page, in the order the work happens: what was observed, where it went wrong
along the chain, what the evidence says, what change is proposed, and what the
cell did after that change was applied.
"""

from __future__ import annotations

import contextlib
import json
import os
import urllib.error
import urllib.request
from typing import Any

import streamlit as st

STATUS_MARK = {"untested": "·", "cleared": "○", "suspect": "●"}


def _api(path: str, *, method: str = "GET", payload: dict | None = None) -> Any:
    base = os.getenv("VISIONDOCTOR_API_URL", "http://127.0.0.1:8000").rstrip("/")
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        base + path, data=body, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        with contextlib.suppress(json.JSONDecodeError):
            detail = json.loads(detail).get("detail", detail)
        raise RuntimeError(detail) from exc


def _style() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2.2rem;}
        .vd-title {font-size: 1.75rem; font-weight: 650; letter-spacing: -.01em;}
        .vd-sub {color: #6b7280; font-size: .9rem;}
        .vd-seg {padding: .45rem .7rem; border-radius: .5rem; margin-bottom: .3rem;
                 border: 1px solid #e5e7eb;}
        .vd-suspect {border-color: #f0a; background: rgba(255,0,170,.06);}
        .vd-cleared {background: rgba(16,185,129,.06);}
        .vd-scope {color: #6b7280; font-size: .78rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar() -> str | None:
    st.sidebar.markdown("### 诊断案件")
    try:
        cases = _api("/api/v1/cases")
    except (RuntimeError, urllib.error.URLError) as exc:
        st.sidebar.error(f"服务不可用：{exc}")
        st.stop()
    labels = {item["case_id"]: f"{item['case_id']} · {item['title']}" for item in cases}
    selected = st.session_state.get("case_id")
    if labels:
        keys = list(labels)
        index = keys.index(selected) if selected in keys else 0
        selected = st.sidebar.radio("已有案件", keys, index=index, format_func=labels.get)
    with st.sidebar.form("new-case"):
        title = st.text_input("新建案件", placeholder="例如：A17 工位抓偏")
        if st.form_submit_button("建案", use_container_width=True) and title.strip():
            selected = _api("/api/v1/cases", method="POST", payload={"title": title})["case_id"]
    st.session_state["case_id"] = selected
    return selected


def _inputs(case_id: str, view: dict[str, Any]) -> None:
    left, right = st.columns(2)
    with left, st.form("observation"):
        st.markdown("**接入观察证据包**")
        directory = st.text_input(
            "证据包目录", value=st.session_state.get("bundle_dir", ""), label_visibility="collapsed"
        )
        if st.form_submit_button("读取", use_container_width=True) and directory.strip():
            st.session_state["bundle_dir"] = directory
            result = _api(
                f"/api/v1/cases/{case_id}/observations",
                method="POST",
                payload={"directory": directory},
            )
            st.success(f"{result['run_id']} · 收入 {result['admitted']} 条证据")
            st.rerun()
    with right, st.form("project"):
        st.markdown("**绑定项目仓库**")
        repository = st.text_input("仓库路径", label_visibility="collapsed")
        seen = (view.get("observation") or {}).get("revision", {})
        revision = st.text_input("提交", value=seen.get("commit", ""))
        command = st.text_input(
            "复跑命令",
            value="python -m pick_demo.replay --input {input} --output {output} --log {log}",
        )
        if st.form_submit_button("绑定", use_container_width=True) and repository.strip():
            _api(
                f"/api/v1/cases/{case_id}/project",
                method="POST",
                payload={
                    "repository": repository,
                    "revision": revision,
                    "replay_command": command.split(),
                },
            )
            st.rerun()


def _observation(view: dict[str, Any]) -> None:
    observation = view.get("observation")
    if not observation:
        st.info("先接入一次观察证据包。产品只从证据包认识现场，不直接连相机或机器人。")
        return
    st.markdown(
        f'<span class="vd-sub">{observation["run_id"]} · 采集于 {observation["collected_at"][:19]}'
        f' · {observation["artifacts"]} 件工件'
        f' · 代码 {observation["revision"].get("commit", "")[:8]}'
        "</span>",
        unsafe_allow_html=True,
    )
    columns = st.columns(max(len(observation["results"]), 1))
    for column, result in zip(columns, observation["results"], strict=False):
        mark = "成功" if result["success"] else "失败"
        column.metric(f"任务 {result['part_id']}", mark, result["classification"])


def _chain(view: dict[str, Any]) -> None:
    st.markdown("#### 诊断链路")
    for row in view["chain"]:
        css = "vd-seg"
        if row["status"] == "suspect":
            css += " vd-suspect"
        elif row["status"] == "cleared":
            css += " vd-cleared"
        note = row["note"] or row["scope"]
        evidence = "　".join(row["evidence_ids"])
        st.markdown(
            f'<div class="{css}"><b>{STATUS_MARK[row["status"]]} {row["segment"]}</b>'
            f'<div class="vd-scope">{note}</div>'
            + (f'<div class="vd-scope">证据 {evidence}</div>' if evidence else "")
            + "</div>",
            unsafe_allow_html=True,
        )


def _turns(case_id: str, view: dict[str, Any]) -> None:
    st.markdown("#### 推进")
    gate = view["gates"]["diagnosis"]
    st.caption(
        "诊断门已通过，源码工具可用" if view["gates"]["source_visible"]
        else "诊断门未通过：" + ("；".join(gate["reasons"]) or "还没有开始调查")
    )
    for turn in view["turns"]:
        with st.expander(f"{turn['turn_id']} · {turn['prompt'][:40]}", expanded=False):
            for call in turn["calls"]:
                st.markdown(
                    f"`{call['name']}` → 交付 {', '.join(call['delivered']) or '（无）'}"
                )
            st.caption(turn["next_step"])
    with st.form("turn"):
        prompt = st.text_area(
            "说一句话推进一步", placeholder="例如：这两个件都抓偏了，帮我定位是哪一段的问题",
            label_visibility="collapsed", height=80,
        )
        if st.form_submit_button("推进一步", type="primary") and prompt.strip():
            with st.spinner("调查中……工具调用由宿主执行并记账"):
                try:
                    _api(f"/api/v1/cases/{case_id}/turns", method="POST",
                         payload={"prompt": prompt})
                except RuntimeError as exc:
                    st.error(str(exc))
                    return
            st.rerun()


def _hypotheses(view: dict[str, Any]) -> None:
    if not view["hypotheses"]:
        return
    st.markdown("#### 当前假设")
    for item in view["hypotheses"]:
        st.markdown(f"**{item['hypothesis_id']} @ {item['target_segment']}** — {item['statement']}")
        st.caption("证据 " + "　".join(item["evidence_ids"]))


def _plans(case_id: str, view: dict[str, Any]) -> None:
    if not view["plans"]:
        return
    st.markdown("#### 候选修复")
    for plan in view["plans"]:
        st.markdown(
            f"**{plan['plan_id']}** @ {plan['target_segment']} · 冻结 `{plan['frozen_hash'][:16]}`"
        )
        st.code(plan["diff"] or "（无改动）", language="diff")
        if plan["approved"] is None:
            columns = st.columns(3)
            approver = columns[0].text_input("审批人", key=f"who-{plan['plan_id']}")
            if columns[1].button("批准", key=f"ok-{plan['plan_id']}", type="primary") and approver:
                _api(f"/api/v1/cases/{case_id}/approvals", method="POST",
                     payload={"plan_id": plan["plan_id"], "approver": approver, "approved": True})
                st.rerun()
            if columns[2].button("退回", key=f"no-{plan['plan_id']}") and approver:
                _api(f"/api/v1/cases/{case_id}/approvals", method="POST",
                     payload={"plan_id": plan["plan_id"], "approver": approver, "approved": False})
                st.rerun()
        else:
            state = "已批准" if plan["approved"] else "已退回"
            st.success(f"{state} · {plan['approver']}")


def _recheck(case_id: str, view: dict[str, Any]) -> None:
    if not view["plans"]:
        return
    st.markdown("#### 现场复核")
    st.caption(
        "隔离环境通过只能证明技术回归。现场恢复只能由改动应用之后新采集的证据授予。"
    )
    if not view["applied_at"]:
        if st.button("已按批准的方案应用到现场"):
            _api(f"/api/v1/cases/{case_id}/applied", method="POST", payload={})
            st.rerun()
        return
    st.caption(f"应用于 {view['applied_at'][:19]}")
    with st.form("recheck"):
        directory = st.text_input("应用后新采集的证据包目录")
        if st.form_submit_button("复核") and directory.strip():
            try:
                _api(f"/api/v1/cases/{case_id}/recheck", method="POST",
                     payload={"directory": directory})
            except RuntimeError as exc:
                st.error(str(exc))
                return
            st.rerun()
    outcome = view.get("recheck")
    if outcome:
        label = {
            "site_recovered": "现场已恢复",
            "site_still_failing": "现场仍然失败",
            "not_a_recheck": "这不构成复核：证据不是在应用之后采集的",
        }[outcome["scope"]]
        (st.success if outcome["scope"] == "site_recovered" else st.warning)(label)
        st.json({"改动前": outcome["before"], "改动后": outcome["after"]})


def _evidence(view: dict[str, Any]) -> None:
    if not view["evidence"]:
        return
    seen = sum(1 for item in view["evidence"] if item["examined"])
    with st.expander(f"证据清单（{len(view['evidence'])} 条，其中 {seen} 条被真的读取过）"):
        st.dataframe(
            [
                {
                    "证据": item["evidence_id"],
                    "来源": item["reference"],
                    "类型": item["media_type"],
                    "采集时间": item["captured_at"][:19],
                    "已读取": "是" if item["examined"] else "",
                }
                for item in view["evidence"]
            ],
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(page_title="VisionDoctor", page_icon="🩺", layout="wide")
    _style()
    case_id = _sidebar()
    st.markdown('<div class="vd-title">VisionDoctor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="vd-sub">从一次观察出发，沿诊断链路定界；根因是软件才产生候选修复，'
        "改不改、上不上线由你决定。</div>",
        unsafe_allow_html=True,
    )
    if not case_id:
        st.info("左侧新建一个案件开始。")
        return
    view = _api(f"/api/v1/cases/{case_id}")
    st.divider()
    _inputs(case_id, view)
    _observation(view)
    st.divider()
    left, right = st.columns([1, 1])
    with left:
        _chain(view)
    with right:
        _turns(case_id, view)
        _hypotheses(view)
    _evidence(view)
    _plans(case_id, view)
    _recheck(case_id, view)


main()
