"""Render the case's actual graph findings, without propagating verdicts."""

from __future__ import annotations

import json
from typing import Any

COLORS = {"untested": "#94a3b8", "cleared": "#059669", "suspect": "#e11d48"}
#: The host's isolation is drawn as the fill, so it never hides the model's verdict.
FILLS = {"candidate": "#fde2cf", "exonerated": "#dcf2e3"}
HOST_LABEL = {
    "candidate": "核算：剩余候选", "exonerated": "核算：已排除", "unexamined": "核算：未检验",
}


def graph_dot(graph: dict[str, Any], standing: dict[str, str] | None = None) -> str:
    standing = standing or {}

    def quote(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    lines = [
        "digraph grasp {",
        'rankdir=TB; bgcolor="transparent";',
        'node [shape=box, style="rounded", fontname="Microsoft YaHei", fontsize=12];',
        'edge [fontname="Microsoft YaHei", fontsize=9];',
    ]
    for node in graph["nodes"]:
        color = COLORS[node["status"]]
        layers = (node.get("finding") or {}).get("layers", [])
        host = standing.get(node["id"])
        label = (node["name"] + ("\n含源码层解释" if "source" in layers else "")
                 + (f"\n{HOST_LABEL[host]}" if host else ""))
        frames = 2 if node.get("software") else 1
        fill = (f', style="rounded,filled", fillcolor="{FILLS[host]}"'
                if host in FILLS else "")
        lines.append(
            f'{quote(node["id"])} [label={quote(label)}, '
            f'color="{color}", fontcolor="{color}", peripheries={frames}{fill}];'
        )
    for edge in graph["edges"]:
        color = COLORS[edge["status"]]
        width = 2.5 if edge["status"] != "untested" else 1
        lines.append(
            f'{quote(edge["source"])} -> {quote(edge["target"])} '
            f'[label={quote(edge["name"])}, color="{color}", penwidth={width}];'
        )
    return "\n".join([*lines, "}"])
