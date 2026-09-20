"""The host's isolation as it stands in a case.

A structural diagnosis enters the case as derived evidence.  The newest one for
each run and part is what stands; older ones stay in the ledger as history.
"""

from __future__ import annotations

from typing import Any

from .case import Case, Evidence

REFERENCE = "derived/structural-diagnosis"


def structural_results(case: Case) -> list[Evidence]:
    latest: dict[tuple[str, str], Evidence] = {}
    for item in case.evidence:
        if item.reference == REFERENCE and item.content:
            latest[(item.content["run_id"], item.content["part_id"])] = item
    return list(latest.values())


def diagnosed_run(case: Case) -> str | None:
    """The first observation is the one under diagnosis; later ones are references."""

    return case.observations[0].run_id if case.observations else None


def node_standing(case: Case, run_id: str | None = None) -> dict[str, str]:
    """``candidate``, ``exonerated`` or ``unexamined`` per node, across one run's parts.

    Isolation is computed per fault mode (``node`` or ``node.mode``).  A node is a
    candidate if any of its modes is one in any part; it is exonerated only if
    every mode that was examined, in every part, was exonerated.
    """

    run_id = run_id or diagnosed_run(case)
    isolations = [
        item.content["isolation"] for item in structural_results(case)
        if item.content["run_id"] == run_id
    ]
    verdicts: dict[str, list[str]] = {}
    for isolation in isolations:
        for key, verdict in (("candidates", "candidate"), ("exonerated", "exonerated"),
                             ("unexamined", "unexamined")):
            for fault in isolation[key]:
                verdicts.setdefault(fault_node(fault), []).append(verdict)
    return {
        node: "candidate" if "candidate" in seen
        else "exonerated" if set(seen) == {"exonerated"} else "unexamined"
        for node, seen in sorted(verdicts.items())
    }


def fault_node(fault: str) -> str:
    return fault.split(".", maxsplit=1)[0]


def isolation_view(case: Case) -> list[dict[str, Any]]:
    """Every standing structural result, the diagnosed run first."""

    run_id = diagnosed_run(case)
    rows = [
        {"evidence_id": item.evidence_id, "diagnosed_run": item.content["run_id"] == run_id,
         **{key: value for key, value in item.content.items() if key != "sources"}}
        for item in structural_results(case)
    ]
    return sorted(rows, key=lambda row: (not row["diagnosed_run"], row["run_id"], row["part_id"]))
