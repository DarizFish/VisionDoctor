"""Structural analysis: what a model can detect and tell apart, before any number.

Only the incidence of unknown variables in equations is used.  A measurement
bound to evidence turns its variables known, so the same template has less
redundancy in a case that lacks a record.  The over-determined part comes from a
maximum matching (Dulmage-Mendelsohn); detectability and isolability follow from
it.  Tests are written in advance: the structure only decides whether a test is
admissible here and which faults it responds to.  No test is generated.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from visiondoctor.case.templates import TEMPLATES, Template, Test

Incidence = dict[str, frozenset[str]]

ASSUMPTIONS = (
    "单故障假设：一次只考虑一个节点出故障",
    "故障一定会使对它敏感的检验违反；因此敏感检验通过即核算排除",
    "结构只看方程含哪些变量，不看数值；检验的数值由预先实现的残差计算器给出",
)


def _matching(incidence: Incidence) -> dict[str, str]:
    """Variable -> equation, maximum cardinality (augmenting paths)."""

    matched: dict[str, str] = {}

    def augment(equation: str, seen: set[str]) -> bool:
        for variable in sorted(incidence[equation]):
            if variable in seen:
                continue
            seen.add(variable)
            if variable not in matched or augment(matched[variable], seen):
                matched[variable] = equation
                return True
        return False

    for equation in sorted(incidence):
        augment(equation, set())
    return matched


def overdetermined(incidence: Incidence) -> frozenset[str]:
    """Equations reachable by alternating paths from an unmatched equation."""

    matched = _matching(incidence)
    frontier = [name for name in incidence if name not in set(matched.values())]
    reached = set(frontier)
    while frontier:
        for variable in incidence[frontier.pop()]:
            nxt = matched.get(variable)
            if nxt is not None and nxt not in reached:
                reached.add(nxt)
                frontier.append(nxt)
    return frozenset(reached)


def redundancy(incidence: Incidence) -> int:
    part = overdetermined(incidence)
    return len(part) - len({variable for name in part for variable in incidence[name]})


class Model:
    """The union of some templates, with the variables this case can measure."""

    def __init__(
        self, templates: Iterable[Template], bound: Iterable[str],
        missing: Iterable[str] = (),
    ) -> None:
        self.templates = tuple(templates)
        self.bound = frozenset(bound)
        #: Variables whose bound record lacks the fields that carry them.
        self.missing = frozenset(missing)
        self.relations = {eq.id: eq.relation for t in self.templates for eq in t.equations}
        self.known = frozenset(
            variable
            for template in self.templates
            for measurement in template.measurements
            if measurement.id in self.bound
            for variable in measurement.provides
        ) - self.missing
        self.incidence: Incidence = {
            eq.id: frozenset(eq.variables) - self.known
            for template in self.templates for eq in template.equations
        }
        faults: dict[str, set[str]] = {}
        self.fault_names: dict[str, str] = {}
        for template in self.templates:
            for fault in template.faults:
                faults.setdefault(fault.id, set()).update(fault.equations)
                self.fault_names[fault.id] = fault.name
        self.faults = {fault: frozenset(eqs) for fault, eqs in faults.items()}
        self.tests = tuple(test for template in self.templates for test in template.tests)
        self.part = overdetermined(self.incidence)

    def detectable(self, node: str, without: frozenset[str] = frozenset()) -> bool:
        if not without:
            return bool(self.faults[node] & self.part)
        rest = {name: vars_ for name, vars_ in self.incidence.items() if name not in without}
        return bool(self.faults[node] & overdetermined(rest))

    def isolable(self, node: str, other: str) -> bool:
        """``node`` stays detectable once every equation ``other`` would break is removed."""

        return self.detectable(node, self.faults[other])

    def admissible(self, test: Test) -> str | None:
        """Why the test cannot run here, or ``None`` if it can."""

        missing = [name for name in test.measurements if name not in self.bound]
        if missing:
            return "未绑定测量：" + "、".join(missing)
        sub = {name: self.incidence[name] for name in test.equations}
        if overdetermined(sub) != frozenset(test.equations):
            lacking = sorted({variable for name in test.equations for variable in sub[name]}
                             & self.missing)
            return ("绑定的记录缺少变量 " + "、".join(lacking) if lacking
                    else "该检验的方程集在当前结构下不是过约束的")
        return None

    def sensitivity(self, test: Test) -> frozenset[str]:
        return frozenset(
            node for node, eqs in self.faults.items() if eqs & set(test.equations)
        )

    def summary(self) -> dict[str, Any]:
        nodes = sorted(self.faults)
        detectable = [node for node in nodes if self.detectable(node)]
        cannot = {
            node: sorted(other for other in detectable
                         if other != node and not self.isolable(node, other))
            for node in detectable
        }
        return {
            "incidence": [
                {"equation": name, "relation": self.relations[name],
                 "unknowns": sorted(self.incidence[name])}
                for name in sorted(self.incidence, key=lambda item: int(item[1:]))
            ],
            "faults": {fault: {"node": fault.split(".", maxsplit=1)[0], "name": name}
                       for fault, name in sorted(self.fault_names.items())},
            "known_variables": sorted(self.known),
            "overdetermined_equations": sorted(self.part, key=lambda item: int(item[1:])),
            "redundancy": redundancy(self.incidence),
            "detectable": detectable,
            "undetectable": [node for node in nodes if node not in detectable],
            "cannot_isolate_from": cannot,
            "non_isolable_groups": _groups(detectable, cannot),
        }


def _groups(nodes: list[str], cannot: dict[str, list[str]]) -> list[list[str]]:
    """Faults that cannot be told apart in either direction, gathered together."""

    groups: list[list[str]] = []
    seen: set[str] = set()
    for node in nodes:
        if node in seen:
            continue
        group, frontier = {node}, [node]
        while frontier:
            current = frontier.pop()
            for other in cannot[current]:
                if current in cannot.get(other, ()) and other not in group:
                    group.add(other)
                    frontier.append(other)
        seen |= group
        if len(group) > 1:
            groups.append(sorted(group))
    return groups


def isolate(
    model: Model, outcomes: dict[str, str], sensitivity: dict[str, frozenset[str]],
) -> dict[str, Any]:
    """Candidates consistent with the tests that ran, under the stated assumptions."""

    passed = [name for name, status in outcomes.items() if status == "pass"]
    failed = [name for name, status in outcomes.items() if status == "fail"]
    exonerated = {node for name in passed for node in sensitivity[name]}
    nodes = sorted(model.faults)
    candidates = [] if not failed else [
        node for node in nodes
        if node not in exonerated and all(node in sensitivity[name] for name in failed)
    ]
    if not failed:
        conclusion = "no_fault_detected" if passed else "nothing_evaluated"
    elif not candidates:
        conclusion = "outside_model"
    else:
        conclusion = "isolated" if len(candidates) == 1 else "ambiguous"
    return {
        "conclusion": conclusion,
        "failed_tests": failed,
        "passed_tests": passed,
        "candidates": candidates,
        "exonerated": sorted(exonerated),
        "unexamined": [node for node in nodes if node not in exonerated and node not in candidates],
        #: A clean result says nothing about these: the bound records cannot see them.
        "undetectable": [node for node in nodes if not model.detectable(node)],
    }


def gaps(
    model: Model, candidates: list[str], sensitivity: dict[str, frozenset[str]],
) -> dict[str, list[list[str]]]:
    """Remaining candidates that only a new test, or only a new measurement, could separate."""

    def by_tests(a: str, b: str) -> bool:
        return any((a in sens) != (b in sens) for sens in sensitivity.values())

    missing_test: list[list[str]] = []
    unseparable: dict[str, list[str]] = {node: [] for node in candidates}
    for index, a in enumerate(candidates):
        for b in candidates[index + 1:]:
            if by_tests(a, b):
                continue
            if model.isolable(a, b) or model.isolable(b, a):
                missing_test.append([a, b])
            else:
                unseparable[a].append(b)
                unseparable[b].append(a)
    return {
        "missing_test": missing_test,
        "missing_measurement": _groups(candidates, unseparable),
    }


def suggestions(model: Model, candidates: list[str]) -> list[dict[str, Any]]:
    """Tests not run yet that would split the remaining candidates, and what to bind for them."""

    if len(candidates) < 2:
        return []
    ran = {test.id for test in model.tests}
    found = []
    for template in TEMPLATES.values():
        for test in template.tests:
            if test.id in ran and not model.admissible(test):
                continue
            templates = {item.id: item for item in (*model.templates, template)}
            trial = Model(
                templates.values(), model.bound | set(test.measurements), model.missing
            )
            if trial.admissible(test):
                continue
            sens = trial.sensitivity(test)
            inside = [node for node in candidates if node in sens]
            outside = [node for node in candidates if node not in sens]
            if inside and outside:
                found.append({
                    "template_id": template.id,
                    "test": test.id,
                    "name": test.name,
                    "bind": [name for name in test.measurements if name not in model.bound],
                    "separates": inside,
                    "from": outside,
                })
    return found
