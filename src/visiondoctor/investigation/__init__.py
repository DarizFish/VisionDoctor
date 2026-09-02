"""Investigating one observation: the read-only surface, the ledger, the view."""

from .agent import InvestigationError, investigate
from .tools import Toolbox
from .transform_check import Residual, TransformCheck, check_transform_chain
from .turn import CALL_BUDGET, DecisionTurn, Investigation, ToolCall
from .view import SOURCE_TOOLS, SYSTEM_PROMPT, TOOLS, build_view, tools_for

__all__ = [
    "CALL_BUDGET",
    "SOURCE_TOOLS",
    "SYSTEM_PROMPT",
    "TOOLS",
    "DecisionTurn",
    "Investigation",
    "InvestigationError",
    "Residual",
    "ToolCall",
    "Toolbox",
    "TransformCheck",
    "build_view",
    "check_transform_chain",
    "investigate",
    "tools_for",
]
