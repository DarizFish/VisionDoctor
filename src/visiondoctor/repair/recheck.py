"""Compare what the cell did before a change with what it does after one.

An isolated replay can only say the software now computes what was intended.
Whether the cell recovered is a different claim, and only evidence collected
from the cell after the change was applied can carry it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from visiondoctor.environment import ObservationBundle


@dataclass(frozen=True)
class Recheck:
    before_run_id: str
    after_run_id: str
    after_created_at: datetime
    applied_at: datetime | None
    before: dict[str, bool]
    after: dict[str, bool]

    @property
    def collected_after_the_change(self) -> bool:
        """Evidence gathered before the change cannot speak about the change."""

        return self.applied_at is not None and self.applied_at < self.after_created_at

    @property
    def recovered(self) -> bool:
        return bool(self.after) and all(self.after.values())

    @property
    def scope(self) -> str:
        if not self.collected_after_the_change:
            return "not_a_recheck"
        return "site_recovered" if self.recovered else "site_still_failing"

    def as_dict(self) -> dict[str, Any]:
        return {
            "before_run_id": self.before_run_id,
            "after_run_id": self.after_run_id,
            "before": self.before,
            "after": self.after,
            "scope": self.scope,
        }


def recheck(
    *,
    before: ObservationBundle,
    after: ObservationBundle,
    applied_at: datetime | None,
) -> Recheck:
    """Read both observations' own task results; conclude nothing about the cause."""

    return Recheck(
        before_run_id=before.run_id,
        after_run_id=after.run_id,
        after_created_at=after.created_at,
        applied_at=applied_at,
        before={item.part_id: item.success for item in before.results},
        after={item.part_id: item.success for item in after.results},
    )
