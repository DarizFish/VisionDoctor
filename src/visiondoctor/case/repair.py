"""What gets approved, and the hash that keeps it from being swapped afterwards."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from .chain import Segment


class RepairPlan(BaseModel):
    """A software candidate, frozen the moment it is put up for review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    case_id: str
    target_segment: Segment
    hypothesis_id: str
    project_revision: str
    diff: str

    @property
    def frozen_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ApprovalRecord(BaseModel):
    """A human decision about one exact plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    frozen_hash: str
    approved: bool
    approver: str
    decided_at: datetime
