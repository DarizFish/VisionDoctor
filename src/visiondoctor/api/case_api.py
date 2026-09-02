"""HTTP surface for the case path.

Every route reads or pushes one case. Nothing here starts a robot, changes a
camera or applies a patch: applying is a human act, and the API only records
that it happened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from visiondoctor.case import CaseService


class NewCase(BaseModel):
    title: str
    guided_motion: bool = True


class Observation(BaseModel):
    directory: str


class Attachment(BaseModel):
    name: str
    media_type: str
    content_base64: str


class Attachments(BaseModel):
    files: list[Attachment]


class Project(BaseModel):
    repository: str
    revision: str
    replay_command: list[str]
    test_command: list[str] | None = None


class Turn(BaseModel):
    prompt: str


class Approval(BaseModel):
    plan_id: str
    approver: str
    approved: bool


def create_case_app(service: CaseService | None = None) -> FastAPI:
    cases = service or CaseService()
    app = FastAPI(title="VisionDoctor cases", version="1")

    def guard(call, *args, **kwargs) -> Any:
        try:
            return call(*args, **kwargs)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "cases": len(cases.records)}

    @app.get("/api/v1/cases")
    def listing() -> list[dict[str, Any]]:
        return cases.listing()

    @app.post("/api/v1/cases", status_code=201)
    def create(body: NewCase) -> dict[str, Any]:
        case_id = cases.create(body.title, guided_motion=body.guided_motion)
        return cases.view(case_id)

    @app.get("/api/v1/cases/{case_id}")
    def view(case_id: str) -> dict[str, Any]:
        return guard(cases.view, case_id)

    @app.post("/api/v1/cases/{case_id}/observations")
    def attach(case_id: str, body: Observation) -> dict[str, Any]:
        return guard(cases.attach_observation, case_id, Path(body.directory))

    @app.post("/api/v1/cases/{case_id}/attachments")
    def attach_files(case_id: str, body: Attachments) -> dict[str, Any]:
        return guard(
            cases.attach_files,
            case_id,
            [item.model_dump() for item in body.files],
        )

    @app.get("/api/v1/cases/{case_id}/evidence/{evidence_id}")
    def evidence_content(case_id: str, evidence_id: str) -> dict[str, Any]:
        return guard(cases.evidence_content, case_id, evidence_id)

    @app.post("/api/v1/cases/{case_id}/project")
    def bind(case_id: str, body: Project) -> dict[str, Any]:
        return guard(
            cases.bind_project,
            case_id,
            Path(body.repository),
            body.revision,
            tuple(body.replay_command),
            tuple(body.test_command) if body.test_command else None,
        )

    @app.post("/api/v1/cases/{case_id}/turns")
    def turn(case_id: str, body: Turn) -> dict[str, Any]:
        return guard(cases.run_turn, case_id, body.prompt)

    @app.post("/api/v1/cases/{case_id}/approvals")
    def approve(case_id: str, body: Approval) -> dict[str, Any]:
        return guard(
            cases.approve, case_id, body.plan_id, body.approver, approved=body.approved
        )

    @app.post("/api/v1/cases/{case_id}/applied")
    def applied(case_id: str) -> dict[str, Any]:
        return {"applied_at": guard(cases.mark_applied, case_id).isoformat()}

    @app.post("/api/v1/cases/{case_id}/recheck")
    def run_recheck(case_id: str, body: Observation) -> dict[str, Any]:
        return guard(cases.run_recheck, case_id, Path(body.directory))

    return app


app = create_case_app()
