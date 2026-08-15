from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from visiondoctor.adapters.gazebo import GazeboAdapter
from visiondoctor.adapters.gazebo_view import GazeboVisualAdapter
from visiondoctor.demo.scenario import run_demo
from visiondoctor.llm import ModelProtocolError, ModelSettings, OpenAICompatibleGateway
from visiondoctor.llm.tools import terminal_tool
from visiondoctor.product import load_incident, run_incident
from visiondoctor.reliability import run_reliability_gate
from visiondoctor.sandbox import DockerPythonRunner


def _default_workspace() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(".visiondoctor") / f"demo-{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visiondoctor")
    subcommands = parser.add_subparsers(dest="command", required=True)
    demo = subcommands.add_parser("demo", help="run the model-driven bundled evaluation case")
    demo.add_argument("--workspace", type=Path, default=None)
    demo.add_argument("--sandbox", choices=("auto", "local", "docker"), default="auto")
    demo.add_argument(
        "--robot-backend", choices=("auto", "dataset", "gazebo"), default="auto"
    )
    product_run = subcommands.add_parser(
        "run", help="diagnose and repair a caller-supplied Incident JSON"
    )
    product_run.add_argument("--incident", type=Path, required=True)
    product_run.add_argument("--workspace", type=Path, default=None)
    product_run.add_argument("--sandbox", choices=("docker",), default="docker")
    product_run.add_argument(
        "--robot-backend", choices=("dataset", "gazebo"), default="dataset"
    )
    subcommands.add_parser("model-check", help="make a real tool-call request to the model API")
    subcommands.add_parser("check-gazebo", help="report optional ROS 2/Gazebo availability")
    subcommands.add_parser("build-sandbox", help="build the hardened Python sandbox image")
    subcommands.add_parser("build-gazebo", help="build the ROS 2/Gazebo/MoveIt/UR image")
    gazebo_view = subcommands.add_parser(
        "gazebo-view", help="open the official Gazebo Qt GUI from the verified Docker image"
    )
    gazebo_view.add_argument("--session-root", type=Path, default=Path(".visiondoctor/gazebo-view"))
    gazebo_view.add_argument("--no-motion", action="store_true")
    gazebo_view.add_argument("--status", action="store_true")
    gazebo_view.add_argument("--stop", action="store_true")
    gazebo_contract = subcommands.add_parser(
        "gazebo-contract", help="run the real headless UR5e fixed-motion contract"
    )
    gazebo_contract.add_argument("--output", type=Path, default=None)
    gazebo_rgbd = subcommands.add_parser(
        "gazebo-rgbd-contract", help="capture and verify a real Gazebo RGB-D frame"
    )
    gazebo_rgbd.add_argument(
        "--output-dir", type=Path, default=Path(".artifacts/gazebo-rgbd-contract")
    )
    gazebo_rgbd.add_argument("--case-id", default="scene-049")
    serve = subcommands.add_parser("serve", help="serve the VisionDoctor HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--data-root", type=Path, default=Path(".visiondoctor/server"))
    reliability = subcommands.add_parser(
        "reliability", help="repeat and qualify the complete deterministic workflow"
    )
    reliability.add_argument("--runs", type=int, default=10)
    reliability.add_argument("--workspace-root", type=Path, required=True)
    reliability.add_argument("--sandbox", choices=("auto", "local", "docker"), default="auto")
    reliability.add_argument(
        "--robot-backend", choices=("auto", "dataset", "gazebo"), default="auto"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "model-check":
        gateway = OpenAICompatibleGateway(ModelSettings.from_environment())
        tool = terminal_tool(
            "report_ready",
            "Confirm that tool calling is available.",
            {"ready": {"type": "boolean"}},
            ["ready"],
        )
        turn = gateway.complete(
            [
                {
                    "role": "system",
                    "content": "Call report_ready with ready=true. Do not answer in plain text.",
                },
                {"role": "user", "content": "Perform the protocol check."},
            ],
            (tool,),
        )
        if not turn.tool_calls or turn.tool_calls[0].name != "report_ready":
            raise ModelProtocolError("model did not perform the required tool call")
        print(json.dumps({"status": "ok", "model": gateway.model}, indent=2))
        return
    if args.command == "check-gazebo":
        status = GazeboAdapter.availability()
        print(
            json.dumps(
                {
                    "available": status.available,
                    "reason": status.reason,
                    "runtime": status.runtime,
                    "image": status.image,
                },
                indent=2,
            )
        )
        return
    if args.command == "build-sandbox":
        runner = DockerPythonRunner()
        project_root = Path(__file__).resolve().parents[2]
        runner.build_image(project_root / "docker" / "sandbox.Dockerfile", project_root)
        print(json.dumps({"image": runner.image, "status": "built"}, indent=2))
        return
    if args.command == "build-gazebo":
        project_root = Path(__file__).resolve().parents[2]
        GazeboAdapter.build_image(project_root)
        print(json.dumps({"image": GazeboAdapter.IMAGE, "status": "built"}, indent=2))
        return
    if args.command == "gazebo-view":
        if args.status and args.stop:
            raise SystemExit("--status and --stop are mutually exclusive")
        project_root = Path(__file__).resolve().parents[2]
        adapter = GazeboVisualAdapter(project_root, args.session_root)
        if args.stop:
            output = adapter.stop()
        elif args.status:
            output = adapter.status()
        else:
            output = adapter.start(run_motion=not args.no_motion)
        print(json.dumps(output, ensure_ascii=True, indent=2))
        return
    if args.command == "gazebo-contract":
        project_root = Path(__file__).resolve().parents[2]
        contract = GazeboAdapter.run_fixed_motion_contract(project_root)
        output = {
            **contract.payload,
            "duration_s": contract.duration_s,
            "image": GazeboAdapter.IMAGE,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            args.output.with_suffix(args.output.suffix + ".log").write_text(
                contract.container_logs,
                encoding="utf-8",
            )
        print(json.dumps(output, ensure_ascii=False, indent=2))
        if not contract.success:
            raise SystemExit(2)
        return
    if args.command == "gazebo-rgbd-contract":
        project_root = Path(__file__).resolve().parents[2]
        contract = GazeboAdapter.run_rgbd_capture_contract(
            project_root,
            args.output_dir,
            case_id=args.case_id,
        )
        print(json.dumps(contract.payload, ensure_ascii=False, indent=2))
        if not contract.success:
            raise SystemExit(2)
        return
    if args.command == "serve":
        import uvicorn

        from visiondoctor.api.app import ApiSettings, create_app

        data_root = args.data_root.resolve()
        settings = ApiSettings(
            data_root=data_root,
            database_path=data_root / "visiondoctor.sqlite3",
        )
        uvicorn.run(create_app(settings), host=args.host, port=args.port)
        return
    if args.command == "reliability":
        summary = run_reliability_gate(
            args.workspace_root,
            runs=args.runs,
            sandbox_mode=args.sandbox,
            robot_backend=args.robot_backend,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not summary["all_passed"]:
            raise SystemExit(2)
        return
    if args.command == "run":
        incident = load_incident(args.incident.resolve())
        workspace = args.workspace or (
            Path(".visiondoctor")
            / f"run-{incident.incident_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
        result = run_incident(
            workspace,
            incident,
            sandbox_mode=args.sandbox,
            robot_backend=args.robot_backend,
        )
        print(
            json.dumps(
                {
                    "incident_id": result.incident.incident_id,
                    "state": result.state,
                    "root_cause": result.diagnosis.root_cause,
                    "model": result.diagnosis.model,
                    "selected_candidate": result.selected_candidate.candidate_id,
                    "run_root": result.run_root,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return
    workspace = args.workspace or _default_workspace()
    result = run_demo(
        workspace,
        sandbox_mode=args.sandbox,
        robot_backend=args.robot_backend,
    )
    print(
        json.dumps(
            {
                "incident_id": result.incident.incident_id,
                "state": result.state,
                "history": result.history,
                "root_cause": result.diagnosis.root_cause,
                "selected_candidate": result.selected_candidate.candidate_id,
                "candidate_decisions": {
                    report.candidate_id: report.decision for report in result.candidate_validations
                },
                "external_gates": [
                    {
                        "name": gate.name,
                        "passed": gate.passed,
                        "details": gate.details,
                    }
                    for gate in result.external_gate_results
                ],
                "run_root": result.run_root,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
