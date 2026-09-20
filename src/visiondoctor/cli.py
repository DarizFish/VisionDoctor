from __future__ import annotations

import argparse
import json
from pathlib import Path

from visiondoctor.adapters.gazebo import GazeboAdapter
from visiondoctor.adapters.gazebo_view import GazeboVisualAdapter
from visiondoctor.llm import ModelProtocolError, ModelSettings, OpenAICompatibleGateway
from visiondoctor.llm.tools import terminal_tool
from visiondoctor.sandbox import DockerPythonRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visiondoctor")
    subcommands = parser.add_subparsers(dest="command", required=True)
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
    serve = subcommands.add_parser("serve", help="serve the VisionDoctor Case HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
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

        from visiondoctor.api.case_api import create_case_app

        uvicorn.run(create_case_app(), host=args.host, port=args.port)
        return
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
