from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .controller import PickCellController


def _controller(args: argparse.Namespace) -> PickCellController:
    return PickCellController(runtime_root=args.runtime_root)


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Operate the standalone PICK-A17 Gazebo cell")
    parser.add_argument("--runtime-root", type=Path, help="override .runtime/gazebo-pick-cell")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("bootstrap-project")
    commands.add_parser("start")
    commands.add_parser("stop")
    capture = commands.add_parser("capture")
    capture.add_argument("--destination", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--workspace", type=Path)
    export = commands.add_parser("export")
    export.add_argument("--run-id", required=True)
    export.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    controller = _controller(args)
    try:
        if args.command == "status":
            result: Any = controller.status()
        elif args.command == "bootstrap-project":
            result = controller.bootstrap_project()
        elif args.command == "start":
            result = controller.start_cell()
        elif args.command == "stop":
            result = controller.stop_cell()
        elif args.command == "capture":
            result = controller.capture(args.destination)
        elif args.command == "run":
            result = controller.run_cycle(args.workspace)
        else:
            result = controller.export_bundle(args.run_id, args.destination)
        _print(result)
        return 0
    except Exception as exc:
        print(f"PICK-A17 command failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
