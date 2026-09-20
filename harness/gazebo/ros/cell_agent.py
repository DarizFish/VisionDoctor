"""Resident cell agent: camera subscriptions and the MoveIt connection stay warm.

A capture or grasp used to start its own ROS process: source the environment,
import the stack, rediscover every topic and action, and wait for ten fresh
observer frames before any work began.  This agent pays that once.  Requests
arrive as one JSON line on a local socket inside the container; progress events
stream back as JSON lines, ending with a single ``result`` line.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grasp_cycle_probe import GraspCycleProbe, run_grasp, write_motion  # noqa: E402
from scene_capture_probe import SceneCaptureProbe, _stamp  # noqa: E402

PORT = int(os.environ.get("PICK_CELL_AGENT_PORT", "18650"))
OBSERVER_WINDOW = 10
FRAME_TIMEOUT_S = 20.0


class LiveRecorder(SceneCaptureProbe):
    """Keeps the newest RGB-D set and a rolling observer clip, ready to save."""

    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()
        self.received: dict[str, float] = {}
        self.window: deque[tuple[float, np.ndarray]] = deque(maxlen=OBSERVER_WINDOW)

    def _rgb(self, message) -> None:
        with self.lock:
            self.rgb = message
            self.received["rgb"] = time.monotonic()

    def _depth(self, message) -> None:
        with self.lock:
            self.depth = message
            self.received["depth"] = time.monotonic()

    def _camera_info(self, message) -> None:
        with self.lock:
            self.camera_info = message
            self.received["camera_info"] = time.monotonic()

    def _observer(self, message) -> None:
        try:
            frame = np.asarray(
                self.bridge.imgmsg_to_cv2(message, desired_encoding="rgb8"), dtype=np.uint8
            ).copy()
        except Exception as exc:
            self.get_logger().warning(f"observer conversion failed: {exc}")
            return
        with self.lock:
            self.window.append((time.monotonic(), frame))

    def frames_after(self, requested_at: float) -> SimpleNamespace:
        """The first aligned RGB-D set taken after the request, with the clip leading to it."""

        deadline = time.monotonic() + FRAME_TIMEOUT_S
        while time.monotonic() < deadline:
            with self.lock:
                fresh = all(
                    self.received.get(name, 0.0) > requested_at
                    for name in ("rgb", "depth", "camera_info")
                )
                clip_ready = (
                    len(self.window) == OBSERVER_WINDOW
                    and self.window[-1][0] > requested_at - 0.5
                )
                if fresh and clip_ready:
                    stamps = (_stamp(self.rgb), _stamp(self.depth), _stamp(self.camera_info))
                    if max(stamps) - min(stamps) <= 0.05:
                        return SimpleNamespace(
                            bridge=self.bridge, rgb=self.rgb, depth=self.depth,
                            camera_info=self.camera_info,
                            observer_frames=[frame for _, frame in self.window],
                        )
            time.sleep(0.02)
        raise TimeoutError("fresh Gazebo sensor data did not arrive after the capture request")


class Connection:
    def __init__(self, conn: socket.socket) -> None:
        self.conn = conn

    def send(self, message: dict[str, Any]) -> None:
        self.conn.sendall((json.dumps(message, sort_keys=True) + "\n").encode("utf-8"))


class ClipWriter:
    """Observer clips written after the capture replies, finished before a bundle is sealed."""

    def __init__(self) -> None:
        self.pending: list[threading.Thread] = []
        self.errors: list[str] = []

    def start(self, snapshot: SimpleNamespace, output: Path) -> None:
        def write() -> None:
            try:
                SceneCaptureProbe.save_observer_clip(snapshot, output)
            except Exception as exc:
                self.errors.append(f"{output}: {type(exc).__name__}: {exc}")

        worker = threading.Thread(target=write, daemon=True)
        worker.start()
        self.pending.append(worker)

    def flush(self) -> list[str]:
        for worker in self.pending:
            worker.join()
        self.pending.clear()
        errors, self.errors = self.errors, []
        return errors


def _capture(
    recorder: LiveRecorder, clips: ClipWriter, request: dict[str, Any], reply: Connection,
) -> None:
    requested_at = time.monotonic()
    reply.send({"type": "event", "event": "waiting_for_frames"})
    snapshot = recorder.frames_after(requested_at)
    reply.send({"type": "event", "event": "saving"})
    output = Path(request["dir"])
    payload = SceneCaptureProbe.save_rgbd(snapshot, output)
    if request.get("defer_clip"):
        clips.start(snapshot, output)
    else:
        SceneCaptureProbe.save_observer_clip(snapshot, output)
    reply.send({"type": "result", **payload})


def _grasp(mover: GraspCycleProbe, request: dict[str, Any], reply: Connection) -> None:
    mover.ik_seed = [float(value) for value in request["seed"]]
    mover.on_event = lambda event: reply.send({"type": "event", **event})
    try:
        payload = run_grasp(mover, request["target"], request.get("pregrasp"))
    finally:
        mover.on_event = None
        mover.ik_seed = None
    write_motion(payload, request.get("output"))
    reply.send({"type": "result", **payload})


def _handle(
    conn: socket.socket, recorder: LiveRecorder, mover: GraspCycleProbe, clips: ClipWriter,
) -> None:
    reply = Connection(conn)
    stream = conn.makefile("r", encoding="utf-8")
    try:
        request = json.loads(stream.readline())
        operation = request.get("op")
        if operation == "ping":
            reply.send({
                "type": "result", "success": True,
                "moveit_ready": mover.move_group.server_is_ready(),
                "observer_frames": len(recorder.window),
            })
        elif operation == "capture":
            _capture(recorder, clips, request, reply)
        elif operation == "flush":
            errors = clips.flush()
            reply.send({"type": "result", "success": not errors, "errors": errors})
        elif operation == "grasp":
            _grasp(mover, request, reply)
        else:
            reply.send({"type": "result", "success": False, "error": f"unknown op {operation}"})
    except Exception as exc:
        reply.send({"type": "result", "success": False, "error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    rclpy.init()
    recorder = LiveRecorder()
    executor = SingleThreadedExecutor()
    executor.add_node(recorder)
    threading.Thread(target=executor.spin, daemon=True).start()
    mover = GraspCycleProbe()
    clips = ClipWriter()
    server = socket.create_server(("127.0.0.1", PORT))
    server.settimeout(0.5)
    print(f"PICK_CELL_AGENT listening on {PORT}", flush=True)
    try:
        while rclpy.ok():
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            with conn:
                _handle(conn, recorder, mover, clips)
    finally:
        server.close()
        executor.shutdown()
        recorder.destroy_node()
        mover.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
