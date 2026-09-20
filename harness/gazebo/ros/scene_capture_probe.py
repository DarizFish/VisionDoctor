from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from cv_bridge import CvBridge
from PIL import Image
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ImageMessage

RGB_TOPIC = "/pick_cell/rgbd/image"
DEPTH_TOPIC = "/pick_cell/rgbd/depth_image"
CAMERA_INFO_TOPIC = "/pick_cell/rgbd/camera_info"
OBSERVER_TOPIC = "/pick_cell/observer"


def _stamp(message: ImageMessage | CameraInfo) -> float:
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9


def _keyframe_indices(frames: list[Any]) -> list[int]:
    return sorted({0, len(frames) // 2, len(frames) - 1})


class SceneCaptureProbe(Node):
    def __init__(self) -> None:
        super().__init__("pick_cell_scene_capture")
        self.bridge = CvBridge()
        self.rgb: ImageMessage | None = None
        self.depth: ImageMessage | None = None
        self.camera_info: CameraInfo | None = None
        self.observer_frames: list[np.ndarray] = []
        self.create_subscription(ImageMessage, RGB_TOPIC, self._rgb, qos_profile_sensor_data)
        self.create_subscription(ImageMessage, DEPTH_TOPIC, self._depth, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self._camera_info, qos_profile_sensor_data
        )
        self.create_subscription(
            ImageMessage, OBSERVER_TOPIC, self._observer, qos_profile_sensor_data
        )

    def _rgb(self, message: ImageMessage) -> None:
        self.rgb = message

    def _depth(self, message: ImageMessage) -> None:
        self.depth = message

    def _camera_info(self, message: CameraInfo) -> None:
        self.camera_info = message

    def _observer(self, message: ImageMessage) -> None:
        if len(self.observer_frames) >= 18:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")
            self.observer_frames.append(np.asarray(frame, dtype=np.uint8).copy())
        except Exception as exc:
            self.get_logger().warning(f"observer conversion failed: {exc}")

    def wait_for_data(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            complete = (
                self.rgb is not None and self.depth is not None and self.camera_info is not None
            )
            if complete and len(self.observer_frames) >= 10:
                stamps = (_stamp(self.rgb), _stamp(self.depth), _stamp(self.camera_info))
                if max(stamps) - min(stamps) <= 0.05:
                    return
        missing = [
            name
            for name, value in (
                ("rgb", self.rgb),
                ("depth", self.depth),
                ("camera_info", self.camera_info),
            )
            if value is None
        ]
        if len(self.observer_frames) < 10:
            missing.append("observer video frames")
        raise TimeoutError("Gazebo sensor data was not received: " + ", ".join(missing))

    def save_observer_clip(self, output: Path) -> None:
        """Write the observer frames, keyframes and GIF that capture.json names."""

        frame_dir = output / "observer-frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(self.observer_frames):
            Image.fromarray(frame, mode="RGB").save(frame_dir / f"observer-{index:03d}.png")
        for index in _keyframe_indices(self.observer_frames):
            Image.fromarray(self.observer_frames[index], mode="RGB").save(
                output / f"observer-key-{index:03d}.png"
            )
        frames = [Image.fromarray(frame, mode="RGB") for frame in self.observer_frames]
        frames[0].save(
            output / "observer.gif", save_all=True, append_images=frames[1:], duration=125, loop=0,
        )

    def save(self, output: Path) -> dict[str, Any]:
        metadata = self.save_rgbd(output)
        self.save_observer_clip(output)
        return metadata

    def save_rgbd(self, output: Path) -> dict[str, Any]:
        """Write what perception consumes, plus the record naming the clip still to come."""

        if self.rgb is None or self.depth is None or self.camera_info is None:
            raise RuntimeError("scene capture is incomplete")
        rgb = np.asarray(
            self.bridge.imgmsg_to_cv2(self.rgb, desired_encoding="rgb8"), dtype=np.uint8
        )
        depth = np.asarray(self.bridge.imgmsg_to_cv2(self.depth, desired_encoding="passthrough"))
        if self.depth.encoding == "16UC1":
            depth = depth.astype(np.float32) * 0.001
        else:
            depth = depth.astype(np.float32)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
            raise ValueError(f"unexpected RGB-D shapes: {rgb.shape}, {depth.shape}")
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(output / "rgb.png")
        np.save(output / "depth.npy", depth, allow_pickle=False)
        camera = {
            "frame_id": self.camera_info.header.frame_id,
            "width": int(self.camera_info.width),
            "height": int(self.camera_info.height),
            "intrinsics": list(self.camera_info.k),
            "distortion": list(self.camera_info.d),
            "stamp_s": _stamp(self.camera_info),
            "rgb_depth_registered": True,
            "depth_unit": "m",
            "depth_measurement": "optical_z",
            "pose_axes": "body_x_forward_y_left_z_up",
        }
        (output / "camera-info.json").write_text(
            json.dumps(camera, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        keyframes = [
            f"observer-key-{index:03d}.png" for index in _keyframe_indices(self.observer_frames)
        ]
        depth_valid = np.isfinite(depth) & (depth > 0)
        metadata = {
            "success": True,
            "capture_id": f"capture-{uuid.uuid4().hex[:12]}",
            "captured_at": datetime.now(UTC).isoformat(),
            "clock_domain": "ros_sim_time_s",
            "sensor_stamps_s": {
                "rgb": _stamp(self.rgb), "depth": _stamp(self.depth),
                "camera_info": _stamp(self.camera_info),
            },
            "rgb_path": "rgb.png",
            "depth_path": "depth.npy",
            "observer_clip_path": "observer.gif",
            "observer_keyframes": keyframes,
            "camera_info_path": "camera-info.json",
            "frame_id": self.camera_info.header.frame_id,
            "depth_valid_ratio": float(depth_valid.mean()),
            "message_stamp_spread_s": max(
                _stamp(self.rgb), _stamp(self.depth), _stamp(self.camera_info)
            )
            - min(_stamp(self.rgb), _stamp(self.depth), _stamp(self.camera_info)),
        }
        (output / "capture.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return metadata


def main() -> int:
    output_value = os.environ.get("PICK_CELL_CAPTURE_DIR")
    if not output_value:
        raise RuntimeError("PICK_CELL_CAPTURE_DIR is required")
    rclpy.init()
    node = SceneCaptureProbe()
    try:
        node.wait_for_data(float(os.environ.get("PICK_CELL_CAPTURE_TIMEOUT_S", "35")))
        payload = node.save(Path(output_value))
        print("PICK_CELL_CAPTURE=" + json.dumps(payload, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            "PICK_CELL_CAPTURE="
            + json.dumps({"success": False, "error": str(exc)}, sort_keys=True),
            flush=True,
        )
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
