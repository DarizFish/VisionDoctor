from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from PIL import Image
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ImageMessage

RGB_TOPIC = "/visiondoctor/rgbd/image"
DEPTH_TOPIC = "/visiondoctor/rgbd/depth_image"
CAMERA_INFO_TOPIC = "/visiondoctor/rgbd/camera_info"
SYNCHRONIZATION_TOLERANCE_S = 0.02


def _stamp_seconds(message: ImageMessage | CameraInfo) -> float:
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9


class GazeboRgbdCaptureProbe(Node):
    def __init__(self) -> None:
        super().__init__("visiondoctor_gazebo_rgbd_capture")
        self.bridge = CvBridge()
        self.rgb_message: ImageMessage | None = None
        self.depth_message: ImageMessage | None = None
        self.camera_info: CameraInfo | None = None
        self.create_subscription(
            ImageMessage, RGB_TOPIC, self._receive_rgb, qos_profile_sensor_data
        )
        self.create_subscription(
            ImageMessage, DEPTH_TOPIC, self._receive_depth, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self._receive_camera_info, qos_profile_sensor_data
        )

    def _receive_rgb(self, message: ImageMessage) -> None:
        self.rgb_message = message

    def _receive_depth(self, message: ImageMessage) -> None:
        self.depth_message = message

    def _receive_camera_info(self, message: CameraInfo) -> None:
        self.camera_info = message

    def wait_for_frame(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (
                self.rgb_message is not None
                and self.depth_message is not None
                and self.camera_info is not None
            ):
                stamps = (
                    _stamp_seconds(self.rgb_message),
                    _stamp_seconds(self.depth_message),
                    _stamp_seconds(self.camera_info),
                )
                if max(stamps) - min(stamps) <= SYNCHRONIZATION_TOLERANCE_S:
                    return
        missing = [
            name
            for name, value in (
                ("rgb", self.rgb_message),
                ("depth", self.depth_message),
                ("camera_info", self.camera_info),
            )
            if value is None
        ]
        raise TimeoutError(f"Gazebo RGB-D messages not received: {', '.join(missing)}")

    def save(self, output_dir: Path) -> dict[str, object]:
        if self.rgb_message is None or self.depth_message is None or self.camera_info is None:
            raise RuntimeError("capture was requested before a complete RGB-D frame arrived")
        rgb = np.asarray(
            self.bridge.imgmsg_to_cv2(self.rgb_message, desired_encoding="rgb8"),
            dtype=np.uint8,
        )
        depth = np.asarray(
            self.bridge.imgmsg_to_cv2(self.depth_message, desired_encoding="passthrough")
        )
        if self.depth_message.encoding == "16UC1":
            depth = depth.astype(np.float32) * 0.001
        else:
            depth = depth.astype(np.float32)
        if rgb.shape != (480, 640, 3):
            raise ValueError(f"unexpected RGB shape: {rgb.shape}")
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"RGB/depth shape mismatch: {rgb.shape} vs {depth.shape}")
        camera_matrix = np.asarray(self.camera_info.k, dtype=float).reshape(3, 3)
        focal_length = min(camera_matrix[0, 0], camera_matrix[1, 1])
        if not np.isfinite(camera_matrix).all() or focal_length <= 0:
            raise ValueError("Gazebo camera intrinsics are invalid")
        output_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = output_dir / "rgb.png"
        depth_path = output_dir / "depth.npy"
        Image.fromarray(rgb, mode="RGB").save(rgb_path)
        np.save(depth_path, depth, allow_pickle=False)
        stamps = (
            _stamp_seconds(self.rgb_message),
            _stamp_seconds(self.depth_message),
            _stamp_seconds(self.camera_info),
        )
        valid = np.isfinite(depth) & (depth > 0)
        return {
            "success": True,
            "rgb_path": rgb_path.name,
            "depth_path": depth_path.name,
            "rgb_encoding": self.rgb_message.encoding,
            "depth_encoding": self.depth_message.encoding,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "camera_matrix": camera_matrix.tolist(),
            "frame_id": self.camera_info.header.frame_id,
            "depth_valid_ratio": float(valid.mean()),
            "message_stamp_spread_s": float(max(stamps) - min(stamps)),
            "captured_at": datetime.now(UTC).isoformat(),
        }


def main() -> int:
    rclpy.init()
    node = GazeboRgbdCaptureProbe()
    try:
        node.wait_for_frame(float(os.getenv("VISIONDOCTOR_CAPTURE_TIMEOUT_S", "30")))
        payload = node.save(Path(os.environ["VISIONDOCTOR_OUTPUT_DIR"]))
        print("VISIONDOCTOR_RGBD_RESULT=" + json.dumps(payload, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            "VISIONDOCTOR_RGBD_RESULT="
            + json.dumps({"success": False, "error": str(exc)}, sort_keys=True),
            flush=True,
        )
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
