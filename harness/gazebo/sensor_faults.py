"""Sensor-stream corruption for controlled experiments; never a product tool.

These are injected observation faults, not a claim to simulate the underlying
optics. The modified pixels are consumed by perception before command generation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def apply_sensor_fault(capture_root: Path, kind: str) -> None:
    if kind not in {"target_depth_dropout", "rgb_underexposure"}:
        raise ValueError(f"unknown sensor experiment: {kind}")
    rgb_path = capture_root / "rgb.png"
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    if kind == "rgb_underexposure":
        shutil.copy2(rgb_path, capture_root / "raw-rgb.png")
        Image.fromarray((rgb.astype(float) * 0.08).astype(np.uint8)).save(rgb_path)
    else:
        # A local depth hole covers high-chroma surface pixels and their small
        # neighbourhood. No object-center labels or diagnostic answers are used.
        colorful = np.ptp(rgb.astype(float), axis=2) > 70
        mask = colorful.copy()
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                mask |= np.roll(colorful, (dy, dx), axis=(0, 1))
        depth_path = capture_root / "depth.npy"
        shutil.copy2(depth_path, capture_root / "raw-depth.npy")
        depth = np.load(depth_path, allow_pickle=False)
        depth[mask] = np.nan
        np.save(depth_path, depth, allow_pickle=False)
        meta_path = capture_root / "capture.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["depth_valid_ratio"] = float((np.isfinite(depth) & (depth > 0)).mean())
        metadata["depth_streams"] = {"sensor_output": "raw-depth.npy", "program_input": "depth.npy"}
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
