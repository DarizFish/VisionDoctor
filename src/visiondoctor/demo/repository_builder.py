from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

BASELINE_SOURCE = '''from __future__ import annotations

import numpy as np


def compose_target_pose(T_base_camera: np.ndarray, T_camera_object: np.ndarray) -> np.ndarray:
    """Return T_base_object using the T_target_source convention."""
    return np.asarray(T_base_camera, dtype=float) @ np.asarray(T_camera_object, dtype=float)
'''

FAULTY_SOURCE = '''from __future__ import annotations

import numpy as np


def compose_target_pose(T_base_camera: np.ndarray, T_camera_object: np.ndarray) -> np.ndarray:
    """Return T_base_object using the T_target_source convention."""
    return np.asarray(T_camera_object, dtype=float) @ np.asarray(T_base_camera, dtype=float)
'''

WEAK_TEST_SOURCE = """from __future__ import annotations

import unittest

import numpy as np

from pose_transformer import compose_target_pose


class PoseTransformerSmokeTest(unittest.TestCase):
    def test_output_is_a_homogeneous_transform(self) -> None:
        T_base_camera = np.eye(4)
        T_camera_object = np.eye(4)
        T_camera_object[:3, 3] = [0.02, -0.01, 0.7]
        actual = compose_target_pose(T_base_camera, T_camera_object)
        self.assertEqual(actual.shape, (4, 4))
        np.testing.assert_allclose(actual[3], [0.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
"""

RUNNER_SOURCE = """from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pose_transformer import compose_target_pose


def main() -> None:
    payload = json.load(sys.stdin)
    inputs = [
        (item["case_id"], np.asarray(item["T_base_camera"], dtype=float),
         np.asarray(item["T_camera_object"], dtype=float))
        for item in payload["cases"]
    ]
    outputs = [
        {"case_id": case_id, "T_base_object": compose_target_pose(a, b).tolist()}
        for case_id, a, b in inputs
    ]
    samples = []
    calibration_samples = []
    repeats = int(payload.get("benchmark_repeats", 300))
    def measure(function) -> float:
        started = time.perf_counter_ns()
        for _repeat in range(repeats):
            for _case_id, a, b in inputs:
                function(a, b)
        return (time.perf_counter_ns() - started) / 1e9 / repeats / len(inputs)

    def calibration(a, b):
        return np.asarray(a, dtype=float) @ np.asarray(b, dtype=float)

    for sample_index in range(15):
        if sample_index % 2:
            calibration_samples.append(measure(calibration))
            samples.append(measure(compose_target_pose))
        else:
            samples.append(measure(compose_target_pose))
            calibration_samples.append(measure(calibration))
    latency = statistics.median(samples)
    for output in outputs:
        output["latency_s"] = latency
    json.dump(
        {
            "results": outputs,
            "benchmark_samples_s": samples,
            "calibration_samples_s": calibration_samples,
            "calibration_latency_s": statistics.median(calibration_samples),
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
"""

TEST_RUNNER_SOURCE = """from __future__ import annotations

import sys
import unittest
from pathlib import Path

root = Path(__file__).parent
sys.path.insert(0, str(root / "src"))
suite = unittest.defaultTestLoader.discover(str(root / "tests"))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
"""


@dataclass(frozen=True)
class DemoRepository:
    path: Path
    baseline_commit: str
    faulty_commit: str


def _run(repository: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    return result.stdout.strip()


def _commit_env(timestamp: str) -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = timestamp
    env["GIT_COMMITTER_DATE"] = timestamp
    return env


def build_demo_repository(root: Path) -> DemoRepository:
    root.mkdir(parents=True, exist_ok=False)
    _run(root, "init", "-b", "main")
    _run(root, "config", "user.name", "VisionDoctor Demo")
    _run(root, "config", "user.email", "visiondoctor@example.invalid")
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "pose_transformer.py").write_text(BASELINE_SOURCE, encoding="utf-8")
    (root / "tests" / "test_pose_transformer.py").write_text(WEAK_TEST_SOURCE, encoding="utf-8")
    (root / "runner.py").write_text(RUNNER_SOURCE, encoding="utf-8")
    (root / "test_runner.py").write_text(TEST_RUNNER_SOURCE, encoding="utf-8")
    (root / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    _run(root, "add", ".")
    _run(
        root,
        "commit",
        "-m",
        "baseline: correct target-source composition",
        env=_commit_env("2026-08-05T09:00:00+08:00"),
    )
    baseline = _run(root, "rev-parse", "HEAD")

    (root / "src" / "pose_transformer.py").write_text(FAULTY_SOURCE, encoding="utf-8")
    _run(root, "add", "src/pose_transformer.py")
    _run(
        root,
        "commit",
        "-m",
        "regression: reverse transform composition",
        env=_commit_env("2026-08-05T09:05:00+08:00"),
    )
    faulty = _run(root, "rev-parse", "HEAD")
    return DemoRepository(path=root.resolve(), baseline_commit=baseline, faulty_commit=faulty)
