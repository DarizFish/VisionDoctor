"""The read-only collection boundary.

An adapter observes; it never acts.  It cannot start a robot, change a camera,
write configuration or run arbitrary commands -- those are external side effects
and go through a human or deployment channel instead.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from .bundle import MANIFEST_NAME, Artifact, ObservationBundle, load_manifest


class EnvironmentAdapter(Protocol):
    """What every collection source must offer.

    Capability discovery and health reporting join this protocol when a live
    adapter first needs them; a recorded bundle answers neither question.
    """

    def collect(self) -> ObservationBundle: ...

    def read_artifact(self, artifact_id: str) -> bytes: ...


class FileBundleAdapter:
    """Read a bundle that was already collected into a directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def collect(self) -> ObservationBundle:
        return load_manifest(self.root)

    def read_artifact(self, artifact_id: str) -> bytes:
        artifact = self._locate(artifact_id)
        path = (self.root / artifact.path).resolve()
        if self.root not in path.parents:
            raise ValueError(f"artifact leaves its bundle: {artifact_id}")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != artifact.sha256:
            raise ValueError(f"artifact does not match its manifest hash: {artifact_id}")
        return payload

    def _locate(self, artifact_id: str) -> Artifact:
        for artifact in self.collect().artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact
        raise KeyError(f"{artifact_id} is not listed in {MANIFEST_NAME}")
