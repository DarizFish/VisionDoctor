from __future__ import annotations

import hashlib
from pathlib import Path

from visiondoctor.schemas import ArtifactRef
from visiondoctor.validation import DefaultMultimodalValidator


def test_validator_detects_evidence_replacement_after_collection(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    path.write_bytes(b"trusted-evidence")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = ArtifactRef(
        artifact_id="scene:rgb",
        path=str(path),
        sha256=digest,
        media_type="image/png",
    )

    assert DefaultMultimodalValidator._check_artifact(artifact, {artifact.artifact_id: digest})

    path.write_bytes(b"replaced-evidence")

    assert not DefaultMultimodalValidator._check_artifact(artifact, {artifact.artifact_id: digest})
