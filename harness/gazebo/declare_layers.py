"""Add the exporter's layer declaration to bundles exported before it existed.

Only ``bundle.json`` changes, using the same rule the controller now applies when
it writes a bundle.  Artifact bytes and their digests are left untouched.

    python -m harness.gazebo.declare_layers <export-dir> [<export-dir> ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .controller import PickCellController


def declare(directory: Path) -> int:
    manifest_path = directory / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = 0
    for artifact in manifest.get("artifacts", []):
        layer = PickCellController.artifact_layer(str(artifact["path"]))
        if artifact.get("layer") != layer:
            artifact["layer"] = layer
            changed += 1
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    for directory in parser.parse_args().directories:
        print(f"{directory}: {declare(directory)} artifacts declared")


if __name__ == "__main__":
    main()
