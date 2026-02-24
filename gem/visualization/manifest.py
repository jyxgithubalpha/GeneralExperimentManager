from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .contracts import VisualizationArtifact


def write_manifest(output_path: Path, artifacts: Sequence[VisualizationArtifact]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "saved": sum(1 for item in artifacts if item.status == "saved"),
            "skipped": sum(1 for item in artifacts if item.status == "skipped"),
            "failed": sum(1 for item in artifacts if item.status == "failed"),
        },
        "artifacts": [item.to_dict() for item in artifacts],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path

