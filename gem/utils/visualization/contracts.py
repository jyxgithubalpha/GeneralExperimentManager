
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


ArtifactStatus = Literal["saved", "skipped", "failed"]


@dataclass(frozen=True)
class PlotRequest:
    name: str
    kind: str
    output_path: Path


@dataclass(frozen=True)
class PlotResult:
    request: PlotRequest
    status: ArtifactStatus
    message: Optional[str] = None


@dataclass(frozen=True)
class VisualizationArtifact:
    name: str
    kind: str
    status: ArtifactStatus
    path: Optional[Path] = None
    message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "path": str(self.path) if self.path is not None else None,
            "message": self.message,
        }
