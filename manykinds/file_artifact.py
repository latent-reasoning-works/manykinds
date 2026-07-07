"""FileArtifact: an opaque reference to a file a tool produced.

The escape hatch for outputs the vocabulary doesn't model structurally — carry a
file (path + format + checksum) through a pipeline unchanged. It has no dims or
coords; ``require`` rejects any structural ask. Needs only the standard library.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass

from manykinds.spec import KindSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class FileArtifact:
    """A reference to an on-disk file: its path, format, and optional checksum."""

    path: str
    format: str = ""  # e.g. "h5ad", "csv"; inferred from the suffix if empty
    checksum: str = ""  # optional sha256 for caching / content-addressed provenance
    provenance: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.format and isinstance(self.path, str):
            object.__setattr__(self, "format", os.path.splitext(self.path)[1].lstrip("."))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        self.validate()

    def validate(self) -> "FileArtifact":
        if not self.path or not isinstance(self.path, str):
            raise ValueError("FileArtifact needs a non-empty path")
        if not os.path.isfile(self.path):
            raise ValueError(f"FileArtifact path is not a file: {self.path!r}")
        return self

    def digest(self) -> str:
        """sha256 of the referenced file (returns the stored checksum if present)."""
        if self.checksum:
            return self.checksum
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def with_digest(self) -> "FileArtifact":
        """A copy with the checksum computed and stored."""
        return FileArtifact(self.path, self.format, self.digest(), self.provenance)

    def require(self, *dims: str, coords: tuple[str, ...] = ()) -> "FileArtifact":
        if dims or coords:
            raise ValueError(
                f"FileArtifact is opaque (no dims/coords); got dims={list(dims)} "
                f"coords={list(coords)}"
            )
        return self

    def tagged(self, op_name: str) -> "FileArtifact":
        return FileArtifact(
            self.path, self.format, self.checksum, self.provenance + (op_name,)
        )

    def spec(self) -> KindSpec:
        return KindSpec("FileArtifact", (), ())

    @staticmethod
    def _normalize(path: str) -> str:
        if not str(path).endswith(".json"):
            raise ValueError(f"path must end in .json, got {path!r}")
        return str(path)

    def serialize(self, path: str) -> None:
        """Write the artifact manifest (not the referenced file itself) as JSON."""
        with open(self._normalize(path), "w") as f:
            json.dump(
                {
                    "path": self.path,
                    "format": self.format,
                    "checksum": self.checksum,
                    "provenance": list(self.provenance),
                },
                f,
            )

    @classmethod
    def load(cls, path):
        with open(cls._normalize(path)) as f:
            d = json.load(f)
        return cls(
            d["path"], d.get("format", ""), d.get("checksum", ""),
            tuple(d.get("provenance", ())),
        )

    def __repr__(self) -> str:
        return f"FileArtifact(path={self.path!r}, format={self.format!r})"
