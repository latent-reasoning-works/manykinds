"""Shared version-1 storage identity and pickle-free NPZ validation.

Unversioned archives are deliberately rejected: their metadata is ambiguous and
old NPZ trails require pickle. Recreate them from trusted source data instead.
"""
import json

import numpy as np

NAMESPACE = "_manykinds"


def format_metadata(kind: str, encoding: str) -> dict:
    return {"format": "manykinds", "version": 1, "kind": kind, "encoding": encoding}


def validate_metadata(metadata, kind: str, encodings: tuple[str, ...]) -> dict:
    if metadata is None:
        raise ValueError("unversioned legacy archive; regenerate with manykinds format v1")
    if not isinstance(metadata, dict) or metadata.get("format") != "manykinds":
        raise ValueError("invalid manykinds format marker")
    if type(metadata.get("version")) is not int or metadata["version"] != 1:
        raise ValueError(f"unsupported manykinds format version: {metadata.get('version')!r}")
    if metadata.get("kind") != kind:
        raise ValueError(f"expected kind {kind}, got {metadata.get('kind')!r}")
    if metadata.get("encoding") not in encodings:
        raise ValueError(f"unsupported {kind} encoding: {metadata.get('encoding')!r}")
    return metadata


def validate_provenance(provenance) -> tuple[str, ...]:
    if not isinstance(provenance, (list, tuple)) or not all(
        isinstance(op, str) for op in provenance
    ):
        raise ValueError("provenance must be a sequence of strings")
    return tuple(provenance)


def validate_ids(ids: np.ndarray, name: str) -> None:
    # Fixed-width strings and plain numeric ids preserve dtype without pickle.
    if ids.ndim != 1 or ids.dtype.kind not in "biufcSU":
        raise ValueError(
            f"{name} must be 1-D with numeric, fixed-width bytes or Unicode dtype; "
            f"got {ids.shape}, {ids.dtype} (object ids are unsupported)"
        )


def save_npz(path: str, kind: str, arrays: dict, provenance) -> None:
    arrays = dict(arrays)
    arrays[NAMESPACE] = np.asarray(json.dumps(format_metadata(kind, "npz")))
    arrays["provenance"] = np.asarray(validate_provenance(provenance), dtype=str)
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise ValueError(f"{name} has object dtype; pickle storage is unsupported")
    np.savez_compressed(path, **arrays)


def load_npz_metadata(archive, kind: str) -> tuple[str, ...]:
    metadata = None
    if NAMESPACE in archive:
        marker = archive[NAMESPACE]
        if marker.ndim != 0 or marker.dtype.kind != "U":
            raise ValueError("invalid manykinds format marker: expected a Unicode scalar")
        metadata = json.loads(marker.item())
    validate_metadata(metadata, kind, ("npz",))
    trail = archive["provenance"]
    if trail.ndim != 1 or trail.dtype.kind != "U":
        raise ValueError("provenance must be a 1-D Unicode array")
    return validate_provenance(trail.tolist())
