"""manykinds — the typed data-kind vocabulary for the latent-reasoning stack.

``Kind`` (the structural protocol) is dependency-light; the concrete kinds
``LabeledArray`` (xarray) and ``SparseGraph`` (numpy) are loaded on first access.
Producers construct these kinds; orchestrators type their ops against the
protocol. See :mod:`manykinds.base` for the contract.
"""
from __future__ import annotations

from manykinds.base import Kind
from manykinds.spec import KindSpec

__all__ = [
    "Kind",
    "KindSpec",
    "LabeledArray",
    "SparseGraph",
    "FileArtifact",
    "Table",
    "Sequence",
]

# name -> submodule that defines it (imported on first attribute access)
_LAZY = {
    "LabeledArray": "labeled_array",
    "SparseGraph": "sparse_graph",
    "FileArtifact": "file_artifact",
    "Table": "table",
    "Sequence": "sequence",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f"manykinds.{_LAZY[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
