"""manykinds — the typed data-kind vocabulary for the latent-reasoning stack.

``Kind`` (the structural protocol), ``KindSpec`` and :mod:`manykinds.params` need no
extra. The array kinds (``LabeledArray``, ``SparseGraph``, ``FileArtifact``, ``Table``,
``Sequence``) are loaded on first access; their packages come with
``manykinds[arrays]`` or from the dependent's own dependencies.
Producers construct these kinds; orchestrators type their ops against the
protocol. See :mod:`manykinds.base` for the contract.
"""
from __future__ import annotations

from manykinds.base import Kind
from manykinds.spec import KindSpec

__all__ = [
    "FileArtifact",
    "Kind",
    "KindSpec",
    "LabeledArray",
    "Sequence",
    "SparseGraph",
    "Table",
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

        # Try the real import. Never check whether the extra is installed: a
        # dependent may supply a kind's packages itself.
        try:
            module = importlib.import_module(f"manykinds.{_LAZY[name]}")
        except ImportError as exc:
            raise ImportError(
                f"manykinds.{name} needs packages that are not installed ({exc}). "
                'Install the array kinds with: uv add "manykinds[arrays]"'
            ) from exc
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
