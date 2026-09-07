"""SparseGraph: a graph as plain numpy arrays (edge list + node ids + weights).

A dependency-free graph kind — no heavy graph library, just numpy. Structurally
satisfies :class:`manykinds.base.Kind`. Only needs numpy.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from manykinds._persistence import load_npz_metadata, save_npz, validate_ids
from manykinds.spec import KindSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class SparseGraph:
    """An E×2 integer edge list + a 1-D array of node ids, optionally weighted.

    ``edge_weights`` (when present) is a 1-D array aligned to ``edges`` — for
    weighted/signed graphs like gene-regulatory networks.
    """

    edges: np.ndarray
    node_ids: np.ndarray
    provenance: tuple[str, ...] = ()
    edge_weights: Optional[np.ndarray] = None

    def __post_init__(self):
        # frozen: bypass the immutability guard to normalize inputs in place.
        object.__setattr__(self, "edges", np.asarray(self.edges))
        object.__setattr__(self, "node_ids", np.asarray(self.node_ids))
        if self.edge_weights is not None:
            object.__setattr__(self, "edge_weights", np.asarray(self.edge_weights))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        self.validate()

    @property
    def _components(self) -> tuple[str, ...]:
        """Named structural components this graph carries (weights only if present)."""
        base = ("edges", "node_ids")
        return base + (("edge_weights",) if self.edge_weights is not None else ())

    def validate(self) -> "SparseGraph":
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError(f"edges must be E×2, got shape {self.edges.shape}")
        if self.node_ids.ndim != 1:
            raise ValueError(f"node_ids must be 1-D, got shape {self.node_ids.shape}")
        if not np.issubdtype(self.edges.dtype, np.integer):
            raise ValueError(f"edges must be integer dtype, got {self.edges.dtype}")
        if self.edge_weights is not None:
            if self.edge_weights.ndim != 1:
                raise ValueError(
                    f"edge_weights must be 1-D, got shape {self.edge_weights.shape}"
                )
            if self.edge_weights.shape[0] != self.edges.shape[0]:
                raise ValueError(
                    f"edge_weights length {self.edge_weights.shape[0]} != "
                    f"{self.edges.shape[0]} edges — one weight per edge"
                )
        return self

    def require(self, *dims: str, coords: tuple[str, ...] = ()) -> "SparseGraph":
        missing_dims = [d for d in dims if d not in self._components]
        if missing_dims:
            raise ValueError(f"requires dims {missing_dims}; got {self._components}")

        # A SparseGraph carries no coords.
        if coords:
            raise ValueError(f"requires coords {list(coords)}; got ()")
        return self

    def tagged(self, op_name: str) -> "SparseGraph":
        """Return a copy with ``op_name`` appended to the provenance trail.

        Part of the ``Kind`` protocol: the op registry appends to this trail as it
        runs each op. Immutable: the original is untouched.
        """
        return SparseGraph(
            self.edges, self.node_ids, self.provenance + (op_name,), self.edge_weights
        )

    def spec(self) -> KindSpec:
        """This graph's structural signature (its named components, no coords)."""
        return KindSpec("SparseGraph", self._components, ())

    @staticmethod
    def _normalize(path: str) -> str:
        if not str(path).endswith(".npz"):
            raise ValueError(f"path must end in .npz, got {path!r}")
        return str(path)

    def serialize(self, path: str) -> None:
        logger.info(f"Serializing {type(self).__name__} to {path}")
        arrays = dict(
            edges=self.edges,
            node_ids=self.node_ids,
        )
        if self.edge_weights is not None:
            arrays["edge_weights"] = self.edge_weights
        validate_ids(self.node_ids, "node_ids")
        save_npz(self._normalize(path), "SparseGraph", arrays, self.provenance)

    @classmethod
    def load(cls, path):
        with np.load(cls._normalize(path), allow_pickle=False) as d:
            provenance = load_npz_metadata(d, "SparseGraph")
            edge_weights = d["edge_weights"] if "edge_weights" in d else None
            node_ids = d["node_ids"]
            validate_ids(node_ids, "node_ids")
            # validate called from __post_init__
            return cls(d["edges"], node_ids, provenance, edge_weights)

    @property
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        return self.edges, self.node_ids

    def __repr__(self) -> str:
        w = "" if self.edge_weights is None else ", weighted"
        return (
            f"SparseGraph(num_nodes={self.node_ids.shape[0]}, "
            f"num_edges={self.edges.shape[0]}{w})"
        )
