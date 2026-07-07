"""The ``Kind`` protocol — the structural contract every kind satisfies.

A *kind* is a self-describing data object that guards its own structure: it
validates on construction and refuses an op that demands structure it lacks
(vs. a bare array whose axes you have to trust). This is the shared vocabulary:
an orchestrator types its ops against the protocol, and a concrete kind is
substitutable iff it offers all three members — no inheritance required.

The concrete kinds (:class:`~manykinds.labeled_array.LabeledArray`,
:class:`~manykinds.sparse_graph.SparseGraph`) live in this package too. This
module stays dependency-free (pure ``typing``) so the protocol — and anything
that only annotates against it — imports without the array stack.

Producers (dataset adapters, model wrappers, any downstream tool) *construct*
these kinds; they do not redefine the vocabulary.

Design contract:
- kinds are *structural only* — domain words like ``cell``/``gene`` live in
  adapters, never in the kind;
- measured time enters as a per-cell ``time`` **coord**, not a dim.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Kind(Protocol):
    """Structural contract every kind satisfies (no inheritance required).

    A concrete kind is substitutable into an op registry iff it offers all three
    members. ``require`` is the precondition/postcondition gate a registry calls;
    ``provenance`` + ``tagged`` are the trail it appends to as it runs ops (the
    state persisted when the system's history is saved).
    """

    provenance: tuple[str, ...]

    def require(self, *dims: str, coords: tuple[str, ...] = ()) -> "Kind":
        """Assert named dims/coords are present; raise cleanly if not. Returns self."""
        ...

    def tagged(self, op_name: str) -> "Kind":
        """Return a copy with ``op_name`` appended to the provenance trail."""
        ...
