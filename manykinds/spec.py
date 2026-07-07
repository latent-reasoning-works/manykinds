"""KindSpec — a data-free structural signature of a kind.

A :class:`~manykinds.base.Kind` describes structure it *has* and enforces it at
runtime via ``require`` (which needs a constructed instance). A ``KindSpec``
describes that same structure *without any data* — the kind's name plus the dims
and coords it carries — so an orchestrator can type an op's inputs/outputs and
check whether a chain is valid **before** running it (and without importing the
array stack).

The two layers share one vocabulary: ``KindSpec.satisfies`` is the plan-time
mirror of ``Kind.require``. A concrete kind can report its own spec via
``.spec()`` so runtime data can be checked against a declared op signature.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KindSpec:
    """The structural signature of a kind: name + dims + coords, no data.

    ``kind`` is the concrete kind's class name (a string, so a planner can carry
    specs without importing the kind). ``dims``/``coords`` are the named axes and
    labels the kind carries.
    """

    kind: str
    dims: tuple[str, ...] = ()
    coords: tuple[str, ...] = ()

    def satisfies(self, need: "KindSpec") -> bool:
        """True iff a value described by ``self`` can flow where ``need`` is required.

        The plan-time analogue of :meth:`Kind.require`: same kind, and ``self``
        provides *every* dim and coord ``need`` asks for (a superset — an op that
        requires ``(cell, gene)`` accepts an input that also carries ``time``).
        """
        return (
            self.kind == need.kind
            and set(need.dims) <= set(self.dims)
            and set(need.coords) <= set(self.coords)
        )

    def __repr__(self) -> str:
        return f"KindSpec({self.kind!r}, dims={self.dims}, coords={self.coords})"
