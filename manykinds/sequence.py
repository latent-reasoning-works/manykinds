"""Sequence: a set of biological sequences over an alphabet (DNA / protein).

N sequences drawn from a fixed alphabet, with optional per-sequence ids. Needs
only numpy; persists to ``.npz``.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from manykinds._persistence import load_npz_metadata, save_npz, validate_ids
from manykinds.spec import KindSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class Sequence:
    """A 1-D array of sequence strings over ``alphabet``, with optional ids."""

    sequences: np.ndarray
    alphabet: str  # e.g. "ACGT" (DNA); the amino-acid set (protein)
    ids: Optional[np.ndarray] = None
    provenance: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "sequences", np.asarray(self.sequences, dtype=str))
        if self.ids is not None:
            object.__setattr__(self, "ids", np.asarray(self.ids))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        self.validate()

    def validate(self) -> "Sequence":
        if self.sequences.ndim != 1 or self.sequences.size == 0:
            raise ValueError(
                f"sequences must be a non-empty 1-D array, got shape {self.sequences.shape}"
            )
        if not self.alphabet:
            raise ValueError("Sequence needs a non-empty alphabet")
        allowed = set(self.alphabet)
        for s in self.sequences:
            bad = set(s) - allowed
            if bad:
                raise ValueError(
                    f"sequence has chars {sorted(bad)} outside alphabet {self.alphabet!r}"
                )
        if self.ids is not None and self.ids.shape[0] != self.sequences.shape[0]:
            raise ValueError(
                f"ids length {self.ids.shape[0]} != {self.sequences.shape[0]} sequences"
            )
        return self

    @property
    def _components(self) -> tuple[str, ...]:
        return ("sequence",)

    def require(self, *dims: str, coords: tuple[str, ...] = ()) -> "Sequence":
        missing = [d for d in dims if d not in self._components]
        if missing:
            raise ValueError(f"requires dims {missing}; got {self._components}")
        avail = ("id",) if self.ids is not None else ()
        missing_c = [c for c in coords if c not in avail]
        if missing_c:
            raise ValueError(f"requires coords {missing_c}; got {avail}")
        return self

    def tagged(self, op_name: str) -> "Sequence":
        return Sequence(
            self.sequences, self.alphabet, self.ids, self.provenance + (op_name,)
        )

    def spec(self) -> KindSpec:
        coords = ("id",) if self.ids is not None else ()
        return KindSpec("Sequence", ("sequence",), coords)

    @staticmethod
    def _normalize(path: str) -> str:
        if not str(path).endswith(".npz"):
            raise ValueError(f"path must end in .npz, got {path!r}")
        return str(path)

    def serialize(self, path: str) -> None:
        arrays = dict(
            sequences=self.sequences,
            alphabet=np.asarray(self.alphabet),
        )
        if self.ids is not None:
            validate_ids(self.ids, "ids")
            arrays["ids"] = self.ids
        save_npz(self._normalize(path), "Sequence", arrays, self.provenance)

    @classmethod
    def load(cls, path):
        with np.load(cls._normalize(path), allow_pickle=False) as d:
            provenance = load_npz_metadata(d, "Sequence")
            ids = d["ids"] if "ids" in d else None
            if ids is not None:
                validate_ids(ids, "ids")
            return cls(d["sequences"], str(d["alphabet"]), ids, provenance)

    def __repr__(self) -> str:
        return f"Sequence(n={self.sequences.shape[0]}, alphabet={self.alphabet!r})"
