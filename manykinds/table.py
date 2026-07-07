"""Table: heterogeneous columnar data (a pandas DataFrame).

For mixed-dtype tables — cell/gene metadata (AnnData ``obs``/``var``), result
tables — that a ``LabeledArray`` can't hold. Its structural signature is its
column names. Persistence is parquet (needs the ``manykinds[table]`` extra).
"""

import json
import logging
from dataclasses import dataclass

import pandas as pd

from manykinds.spec import KindSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class Table:
    """A pandas DataFrame of heterogeneous columns."""

    df: pd.DataFrame
    provenance: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "provenance", tuple(self.provenance))
        self.validate()

    def validate(self) -> "Table":
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError("Table must wrap a pandas DataFrame")
        if self.df.shape[1] == 0:
            raise ValueError("Table has no columns")
        return self

    def require(self, *dims: str, coords: tuple[str, ...] = ()) -> "Table":
        # A table's structure is its columns; required dims are column names.
        missing = [c for c in dims if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"requires columns {missing}; got {list(self.df.columns)}"
            )
        if coords:
            raise ValueError(f"Table carries no coords; got {list(coords)}")
        return self

    def tagged(self, op_name: str) -> "Table":
        return Table(self.df, self.provenance + (op_name,))

    def spec(self) -> KindSpec:
        """Structural signature: the column names as dims."""
        return KindSpec("Table", tuple(map(str, self.df.columns)), ())

    @staticmethod
    def _normalize(path: str) -> str:
        if not str(path).endswith(".parquet"):
            raise ValueError(f"path must end in .parquet, got {path!r}")
        return str(path)

    def serialize(self, path: str) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "Table persistence needs parquet: pip install 'manykinds[table]'"
            ) from e
        table = pa.Table.from_pandas(self.df)
        meta = dict(table.schema.metadata or {})
        # provenance rides in the parquet key-value metadata so it round-trips.
        meta[b"manykinds_provenance"] = json.dumps(list(self.provenance)).encode()
        table = table.replace_schema_metadata(meta)
        pq.write_table(table, self._normalize(path))

    @classmethod
    def load(cls, path):
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "Table persistence needs parquet: pip install 'manykinds[table]'"
            ) from e
        table = pq.read_table(cls._normalize(path))
        raw = (table.schema.metadata or {}).get(b"manykinds_provenance")
        provenance = tuple(json.loads(raw)) if raw else ()
        return cls(table.to_pandas(), provenance)

    def __repr__(self) -> str:
        return (
            f"Table(rows={self.df.shape[0]}, columns={list(self.df.columns)}, "
            f"provenance={self.provenance})"
        )
