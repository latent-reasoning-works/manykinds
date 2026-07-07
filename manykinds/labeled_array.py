"""LabeledArray: an xarray DataArray with named dimensions.

The canonical concrete kind for cell×gene matrices and embeddings. Structurally
satisfies :class:`manykinds.base.Kind` (``provenance`` / ``require`` /
``tagged``) without inheriting from it — the registry types against the protocol.
Needs the ``manykinds`` extra (xarray + zarr, and sparse for sparse-backed
arrays).
"""

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

from manykinds.spec import KindSpec

logger = logging.getLogger(__name__)


def _json_safe(attrs: dict) -> dict:
    """Coerce numpy scalars / bytes in attrs to JSON-serializable values.

    The sparse path stashes ``da.attrs`` as a nested dict inside zarr attrs, which
    xarray does NOT recursively coerce — so a numpy int/bool/bytes value (e.g. a
    ``genome`` tag read from an ``.h5``) would crash zarr's JSON encoder. Mirror
    what xarray does for top-level attrs.
    """
    out = {}
    for k, v in attrs.items():
        if isinstance(v, np.generic):
            v = v.item()
        elif isinstance(v, bytes):
            v = v.decode()
        out[str(k)] = v
    return out


@dataclass(frozen=True, eq=False)
class LabeledArray:
    """xarray DataArray with named dimensions.

    Metadata: domain attrs (e.g. genome) live in ``da.attrs``; labels (e.g.
    gene_ids) in ``da.coords``. ``provenance`` is the trail of ops already applied.
    """

    da: xr.DataArray
    provenance: tuple[str, ...] = ()

    def __post_init__(self):
        # frozen: bypass the immutability guard to normalize provenance to a tuple
        object.__setattr__(self, "provenance", tuple(self.provenance))
        self.validate()

    def validate(self) -> "LabeledArray":
        if not isinstance(self.da, xr.DataArray):
            raise ValueError("LabeledArray must wrap a DataArray")
        if self.da.size == 0:
            raise ValueError(f"LabeledArray is empty (shape {self.da.shape})")
        return self

    def require(self, *dims: str, coords: tuple[str, ...] = ()) -> "LabeledArray":
        missing_dims = [d for d in dims if d not in self.da.dims]
        if missing_dims:
            raise ValueError(f"requires dims {missing_dims}; got {tuple(self.da.dims)}")

        missing_coords = [c for c in coords if c not in self.da.coords]
        if missing_coords:
            raise ValueError(f"requires coords {missing_coords}; got {tuple(self.da.coords)}")
        return self

    def tagged(self, op_name: str) -> "LabeledArray":
        """Return a copy with ``op_name`` appended to the provenance trail.

        Part of the ``Kind`` protocol: the op registry appends to this trail as it
        runs each op. Immutable: the original is untouched.
        """
        return LabeledArray(self.da, self.provenance + (op_name,))

    def spec(self) -> KindSpec:
        """This array's structural signature (name + dims + coords), data-free."""
        return KindSpec("LabeledArray", tuple(self.da.dims), tuple(self.da.coords))

    @staticmethod
    def _normalize(path: str) -> str:
        if not str(path).endswith(".zarr"):
            raise ValueError(f"path must end in .zarr, got {path!r}")
        return str(path)

    def serialize(self, path: str) -> None:
        path = self._normalize(path)
        logger.info(f"Serializing {type(self).__name__} to {path}")
        import sparse

        da = self.da
        # Dense arrays serialize natively; only sparse needs the COO-component
        # format below (zarr can't serialize a sparse-backed DataArray directly).
        if not isinstance(da.data, sparse.COO):
            # provenance rides in attrs (only when non-empty) so the op trail
            # survives the round-trip, reusing xarray's attrs-preservation.
            if self.provenance:
                da = da.assign_attrs(provenance=list(self.provenance))
            da.to_zarr(path, mode="w")
            return

        coo = da.data
        ds = xr.Dataset(
            {
                "coo_coords": (("ndim", "nnz"), coo.coords),
                "coo_data": (("nnz",), coo.data),
            },
            attrs={
                "shape": list(coo.shape),
                "dims": list(da.dims),
                "fill_value": coo.fill_value.item(),
                "name": da.name or "",
                "da_attrs": _json_safe(da.attrs),
                "provenance": list(self.provenance),
                # in conjunction with the loop below, keeps additional coords
                # aligned to their dim (e.g. cell + time)
                "coord_dims": {name: list(c.dims) for name, c in da.coords.items()},
            },
        )

        for name, c in da.coords.items():
            ds = ds.assign_coords(
                {f"coord_{name}": (tuple(f"len_{d}" for d in c.dims), c.values)}
            )
        ds.to_zarr(path, mode="w")

    @classmethod
    def load(cls, path):
        path = cls._normalize(path)
        ds = xr.open_zarr(path)
        if "coo_data" not in ds:
            # Dense: reload as a DataArray to preserve its numpy backing.
            da = xr.open_dataarray(path, engine="zarr")
            # pull provenance back out of attrs so it doesn't linger as a domain attr
            provenance = tuple(da.attrs.pop("provenance", ()))
            return cls(da, provenance=provenance)  # validate via __post_init__

        import sparse

        coo = sparse.COO(
            coords=ds["coo_coords"].values,
            data=ds["coo_data"].values,
            shape=tuple(ds.attrs["shape"]),
            fill_value=ds.attrs["fill_value"],
        )
        coord_dims = ds.attrs["coord_dims"]
        coords = {
            name: (tuple(coord_dims[name]), ds[f"coord_{name}"].values)
            for name in coord_dims
        }
        da = xr.DataArray(
            coo, dims=ds.attrs["dims"], coords=coords, name=ds.attrs["name"] or None
        )
        da.attrs = dict(ds.attrs.get("da_attrs", {}))
        provenance = tuple(ds.attrs.get("provenance", ()))
        return cls(da, provenance=provenance)  # validate called from __post_init__

    def __repr__(self) -> str:
        return (
            f"LabeledArray(dims={list(self.da.dims)}, shape={self.da.shape}, "
            f"provenance={self.provenance})"
        )
