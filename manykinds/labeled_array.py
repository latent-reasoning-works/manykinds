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

from manykinds._persistence import (
    NAMESPACE, format_metadata, validate_metadata, validate_provenance,
)
from manykinds.spec import KindSpec

logger = logging.getLogger(__name__)


def _json_safe(value):
    """Recursively coerce numpy scalars, arrays and UTF-8 bytes for JSON attrs."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _domain_attrs(attrs: dict) -> dict:
    if NAMESPACE in attrs:
        raise ValueError(f"{NAMESPACE!r} is reserved for storage metadata")
    return _json_safe(attrs)


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
        is_sparse = isinstance(da.data, sparse.COO)
        metadata = format_metadata("LabeledArray", "coo" if is_sparse else "dense")
        metadata.update(
            dims=list(da.dims), name=da.name, attrs=_domain_attrs(da.attrs),
            provenance=list(validate_provenance(self.provenance)), coords=[],
        )
        # Caller names and attrs live in the envelope, outside xarray's storage
        # conventions. Internal variable names never determine the encoding.
        storage_dims = {dim: f"dim_{i}" for i, dim in enumerate(da.dims)}
        if is_sparse:
            coo = da.data
            variables = {
                "coo_coords": (("ndim", "nnz"), coo.coords),
                "coo_data": (("nnz",), coo.data),
            }
            metadata.update(shape=list(coo.shape), fill_value=_json_safe(coo.fill_value))
        else:
            variables = {"data": (tuple(storage_dims.values()), da.data)}

        for i, (name, coord) in enumerate(da.coords.items()):
            variable = f"coord_{i}"
            metadata["coords"].append({
                "name": name, "dims": list(coord.dims), "variable": variable,
                "attrs": _domain_attrs(coord.attrs),
            })
            variables[variable] = (
                tuple(storage_dims[d] for d in coord.dims), coord.values,
            )
        ds = xr.Dataset(variables, attrs={NAMESPACE: metadata})
        ds.to_zarr(path, mode="w")

    @classmethod
    def load(cls, path):
        path = cls._normalize(path)
        with xr.open_zarr(path) as ds:
            metadata = validate_metadata(
                ds.attrs.get(NAMESPACE), "LabeledArray", ("dense", "coo"),
            )
            provenance = validate_provenance(metadata.get("provenance"))
            if metadata["encoding"] == "dense":
                data = ds["data"].values
            else:
                import sparse

                data = sparse.COO(
                    coords=ds["coo_coords"].values,
                    data=ds["coo_data"].values,
                    shape=tuple(metadata["shape"]),
                    fill_value=metadata["fill_value"],
                )
            coords = {
                coord["name"]: (
                    tuple(coord["dims"]), ds[coord["variable"]].values,
                    _domain_attrs(coord["attrs"]),
                )
                for coord in metadata["coords"]
            }
            da = xr.DataArray(
                data, dims=metadata["dims"], coords=coords, name=metadata["name"],
                attrs=_domain_attrs(metadata["attrs"]),
            )
        return cls(da, provenance=provenance)  # validate via __post_init__

    def __repr__(self) -> str:
        return (
            f"LabeledArray(dims={list(self.da.dims)}, shape={self.da.shape}, "
            f"provenance={self.provenance})"
        )
