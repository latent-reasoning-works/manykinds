"""Persistence separates caller data from the versioned storage envelope."""
from pathlib import Path

import numpy as np
import pytest
import sparse
import xarray as xr

from manykinds import LabeledArray, Sequence, SparseGraph


@pytest.fixture(params=["dense", "sparse"])
def labeled(request):
    data = np.array([[1, 0], [0, 2]])
    if request.param == "sparse":
        data = sparse.COO.from_numpy(data)
    return xr.DataArray(data, dims=("cell", "gene"), name="counts")


@pytest.mark.parametrize("trail", [(), ("normalize", "embed")])
def test_domain_provenance_and_trail_are_independent(labeled, trail, tmp_path):
    labeled.attrs = {"provenance": "source", "genome": "GRCh38"}
    kind = LabeledArray(labeled, trail)
    path = tmp_path / "array.zarr"
    kind.serialize(path)
    loaded = LabeledArray.load(path)
    assert loaded.da.attrs == labeled.attrs
    assert loaded.da.attrs["provenance"] == "source"
    assert loaded.provenance == trail
    assert kind.da.attrs["provenance"] == "source"
    assert kind.provenance == trail


def test_dense_array_named_coo_data(tmp_path):
    original = xr.DataArray([1, 2], dims="cell", name="coo_data")
    path = tmp_path / "array.zarr"
    LabeledArray(original).serialize(path)
    loaded = LabeledArray.load(path)
    xr.testing.assert_identical(loaded.da, original)
    assert isinstance(loaded.da.data, np.ndarray)


def test_coordinate_attrs_round_trip(labeled, tmp_path):
    labeled = labeled.assign_coords(
        cell=("cell", [0, 1], {"axis": "sample"}),
        time=("cell", [1.0, 2.0], {"units": "s"}),
        batch=((), "a", {"source": "lab"}),
        distance=(("cell", "gene"), [[1, 2], [3, 4]], {"units": "m"}),
    )
    path = tmp_path / "array.zarr"
    LabeledArray(labeled).serialize(path)
    loaded = LabeledArray.load(path)
    xr.testing.assert_identical(loaded.da, labeled)


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.bytes_(b"GRCh38"), "GRCh38"),
        ({"count": np.int64(2), "flags": [np.bool_(True), {"tag": np.bytes_(b"ok")}]},
         {"count": 2, "flags": [True, {"tag": "ok"}]}),
    ],
)
def test_recursive_numpy_attrs(labeled, value, expected, tmp_path):
    labeled.attrs["metadata"] = value
    path = tmp_path / "array.zarr"
    LabeledArray(labeled).serialize(path)
    assert LabeledArray.load(path).da.attrs["metadata"] == expected


@pytest.fixture(params=[SparseGraph, Sequence])
def npz_kind(request):
    if request.param is SparseGraph:
        return SparseGraph([[0, 1]], ["a", "b"], ("build",))
    return Sequence(["AC", "GT"], "ACGT", ["a", "b"], ("build",))


def test_npz_contains_no_object_arrays(npz_kind, tmp_path):
    path = tmp_path / "kind.npz"
    npz_kind.serialize(path)
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            assert not archive[name].dtype.hasobject
    assert type(npz_kind).load(path).provenance == ("build",)


class _Payload:
    """If unpickled, leave a marker proving code execution."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return Path.write_text, (self.marker, "executed")


@pytest.mark.parametrize("payload_field", ["provenance", "ids"])
def test_npz_refuses_pickle_payload(npz_kind, payload_field, tmp_path):
    path = tmp_path / "kind.npz"
    marker = tmp_path / "payload-executed"
    npz_kind.serialize(path)
    # Keep the writer's header so a current-format archive is also tested.
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "provenance"}
    arrays["provenance"] = np.asarray(["build"])
    field = "node_ids" if payload_field == "ids" and isinstance(npz_kind, SparseGraph) else payload_field
    arrays[field] = np.asarray([_Payload(marker)], dtype=object)
    np.savez_compressed(path, **arrays)
    try:
        with pytest.raises(ValueError):
            type(npz_kind).load(path)
    finally:
        assert not marker.exists(), "loading the archive executed its pickle payload"


@pytest.mark.parametrize("location", ["array", "coord"])
def test_reserved_attrs_rejected_before_writing(labeled, location, tmp_path):
    if location == "array":
        labeled.attrs["_manykinds"] = {"caller": True}
    else:
        labeled = labeled.assign_coords(time=("cell", [1, 2], {"_manykinds": "caller"}))
    path = tmp_path / "array.zarr"
    with pytest.raises(ValueError, match="reserved"):
        LabeledArray(labeled).serialize(path)
    assert not path.exists()


def test_storage_names_and_attrs_are_caller_data(labeled, tmp_path):
    labeled = labeled.rename({"cell": "nnz", "gene": "coo_coords"})
    labeled.name = ""
    labeled.attrs = {"kind": "custom", "version": 99, "encoding": "custom",
                     "coordinates": "domain", "shape": "custom"}
    labeled = labeled.assign_coords(
        coo_data=("nnz", [1, 2], {"provenance": "clock", "units": np.bytes_(b"s"),
                                   "metadata": {"count": np.int64(2)}}),
    )
    path = tmp_path / "array.zarr"
    LabeledArray(labeled, ("op",)).serialize(path)
    loaded = LabeledArray.load(path)
    assert loaded.provenance == ("op",)
    assert loaded.da.attrs == labeled.attrs
    assert "provenance" not in loaded.da.attrs
    assert loaded.da.name == ""
    assert loaded.da.dims == labeled.dims
    assert loaded.da.coords["coo_data"].attrs == {
        "provenance": "clock", "units": "s", "metadata": {"count": 2},
    }
    with xr.open_zarr(path) as stored:
        metadata = stored.attrs["_manykinds"]
        assert metadata["format"] == "manykinds"
        assert metadata["version"] == 1
        assert metadata["kind"] == "LabeledArray"
        assert metadata["encoding"] == ("coo" if isinstance(labeled.data, sparse.COO) else "dense")


@pytest.mark.parametrize(
    "field, value, error",
    [("format", "other", "format marker"), ("version", 99, "version"),
     ("version", True, "version"), ("kind", "Sequence", "kind"),
     ("encoding", "other", "encoding")],
)
def test_zarr_rejects_unsupported_identity(tmp_path, field, value, error):
    metadata = {"format": "manykinds", "version": 1, "kind": "LabeledArray", "encoding": "dense"}
    metadata[field] = value
    path = tmp_path / "array.zarr"
    xr.Dataset(attrs={"_manykinds": metadata}).to_zarr(path)
    with pytest.raises(ValueError, match=error):
        LabeledArray.load(path)


@pytest.mark.parametrize("encoding", ["dense", "coo"])
def test_zarr_rejects_unversioned_archives(tmp_path, encoding):
    path = tmp_path / "legacy.zarr"
    if encoding == "dense":
        xr.DataArray([1, 2], attrs={"provenance": "source"}).to_zarr(path)
    else:
        xr.Dataset({"coo_data": ("nnz", [1]), "coo_coords": (("ndim", "nnz"), [[0]])},
                   attrs={"shape": [2], "dims": ["cell"], "fill_value": 0,
                          "name": "", "da_attrs": {}, "provenance": [], "coord_dims": {}}).to_zarr(path)
    with pytest.raises(ValueError, match="unversioned legacy"):
        LabeledArray.load(path)


@pytest.mark.parametrize("legacy_trail", [None, "unicode", "object", "payload"])
def test_npz_rejects_unversioned_archives(npz_kind, tmp_path, legacy_trail):
    path = tmp_path / "legacy.npz"
    marker = tmp_path / "payload-executed"
    npz_kind.serialize(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "_manykinds"}
    if legacy_trail is None:
        del arrays["provenance"]
    elif legacy_trail in ("object", "payload"):
        value = "build" if legacy_trail == "object" else _Payload(marker)
        arrays["provenance"] = np.asarray([value], dtype=object)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="unversioned legacy"):
        type(npz_kind).load(path)
    assert not marker.exists()


@pytest.mark.parametrize("dtype", ["bool", "int64", "uint32", "float64", "complex128", "S4", "U4"])
def test_npz_supported_ids_preserve_dtype(npz_kind, tmp_path, dtype):
    ids = np.asarray([0, 1], dtype=dtype)
    if isinstance(npz_kind, SparseGraph):
        kind = SparseGraph(npz_kind.edges, ids)
        id_name = "node_ids"
    else:
        kind = Sequence(npz_kind.sequences, npz_kind.alphabet, ids)
        id_name = "ids"
    path = tmp_path / "kind.npz"
    kind.serialize(path)
    loaded = type(kind).load(path)
    assert loaded.provenance == ()
    actual = getattr(loaded, id_name)
    np.testing.assert_array_equal(actual, ids)
    assert actual.dtype == ids.dtype


@pytest.mark.parametrize("dtype", [object, "datetime64[D]"])
def test_npz_rejects_unsupported_ids_on_write_and_read(npz_kind, tmp_path, dtype):
    ids = np.asarray([0, 1], dtype=dtype)
    id_name = "node_ids" if isinstance(npz_kind, SparseGraph) else "ids"
    path = tmp_path / "kind.npz"
    npz_kind.serialize(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays[id_name] = ids
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        type(npz_kind).load(path)
    if isinstance(npz_kind, SparseGraph):
        kind = SparseGraph(npz_kind.edges, ids)
    else:
        kind = Sequence(npz_kind.sequences, npz_kind.alphabet, ids)
    target = tmp_path / "invalid.npz"
    with pytest.raises(ValueError, match="dtype"):
        kind.serialize(target)
    assert not target.exists()


@pytest.mark.parametrize("field, value, error", [
    ("format", "other", "format marker"), ("version", 2, "version"),
    ("kind", "LabeledArray", "kind"), ("encoding", "dense", "encoding"),
])
def test_npz_rejects_unsupported_identity(npz_kind, tmp_path, field, value, error):
    import json

    path = tmp_path / "kind.npz"
    npz_kind.serialize(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(arrays["_manykinds"].item())
    assert metadata == {"format": "manykinds", "version": 1,
                        "kind": type(npz_kind).__name__, "encoding": "npz"}
    metadata[field] = value
    arrays["_manykinds"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match=error):
        type(npz_kind).load(path)
