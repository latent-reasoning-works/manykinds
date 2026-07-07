"""KindSpec — data-free structural signatures + the plan-time satisfies check."""
import numpy as np
import xarray as xr

from manykinds import KindSpec, LabeledArray, SparseGraph


# --------------------------------------------------------------------------- #
# satisfies(): the plan-time mirror of Kind.require
# --------------------------------------------------------------------------- #


def test_exact_match_satisfies():
    s = KindSpec("LabeledArray", ("cell", "gene"), ("time",))
    assert s.satisfies(KindSpec("LabeledArray", ("cell", "gene"), ("time",)))


def test_superset_satisfies_subset_requirement():
    # a producer carrying extra structure can feed a consumer that needs less
    producer = KindSpec("LabeledArray", ("cell", "gene"), ("time", "gene_ids"))
    assert producer.satisfies(KindSpec("LabeledArray", ("cell", "gene"), ("time",)))


def test_missing_dim_does_not_satisfy():
    producer = KindSpec("LabeledArray", ("cell",))
    assert not producer.satisfies(KindSpec("LabeledArray", ("cell", "gene")))


def test_missing_coord_does_not_satisfy():
    producer = KindSpec("LabeledArray", ("cell", "gene"))
    assert not producer.satisfies(KindSpec("LabeledArray", ("cell", "gene"), ("time",)))


def test_different_kind_does_not_satisfy():
    la = KindSpec("LabeledArray", ("cell", "gene"))
    assert not la.satisfies(KindSpec("SparseGraph", ("edges", "node_ids")))


def test_kindspec_is_hashable_and_frozen():
    s = KindSpec("LabeledArray", ("cell",))
    assert {s, s} == {s}  # frozen dataclass → hashable


# --------------------------------------------------------------------------- #
# .spec(): a concrete kind reports its own signature, consistent with require
# --------------------------------------------------------------------------- #


def _toy_la():
    return LabeledArray(
        xr.DataArray(
            np.zeros((3, 2)),
            dims=("cell", "gene"),
            coords={"time": ("cell", np.array([0, 0, 1]))},
        )
    )


def test_labeled_array_spec_reflects_dims_and_coords():
    spec = _toy_la().spec()
    assert spec.kind == "LabeledArray"
    assert spec.dims == ("cell", "gene")
    assert "time" in spec.coords


def test_sparse_graph_spec_has_components_no_coords():
    spec = SparseGraph(np.array([[0, 1], [1, 2]]), np.array([0, 1, 2])).spec()
    assert spec.kind == "SparseGraph"
    assert spec.dims == ("edges", "node_ids")
    assert spec.coords == ()


def test_spec_agrees_with_require():
    # anything require() accepts, the derived spec should satisfy — one vocabulary
    la = _toy_la()
    need = KindSpec("LabeledArray", ("cell", "gene"), ("time",))
    la.require("cell", "gene", coords=("time",))  # runtime: passes
    assert la.spec().satisfies(need)  # plan-time: agrees


def test_spec_agrees_with_require_on_rejection():
    la = _toy_la()  # has no "latent" dim
    need = KindSpec("LabeledArray", ("cell", "latent"))
    assert not la.spec().satisfies(need)  # plan-time rejects, as require would
