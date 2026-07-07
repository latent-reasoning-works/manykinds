"""FileArtifact, Table, Sequence, and weighted SparseGraph."""
import numpy as np
import pandas as pd
import pytest

from manykinds import Kind, KindSpec, FileArtifact, Table, Sequence, SparseGraph


# --------------------------------------------------------------------------- #
# SparseGraph edge_weights
# --------------------------------------------------------------------------- #


def test_weighted_graph_conforms_and_validates():
    g = SparseGraph(np.array([[0, 1], [1, 2]]), np.array([0, 1, 2]),
                    edge_weights=np.array([0.5, -1.0]))
    assert isinstance(g, Kind)
    assert g.spec() == KindSpec("SparseGraph", ("edges", "node_ids", "edge_weights"), ())
    assert g.require("edge_weights") is g  # a weight-consuming op can require it


def test_unweighted_graph_lacks_weight_component():
    g = SparseGraph(np.array([[0, 1]]), np.array([0, 1]))
    assert g.spec().dims == ("edges", "node_ids")
    with pytest.raises(ValueError, match="requires dims"):
        g.require("edge_weights")


def test_weight_length_must_match_edges():
    with pytest.raises(ValueError, match="one weight per edge"):
        SparseGraph(np.array([[0, 1], [1, 2]]), np.array([0, 1, 2]),
                    edge_weights=np.array([0.5]))


def test_weighted_graph_round_trip(tmp_path):
    g = SparseGraph(np.array([[0, 1], [1, 2]]), np.array([0, 1, 2]),
                    edge_weights=np.array([0.5, -1.0])).tagged("grn")
    p = str(tmp_path / "g.npz")
    g.serialize(p)
    r = SparseGraph.load(p)
    assert r.provenance == ("grn",)
    assert np.allclose(r.edge_weights, [0.5, -1.0])


# --------------------------------------------------------------------------- #
# FileArtifact
# --------------------------------------------------------------------------- #


def _artifact_file(tmp_path):
    f = tmp_path / "data.h5ad"
    f.write_bytes(b"fake h5ad bytes")
    return str(f)


def test_file_artifact_conforms_and_infers_format(tmp_path):
    a = FileArtifact(_artifact_file(tmp_path))
    assert isinstance(a, Kind)
    assert a.format == "h5ad"
    assert a.spec() == KindSpec("FileArtifact", (), ())


def test_file_artifact_is_opaque(tmp_path):
    a = FileArtifact(_artifact_file(tmp_path))
    with pytest.raises(ValueError, match="opaque"):
        a.require("cell")


def test_file_artifact_missing_file_rejected():
    with pytest.raises(ValueError, match="not a file"):
        FileArtifact("/no/such/file.csv")


def test_file_artifact_digest_and_manifest_round_trip(tmp_path):
    a = FileArtifact(_artifact_file(tmp_path)).with_digest().tagged("download")
    assert len(a.checksum) == 64  # sha256 hex
    mp = str(tmp_path / "manifest.json")
    a.serialize(mp)
    r = FileArtifact.load(mp)
    assert r.checksum == a.checksum and r.provenance == ("download",)


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #


def _toy_df():
    return pd.DataFrame({"cluster": [0, 1, 1], "label": ["a", "b", "c"],
                         "score": [0.1, 0.9, 0.5]})


def test_table_conforms_and_specs_columns():
    t = Table(_toy_df())
    assert isinstance(t, Kind)
    assert set(t.spec().dims) == {"cluster", "label", "score"}
    assert t.require("cluster", "label") is t


def test_table_require_missing_column():
    with pytest.raises(ValueError, match="requires columns"):
        Table(_toy_df()).require("nope")


def test_table_parquet_round_trip_preserves_dtypes(tmp_path):
    t = Table(_toy_df()).tagged("cluster_summary")
    p = str(tmp_path / "t.parquet")
    t.serialize(p)
    r = Table.load(p)
    assert r.provenance == ("cluster_summary",)
    assert list(r.df.columns) == ["cluster", "label", "score"]
    assert r.df["score"].dtype == np.float64 and r.df["cluster"].dtype == np.int64


# --------------------------------------------------------------------------- #
# Sequence
# --------------------------------------------------------------------------- #


def test_sequence_conforms_and_validates_alphabet():
    s = Sequence(np.array(["ACGT", "TTGG"]), alphabet="ACGT")
    assert isinstance(s, Kind)
    assert s.spec() == KindSpec("Sequence", ("sequence",), ())


def test_sequence_rejects_out_of_alphabet():
    with pytest.raises(ValueError, match="outside alphabet"):
        Sequence(np.array(["ACGX"]), alphabet="ACGT")


def test_sequence_with_ids_carries_id_coord():
    s = Sequence(np.array(["ACGT", "TTGG"]), alphabet="ACGT", ids=np.array(["s1", "s2"]))
    assert s.spec().coords == ("id",)
    assert s.require("sequence", coords=("id",)) is s


def test_sequence_round_trip(tmp_path):
    s = Sequence(np.array(["ACGT", "TTGG"]), alphabet="ACGT",
                 ids=np.array(["s1", "s2"])).tagged("tokenize")
    p = str(tmp_path / "s.npz")
    s.serialize(p)
    r = Sequence.load(p)
    assert r.provenance == ("tokenize",) and r.alphabet == "ACGT"
    assert list(r.sequences) == ["ACGT", "TTGG"] and list(r.ids) == ["s1", "s2"]
