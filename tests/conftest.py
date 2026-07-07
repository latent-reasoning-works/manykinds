"""Zack Warren's kind test suite (originally from manylatents-omics#53). Skips
wholesale if the array stack is somehow absent."""
import pytest

pytest.importorskip("xarray")
pytest.importorskip("sparse")
