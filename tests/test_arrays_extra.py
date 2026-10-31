"""The array packages are an extra (``manykinds[arrays]``).

``Kind``, ``KindSpec`` and ``manykinds.params`` import with no extra. An array kind
loads on first access by trying its real import; only when that import fails does
``manykinds`` raise an ``ImportError`` that names the kind and the extra.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ARRAY_PACKAGES = ("numpy", "xarray", "zarr", "sparse", "numba")

# The core dependency list that manykinds 0.1.1 installed.
CORE_0_1_1 = [
    "numpy>=1.23",
    "xarray>=2024.0",
    "zarr>=2.18",
    "sparse>=0.15",
    "numba>=0.60",
]


def _run_blocked(blocked: tuple[str, ...], body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter in which ``blocked`` cannot be imported."""
    assert REPO_ROOT.is_dir()
    prelude = "import sys\n" + "".join(f"sys.modules[{name!r}] = None\n" for name in blocked)
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(body)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        cwd=REPO_ROOT,
    )


def test_light_install():
    result = _run_blocked(
        ARRAY_PACKAGES,
        """
        import manykinds
        import manykinds.params
        assert manykinds.Kind is not None
        assert manykinds.KindSpec is not None
        for name in ("numpy", "xarray", "zarr", "sparse", "numba"):
            assert sys.modules.get(name) is None, name
        print("ok")
        """,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_missing_extra_is_named():
    result = _run_blocked(
        ARRAY_PACKAGES,
        """
        import manykinds
        try:
            manykinds.LabeledArray
        except ImportError as exc:
            assert exc.__cause__ is not None, "not chained from the original error"
            print(str(exc))
        else:
            raise SystemExit("no ImportError")
        """,
    )
    assert result.returncode == 0, result.stderr
    message = result.stdout
    assert "LabeledArray" in message
    assert "manykinds[arrays]" in message
    assert 'uv add "manykinds[arrays]"' in message
    assert "pip" not in message


def test_kind_with_its_packages_imports_without_extra():
    result = _run_blocked(
        ("xarray", "zarr", "sparse", "numba"),
        """
        from manykinds import SparseGraph
        assert SparseGraph.__name__ == "SparseGraph"
        print("ok")
        """,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_extra_restores_0_1_1():
    pyproject = REPO_ROOT / "pyproject.toml"
    assert pyproject.is_file()
    project = tomllib.loads(pyproject.read_text())["project"]
    assert project["dependencies"] == []
    extras = project["optional-dependencies"]
    assert extras["arrays"] == CORE_0_1_1
    assert set(CORE_0_1_1) <= set(extras["table"])
    assert "pyarrow>=14" in extras["table"]
    assert set(CORE_0_1_1) <= set(extras["dev"])


def test_array_kinds_import_in_dev_env():
    from manykinds import LabeledArray, SparseGraph

    assert LabeledArray.__name__ == "LabeledArray"
    assert SparseGraph.__name__ == "SparseGraph"
