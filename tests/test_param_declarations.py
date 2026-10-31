"""Parameter declarations: the version-1 format, its two checks and the command line.

The format is defined by the prose in docs/param-declarations.md, by
``manykinds.params.check_declaration`` and by the fixtures in
``tests/fixtures/param_declarations/``. There is no schema file.
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from manykinds import params

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "param_declarations"
EXPECTED = FIXTURES / "expected.json"

ENTRY_KEYS = {
    "legal", "default", "sensible", "ladder", "limits", "role", "precomputed", "cost",
    "meaning", "more", "less",
}


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


# ---------------------------------------------------------------------------
# The module imports only the standard library
# ---------------------------------------------------------------------------

def test_import_loads_no_array_package_and_no_yaml(tmp_path):
    code = (
        "import json, sys\n"
        "import manykinds.params\n"
        "heavy = ('numpy', 'xarray', 'zarr', 'sparse', 'numba', 'yaml')\n"
        "print(json.dumps({'file': manykinds.params.__file__,"
        " 'loaded': sorted(m for m in heavy if m in sys.modules)}))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=_env(),
        cwd=str(tmp_path), check=True,
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert Path(result["file"]).resolve().is_relative_to(REPO / "manykinds")
    assert result["loaded"] == []


def test_package_import_does_not_load_params(tmp_path):
    code = "import sys, manykinds\nprint('manykinds.params' in sys.modules)\n"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=_env(),
        cwd=str(tmp_path), check=True,
    )
    assert out.stdout.strip() == "False"


def test_closed_sets_are_named():
    assert params.SCHEMA_VERSION == 1
    assert params.GROUPS == ("latent", "lightning", "tool")
    assert params.TYPES == ("int", "float", "bool", "choice", "str")
    assert params.ROLES == ("knob", "fixed", "infra")
    assert params.FACTS == ("n_samples", "n_features")
    assert params.SCALES == ("linear", "log")
    assert params.SOURCE_KINDS == ("docs", "paper", "measured")
    assert params.COST_DIRECTIONS == ("up", "down", "none")
    assert params.OPS == ("le", "lt")


def test_no_json_schema_file_ships():
    package = REPO / "manykinds"
    assert package.is_dir(), package
    assert FIXTURES.is_dir(), FIXTURES
    for root in (package, FIXTURES):
        assert not [p for p in root.rglob("*") if "schema" in p.name.lower()]


# ---------------------------------------------------------------------------
# The format: fixtures
# ---------------------------------------------------------------------------

def _fixtures(kind: str) -> list[Path]:
    root = FIXTURES / kind
    assert root.is_dir(), root
    files = sorted(root.glob("*.json"))
    assert files, root
    return files


def _load(path: Path):
    with path.open() as handle:
        return json.load(handle)


def _expected() -> dict[str, str]:
    return _load(EXPECTED)


@pytest.mark.parametrize("path", _fixtures("valid"), ids=lambda p: p.stem)
def test_valid_fixture_has_no_problem(path):
    assert params.check_declaration(_load(path)) == []


def test_one_valid_fixture_per_group():
    groups = sorted(_load(p)["group"] for p in _fixtures("valid"))
    assert groups == ["latent", "lightning", "tool"]


@pytest.mark.parametrize("path", _fixtures("invalid"), ids=lambda p: p.stem)
def test_invalid_fixture_gives_its_named_message(path):
    expected = _expected()[path.stem]
    problems = params.check_declaration(_load(path))
    assert any(p.startswith(expected) for p in problems), (expected, problems)


def test_every_invalid_fixture_names_its_message_and_path():
    stems = {p.stem for p in _fixtures("invalid")}
    expected = _expected()
    assert set(expected) == stems
    for stem, message in expected.items():
        path, sep, _ = message.partition(": ")
        assert sep and path.startswith("params."), (stem, message)


def test_invalid_fixtures_cover_every_rule():
    # One fixture per row of the table of refusals in docs/param-declarations.md.
    assert {
        "unknown_key", "missing_key", "bad_type", "default_refused", "sensible_log_lo",
        "limit_fact", "bad_role", "empty_meaning", "bad_cost",
    } <= {p.stem for p in _fixtures("invalid")}


def test_hand_written_json_tool_declaration_passes():
    doc = _load(FIXTURES / "valid" / "tool.json")
    assert doc["group"] == "tool"
    assert params.check_declaration(doc) == []


def _valid_doc() -> dict:
    return _load(FIXTURES / "valid" / "tool.json")


def _sensible(**overrides) -> dict:
    sensible = {"lo": 1, "hi": 5, "scale": "linear", "source": {"kind": "docs", "ref": "x"}}
    sensible.update(overrides)
    return sensible


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda d: d.update(schema_version=2), "schema_version: 2 is not 1"),
        (lambda d: d.update(group="model"), "group: 'model' is not one of latent, lightning, tool"),
        (lambda d: d.pop("origin"), "origin: missing key 'origin'"),
        (lambda d: d.update(extra=1), "extra: unknown key 'extra'"),
        (lambda d: d.update(method=""), "method: expected a non-empty text"),
        (lambda d: d["params"]["min_counts"].update(ladder=[100, -1]),
         "params.min_counts.ladder[1]: -1 is refused by legal"),
        (lambda d: d["params"]["min_counts"].update(sensible=_sensible(lo=-5, hi=10)),
         "params.min_counts.sensible.lo: -5 is refused by legal"),
        (lambda d: d["params"]["min_counts"].update(sensible=_sensible(lo=10, hi=5)),
         "params.min_counts.sensible: lo 10 is greater than hi 5"),
        (lambda d: d["params"]["min_counts"].update(sensible=_sensible(scale="exp")),
         "params.min_counts.sensible.scale: 'exp' is not one of linear, log"),
        (lambda d: d["params"]["min_counts"].update(
            sensible=_sensible(source={"kind": "guess", "ref": "x"})),
         "params.min_counts.sensible.source.kind: 'guess' is not one of docs, paper, measured"),
        (lambda d: d["params"]["min_counts"].update(precomputed=True),
         "params.min_counts.precomputed: precomputed is true but role is 'knob', not 'fixed'"),
        (lambda d: d["params"]["min_counts"].update(meaning="x" * 121),
         "params.min_counts.meaning: must be 1 to 120 characters"),
        (lambda d: d["params"]["min_counts"].update(more=""),
         "params.min_counts.more: must be null or 1 to 120 characters"),
        (lambda d: d["params"]["min_counts"].update(less="x" * 121),
         "params.min_counts.less: must be null or 1 to 120 characters"),
        (lambda d: d["params"]["min_counts"].update(
            cost={"time": "sideways", "memory": "none", "note": None}),
         "params.min_counts.cost.time: 'sideways' is not one of up, down, none"),
        (lambda d: d["params"]["min_counts"].update(
            cost={"time": "up", "memory": "slow", "note": None}),
         "params.min_counts.cost.memory: 'slow' is not one of up, down, none"),
        (lambda d: d["params"]["min_counts"].update(cost={"time": "up", "memory": "up"}),
         "params.min_counts.cost.note: missing key 'note'"),
        (lambda d: d["params"]["min_counts"].update(
            cost={"time": "up", "memory": "up", "note": None, "screen": "x"}),
         "params.min_counts.cost.screen: unknown key 'screen'"),
        (lambda d: d["params"]["min_counts"]["limits"].append(
            {"op": "ge", "fact": "n_samples", "offset": 0}),
         "params.min_counts.limits[0].op: 'ge' is not one of le, lt"),
        (lambda d: d["params"]["flavor"].update(default="pearson"),
         "params.flavor.default: 'pearson' is refused by legal"),
    ],
)
def test_check_declaration_refuses(mutate, expected):
    doc = _valid_doc()
    mutate(doc)
    problems = params.check_declaration(doc)
    assert any(p.startswith(expected) for p in problems), (expected, problems)


def test_check_declaration_reports_every_problem():
    doc = _valid_doc()
    doc["params"]["min_counts"]["role"] = "tunable"
    doc["params"]["flavor"]["meaning"] = ""
    assert len(params.check_declaration(doc)) == 2


def test_every_message_starts_with_a_dotted_field_path():
    paths = re.compile(r"^[A-Za-z_]\w*(\.\w+|\[\d+\])*: ")
    for path in _fixtures("invalid"):
        for message in params.check_declaration(_load(path)):
            assert paths.match(message), message


def test_check_declaration_of_a_non_mapping():
    assert params.check_declaration([]) == ["declaration: expected a mapping, got list"]


# ---------------------------------------------------------------------------
# check reads legal and limits only
# ---------------------------------------------------------------------------

def _knn_doc(**entry) -> dict:
    doc = _valid_doc()
    knn = copy.deepcopy(doc["params"]["min_counts"])
    knn["legal"]["min"] = 1
    knn["default"] = 5
    knn.update(entry)
    doc["params"]["knn"] = knn
    doc["params"]["n_jobs"] = {
        **copy.deepcopy(doc["params"]["min_counts"]),
        "default": 1, "ladder": None, "role": "infra", "meaning": "Number of parallel jobs.",
        "more": None, "less": None,
    }
    doc["params"]["n_jobs"]["legal"]["min"] = None
    assert params.check_declaration(doc) == []
    return doc


def test_check_accepts_a_value_off_the_ladder():
    doc = _knn_doc(ladder=[5, 15, 30])
    assert params.check(doc, {"knn": 7}) == []


def test_check_refuses_a_value_that_a_data_limit_refuses():
    doc = _knn_doc(limits=[{"op": "le", "fact": "n_samples", "offset": -2}])
    problems = params.check(doc, {"knn": 49}, n_samples=50)
    assert len(problems) == 1
    assert "knn" in problems[0] and "n_samples" in problems[0]
    assert params.check(doc, {"knn": 48}, n_samples=50) == []


def test_check_skips_a_limit_whose_fact_is_not_given():
    doc = _knn_doc(limits=[{"op": "le", "fact": "n_samples", "offset": -2}])
    assert params.check(doc, {"knn": 100}) == []


def test_check_refuses_an_infra_parameter_by_role():
    problems = params.check(_knn_doc(), {"n_jobs": 4})
    assert len(problems) == 1
    assert "n_jobs" in problems[0] and "infra" in problems[0]


def test_check_refuses_a_fixed_parameter_by_role():
    problems = params.check(_valid_doc(), {"n_top_genes": 100})
    assert len(problems) == 1
    assert "n_top_genes" in problems[0] and "fixed" in problems[0]


def test_check_refuses_illegal_and_unknown_values():
    doc = _valid_doc()
    assert params.check(doc, {"min_counts": 0, "flavor": "cell_ranger"}) == []
    problems = params.check(doc, {"min_counts": -1, "flavor": "pearson", "nope": 1})
    assert len(problems) == 3
    assert any(p.startswith("min_counts=-1") for p in problems)
    assert any("nope" in p for p in problems)


def test_check_accepts_a_value_listed_in_also():
    doc = _knn_doc()
    doc["params"]["knn"]["legal"]["also"] = ["auto"]
    assert params.check(doc, {"knn": "auto"}) == []
    assert len(params.check(doc, {"knn": "other"})) == 1


def test_check_does_not_read_sensible_or_ladder():
    doc = _valid_doc()
    doc["params"]["min_counts"]["sensible"] = {
        "lo": 100, "hi": 500, "scale": "linear", "source": {"kind": "measured", "ref": "x"},
    }
    assert params.check(doc, {"min_counts": 7}) == []


def test_check_applies_n_features_limits():
    doc = _valid_doc()
    doc["params"]["n_top_genes"]["role"] = "knob"
    doc["params"]["n_top_genes"]["precomputed"] = False
    assert params.check(doc, {"n_top_genes": 30}, n_features=30) == []
    problems = params.check(doc, {"n_top_genes": 31}, n_features=30)
    assert len(problems) == 1 and "n_features" in problems[0]


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

def _cli(*args: str, without_yaml: bool = False) -> subprocess.CompletedProcess:
    if without_yaml:
        code = (
            "import runpy, sys\n"
            "sys.modules['yaml'] = None\n"
            f"sys.argv = ['manykinds.params', *{list(args)!r}]\n"
            "runpy.run_module('manykinds.params', run_name='__main__', alter_sys=True)\n"
        )
        command = [sys.executable, "-c", code]
    else:
        command = [sys.executable, "-m", "manykinds.params", *args]
    return subprocess.run(
        command, capture_output=True, text=True, env=_env(), cwd=str(REPO),
        stdin=subprocess.DEVNULL, check=False,
    )


def test_cli_check_names_the_field(tmp_path):
    doc = _knn_doc()
    doc["params"]["knn"]["default"] = 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    out = _cli("check", str(bad))
    assert out.returncode == 1, out.stderr
    result = json.loads(out.stdout)
    assert result["ok"] is False
    assert any(p["path"] == "params.knn.default" for p in result["problems"]), result
    assert all(set(p) == {"path", "message"} for p in result["problems"])


def test_cli_check_valid_json_file_exits_0(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_valid_doc()))
    out = _cli("check", str(good))
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == {"ok": True, "problems": []}


@pytest.mark.parametrize("path", _fixtures("invalid"), ids=lambda p: p.stem)
def test_cli_check_invalid_fixture_exits_1(path):
    out = _cli("check", str(path))
    assert out.returncode == 1, out.stderr
    expected_path = _expected()[path.stem].partition(": ")[0]
    assert any(p["path"] == expected_path for p in json.loads(out.stdout)["problems"])


@pytest.mark.parametrize("content", [None, "{not json", "\xff"])
def test_cli_check_unreadable_file_exits_2(tmp_path, content):
    path = tmp_path / "file.json"
    if content == "\xff":
        path.write_bytes(b"\xff\xfe\x00")
    elif content is not None:
        path.write_text(content)
    out = _cli("check", str(path))
    assert out.returncode == 2
    result = json.loads(out.stdout)
    assert result["ok"] is False and result["problems"]


def test_cli_check_readable_non_mapping_exits_1(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2]")
    out = _cli("check", str(path))
    assert out.returncode == 1
    assert json.loads(out.stdout)["problems"] == [
        {"path": "declaration", "message": "expected a mapping, got list"}
    ]


def test_cli_check_yaml_without_pyyaml_exits_2_and_names_pyyaml(tmp_path):
    path = tmp_path / "decl.yaml"
    path.write_text("schema_version: 1\n")
    out = _cli("check", str(path), without_yaml=True)
    assert out.returncode == 2, (out.stdout, out.stderr)
    assert "PyYAML" in out.stderr
    assert "pip" not in out.stderr
    assert json.loads(out.stdout)["ok"] is False


def test_cli_check_yaml_with_pyyaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(_valid_doc()))
    out = _cli("check", str(good))
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == {"ok": True, "problems": []}
    broken = tmp_path / "broken.yml"
    broken.write_text("{not: [valid")
    assert _cli("check", str(broken)).returncode == 2


def test_cli_prints_exactly_one_json_object(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_valid_doc()))
    out = _cli("check", str(good))
    assert len(out.stdout.strip().splitlines()) == 1
