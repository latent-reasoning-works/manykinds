# Parameter declarations

A parameter declaration says which values each parameter of a method or a tool accepts, in
plain data that any program can read. A caller that offers a method's parameters reads the
declaration instead of copying each library's checks. A declaration is JSON or YAML, and no
Python type is needed to write or read it.

The contract lives in `manykinds.params`. It imports only the standard library, so a tool author
checks a manifest without an array package or any method code.

The definition of the format is this page, the function `manykinds.params.check_declaration`
and the fixtures in `tests/fixtures/param_declarations/`. There is no schema file.

## Terms

| Term | Definition |
|---|---|
| declaration | The plain-data document, version 1, that describes the parameters of one method or one tool. |
| parameter entry | The part of a declaration for one parameter. Its keys are in the second table below. |
| producer | Any source of declarations: a method library that generates them, a tool manifest, a local tool, another lab's library. |
| knob | A parameter whose `role` is `knob`. A caller may set it. |
| data fact | One of the two names a data limit may use: `n_samples` and `n_features`. |

## The declaration

A declaration, version 1, is a mapping with exactly these keys:

| Key | Value |
|---|---|
| `schema_version` | `1` |
| `method` | the method or tool name, a non-empty text |
| `group` | `latent`, `lightning` or `tool` |
| `origin` | the producer, a non-empty text |
| `params` | a mapping of parameter name to parameter entry |

A parameter entry has exactly these keys:

| Key | Value |
|---|---|
| `legal` | `{"type", "min", "min_open", "max", "max_open", "choices", "also"}`. `type` is `int`, `float`, `bool`, `choice` or `str`. Bounds are numbers or `null`; `min_open: true` excludes the bound. `choices` is the list for `choice` (and for a `str` with named options), else `[]`. `also` lists other legal values, such as `["auto"]` or `[null]`. This is what is checked. |
| `default` | the constructor default, the config value, or the manifest value |
| `sensible` | `null`, or `{"lo", "hi", "scale", "source"}`: the range where the output changes, on `scale` `linear` or `log`, with `source` `{"kind": "docs" or "paper" or "measured", "ref": <text>}`. Never invented: `null` when no source says where the output changes. |
| `ladder` | `null`, or a non-empty list of preset values, for example `[5, 15, 30]` for `knn` |
| `limits` | a list of `{"op": "le" or "lt", "fact": "n_samples" or "n_features", "offset": <int>}`. Each means `value <op> fact + offset`. Only for `int` and `float`. |
| `role` | `knob`, a value a caller may set; `fixed`, a value that is part of the result and is recorded, such as a seed or a precomputed value; or `infra`, a value that changes how a fit runs but not what it computes, such as `n_jobs`, `verbose`, `device` or `backend`. Only a `knob` is offered. |
| `precomputed` | `true` when the value was recorded from a precomputation; then `role` is `fixed` |
| `cost` | `null`, or `{"time", "memory", "note"}`: how a larger value moves compute. `time` and `memory` are each `up`, `down` or `none`; `note` is text or `null`. It describes compute only. |
| `meaning` | what the parameter controls, 1 to 120 characters |
| `more`, `less` | what a larger and a smaller value does, 1 to 120 characters, or `null` |

The closed sets are constants of `manykinds.params`, so a producer names them instead of
copying them: `SCHEMA_VERSION`, `GROUPS`, `TYPES`, `ROLES`, `FACTS`, `OPS`, `SCALES`,
`SOURCE_KINDS` and `COST_DIRECTIONS`.

A small declaration for a tool, written by hand:

```json
{
  "schema_version": 1,
  "method": "count_matrix_filter",
  "group": "tool",
  "origin": "an-example-lab",
  "params": {
    "min_counts": {
      "legal": {"type": "int", "min": 0, "min_open": false, "max": null, "max_open": false,
                "choices": [], "also": []},
      "default": 200,
      "sensible": null,
      "ladder": [100, 200, 500],
      "limits": [],
      "role": "knob",
      "precomputed": false,
      "cost": null,
      "meaning": "Smallest total count that a cell must have to be kept.",
      "more": "Fewer cells are kept.",
      "less": "More cells are kept."
    }
  }
}
```

## The two checks

| Function | Returns |
|---|---|
| `check_declaration(doc) -> list[str]` | one message per problem, `[]` for a valid document. Each message starts with the dotted path of its field and `: `, for example `params.knn.default: 0 is refused by legal (below min 1)`. |
| `check(declaration, values, *, n_samples=None, n_features=None) -> list[str]` | one message per value that the declaration refuses: an unknown parameter, a role other than `knob` (the message names the role), a value that `legal` refuses, or a value that a limit refuses when its fact is given. |

`check_declaration` refuses:

| Problem | Example |
|---|---|
| an unknown or a missing key | an entry without `role` |
| a `legal.type` outside `TYPES` | `"integer"` |
| a `default`, a `ladder` value or a `sensible` bound that `legal` refuses | `knn` with `default: 0` and `legal.min: 1` |
| `sensible.lo > sensible.hi`, a `scale` outside `SCALES`, `log` with `lo <= 0`, or a `source.kind` outside `SOURCE_KINDS` | `{"lo": 0, "hi": 10, "scale": "log"}` |
| a `limits` fact outside `FACTS`, or an `op` other than `le` or `lt` | `"n_genes"` |
| a `role` outside `ROLES`; or `precomputed: true` with a role other than `fixed` | `"tunable"` |
| a `cost` with a `time` or `memory` outside `COST_DIRECTIONS`, or another key | `{"direction": "up"}` |
| a `meaning` that is empty or longer than 120 characters; a `more` or `less` that is neither `null` nor 1 to 120 characters | `""` |

It also refuses the shape errors that these rules need: a value of the wrong type, `min` greater
than `max`, bounds on a type that is not `int` or `float`, and `choices` on a type that is not
`choice` or `str`.

`check` reads `legal` and `limits` only. `ladder` and `sensible` describe; they never restrict a
value, so any legal value may be submitted. They are for studies, front ends and coarse
policies. A limit is checked only when its fact is given.

```python
import json
from manykinds import params

with open("count_matrix_filter.json") as handle:
    declaration = json.load(handle)

params.check_declaration(declaration)              # [] when the format is followed
params.check(declaration, {"min_counts": 7})       # [] : off the ladder, inside legal
params.check(declaration, {"min_counts": -1})
# ['min_counts=-1: refused by legal (below min 0)']
```

## Command line

```sh
python -m manykinds.params check FILE
```

It prints exactly one JSON object on stdout, `{"ok": bool, "problems": [{"path", "message"}]}`.
The command line splits each message of `check_declaration` into its `path` and the rest.

| Exit | Meaning |
|---|---|
| 0 | the declaration is valid |
| 1 | the declaration has problems |
| 2 | the file is unreadable |

A file named `*.json`, or any name other than `*.yaml` and `*.yml`, is read as JSON with the
standard library. A `*.yaml` or `*.yml` file is read only when PyYAML is importable; otherwise
the command exits 2 and names PyYAML on stderr. PyYAML is not a dependency of manykinds. To
read YAML, add it to your project:

```sh
uv add pyyaml
```

Or run the check once with PyYAML:

```sh
uv run --with pyyaml python -m manykinds.params check FILE
```

## Versions

A change of the format is a new `schema_version`. Version 1 documents stay valid.
