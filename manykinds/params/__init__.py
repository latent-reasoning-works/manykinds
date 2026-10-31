"""Parameter declarations: which values each parameter of a method or tool accepts, as plain data.

A declaration (version 1) is a mapping::

    {"schema_version": 1, "method": <name>, "group": "latent" | "lightning" | "tool",
     "origin": <producer>, "params": {<name>: <parameter entry>}}

and a parameter entry has exactly the keys ``legal``, ``default``, ``sensible``, ``ladder``,
``limits``, ``role``, ``precomputed``, ``cost``, ``meaning``, ``more`` and ``less``. The
format is defined by the prose in docs/param-declarations.md, by :func:`check_declaration` and
by the fixtures in ``tests/fixtures/param_declarations/``. There is no schema file.

Two checks:

- :func:`check_declaration` returns the problems of a declaration from any producer. Each
  message starts with the dotted path of its field and ``": "``.
- :func:`check` returns one message per submitted value that a declaration refuses. It reads
  ``legal`` and ``limits`` only; ``sensible`` and ``ladder`` describe and never restrict.

This module imports only the standard library, so a caller that only checks declarations loads
no array package. ``python -m manykinds.params check FILE`` is the command line.
"""
from __future__ import annotations

import math
import numbers
from typing import Any

__all__ = [
    "COST_DIRECTIONS",
    "FACTS",
    "GROUPS",
    "OPS",
    "ROLES",
    "SCALES",
    "SCHEMA_VERSION",
    "SOURCE_KINDS",
    "TYPES",
    "check",
    "check_declaration",
]

SCHEMA_VERSION = 1
GROUPS = ("latent", "lightning", "tool")
TYPES = ("int", "float", "bool", "choice", "str")
ROLES = ("knob", "fixed", "infra")
SCALES = ("linear", "log")
SOURCE_KINDS = ("docs", "paper", "measured")
FACTS = ("n_samples", "n_features")
OPS = ("le", "lt")
COST_DIRECTIONS = ("up", "down", "none")
TEXT_MAX = 120

DOC_KEYS = ("schema_version", "method", "group", "origin", "params")
ENTRY_KEYS = (
    "legal", "default", "sensible", "ladder", "limits", "role", "precomputed", "cost",
    "meaning", "more", "less",
)
LEGAL_KEYS = ("type", "min", "min_open", "max", "max_open", "choices", "also")
SENSIBLE_KEYS = ("lo", "hi", "scale", "source")
SOURCE_REF_KEYS = ("kind", "ref")
LIMIT_KEYS = ("op", "fact", "offset")
COST_KEYS = ("time", "memory", "note")


# ---------------------------------------------------------------------------
# Values against legal
# ---------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _same(a: Any, b: Any) -> bool:
    """Equality that keeps bools, numbers, strings and None apart."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if a is None or b is None:
        return a is None and b is None
    if _is_number(a) and _is_number(b):
        return a == b
    return type(a) is type(b) and a == b


def _contains(items: Any, value: Any) -> bool:
    return isinstance(items, list) and any(_same(value, item) for item in items)


def _legal_refusal(legal: dict, value: Any) -> str | None:
    """Return why ``legal`` refuses ``value``, or None when it accepts it."""
    if _contains(legal.get("also"), value):
        return None
    kind = legal.get("type")
    choices = legal.get("choices") or []
    if kind == "bool":
        return None if isinstance(value, bool) else "not a bool"
    if kind == "choice":
        return None if _contains(choices, value) else "not one of the choices"
    if kind == "str":
        if not isinstance(value, str):
            return "not a str"
        if choices and not _contains(choices, value):
            return "not one of the choices"
        return None
    if kind == "int":
        if not (isinstance(value, numbers.Integral) and not isinstance(value, bool)):
            return "not an int"
    elif kind == "float":
        if not _is_number(value):
            return "not a number"
        if isinstance(value, float) and math.isnan(value):
            return "not a number"
    else:
        return "legal.type is not known"
    lo, hi = legal.get("min"), legal.get("max")
    if lo is not None and (value < lo or (legal.get("min_open") and value == lo)):
        return f"below min {lo}" if value < lo else f"not above min {lo}"
    if hi is not None and (value > hi or (legal.get("max_open") and value == hi)):
        return f"above max {hi}" if value > hi else f"not below max {hi}"
    return None


def _limit_refusal(limit: dict, value: Any, facts: dict[str, int | None]) -> str | None:
    fact = limit.get("fact")
    known = facts.get(fact)
    if known is None or not _is_number(value):
        return None
    bound = known + limit.get("offset", 0)
    op = limit.get("op")
    ok = value <= bound if op == "le" else value < bound
    if ok:
        return None
    sign = "<=" if op == "le" else "<"
    offset = limit.get("offset", 0)
    rhs = fact if not offset else f"{fact} {'+' if offset > 0 else '-'} {abs(offset)}"
    return f"must be {sign} {rhs} ({fact}={known})"


# ---------------------------------------------------------------------------
# check_declaration
# ---------------------------------------------------------------------------

def _keys(where: str, mapping: dict, allowed: tuple[str, ...], problems: list[str]) -> bool:
    ok = True
    for key in mapping:
        if key not in allowed:
            problems.append(f"{_path(where, key)}: unknown key {key!r}")
            ok = False
    for key in allowed:
        if key not in mapping:
            problems.append(f"{_path(where, key)}: missing key {key!r}")
            ok = False
    return ok


def _path(where: str, key: Any) -> str:
    return f"{where}.{key}" if where else str(key)


def _one_of(where: str, value: Any, allowed: tuple[str, ...], problems: list[str]) -> bool:
    if isinstance(value, str) and value in allowed:
        return True
    problems.append(f"{where}: {value!r} is not one of {', '.join(allowed)}")
    return False


def _text(where: str, value: Any, nullable: bool, problems: list[str]) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not 1 <= len(value) <= TEXT_MAX:
        prefix = "null or " if nullable else ""
        problems.append(f"{where}: must be {prefix}1 to {TEXT_MAX} characters")


def _check_legal(where: str, legal: Any, problems: list[str]) -> bool:
    """Check the shape of ``legal``. Return True when values can be tested against it."""
    if not isinstance(legal, dict):
        problems.append(f"{where}: expected a mapping")
        return False
    if not _keys(where, legal, LEGAL_KEYS, problems):
        return False
    ok = _one_of(f"{where}.type", legal["type"], TYPES, problems)
    for key in ("min", "max"):
        if legal[key] is not None and not _is_number(legal[key]):
            problems.append(f"{where}.{key}: expected a number or null, got {legal[key]!r}")
            ok = False
    for key in ("min_open", "max_open"):
        if not isinstance(legal[key], bool):
            problems.append(f"{where}.{key}: expected a bool, got {legal[key]!r}")
            ok = False
    for key in ("choices", "also"):
        if not isinstance(legal[key], list):
            problems.append(f"{where}.{key}: expected a list, got {legal[key]!r}")
            ok = False
    if not ok:
        return False
    if legal["type"] not in ("int", "float") and (legal["min"] is not None or legal["max"] is not None):
        problems.append(f"{where}: min and max are only for int and float")
    if legal["type"] not in ("choice", "str") and legal["choices"]:
        problems.append(f"{where}.choices: only for choice and str")
    if legal["type"] == "choice" and not legal["choices"] and not legal["also"]:
        problems.append(f"{where}.choices: a choice needs at least one choice")
    if (legal["min"] is not None and legal["max"] is not None and legal["min"] > legal["max"]):
        problems.append(f"{where}: min {legal['min']} is greater than max {legal['max']}")
    return True


def _check_value(where: str, legal: dict, value: Any, problems: list[str]) -> None:
    why = _legal_refusal(legal, value)
    if why is not None:
        problems.append(f"{where}: {value!r} is refused by legal ({why})")


def _check_entry(where: str, entry: Any, problems: list[str]) -> None:
    if not isinstance(entry, dict):
        problems.append(f"{where}: expected a mapping")
        return
    _keys(where, entry, ENTRY_KEYS, problems)
    legal_ok = "legal" in entry and _check_legal(f"{where}.legal", entry["legal"], problems)
    legal = entry.get("legal")
    numeric = legal_ok and legal["type"] in ("int", "float")

    if legal_ok and "default" in entry:
        _check_value(f"{where}.default", legal, entry["default"], problems)

    ladder = entry.get("ladder")
    if ladder is not None:
        if not isinstance(ladder, list) or not ladder:
            problems.append(f"{where}.ladder: expected null or a non-empty list")
        elif legal_ok:
            for i, value in enumerate(ladder):
                _check_value(f"{where}.ladder[{i}]", legal, value, problems)

    sensible = entry.get("sensible")
    if sensible is not None:
        _check_sensible(f"{where}.sensible", sensible, legal if legal_ok else None, numeric, problems)

    limits = entry.get("limits", [])
    if not isinstance(limits, list):
        problems.append(f"{where}.limits: expected a list")
    else:
        for i, limit in enumerate(limits):
            lw = f"{where}.limits[{i}]"
            if not isinstance(limit, dict):
                problems.append(f"{lw}: expected a mapping")
                continue
            if not _keys(lw, limit, LIMIT_KEYS, problems):
                continue
            _one_of(f"{lw}.op", limit["op"], OPS, problems)
            _one_of(f"{lw}.fact", limit["fact"], FACTS, problems)
            offset = limit["offset"]
            if not isinstance(offset, numbers.Integral) or isinstance(offset, bool):
                problems.append(f"{lw}.offset: expected an int, got {offset!r}")
            if legal_ok and not numeric:
                problems.append(f"{lw}: limits are only for int and float")

    role_ok = "role" in entry and _one_of(f"{where}.role", entry["role"], ROLES, problems)
    precomputed = entry.get("precomputed", False)
    if not isinstance(precomputed, bool):
        problems.append(f"{where}.precomputed: expected a bool, got {precomputed!r}")
    elif precomputed and role_ok and entry["role"] != "fixed":
        problems.append(
            f"{where}.precomputed: precomputed is true but role is {entry['role']!r}, not 'fixed'"
        )

    cost = entry.get("cost")
    if cost is not None:
        if not isinstance(cost, dict):
            problems.append(f"{where}.cost: expected null or a mapping")
        elif _keys(f"{where}.cost", cost, COST_KEYS, problems):
            _one_of(f"{where}.cost.time", cost["time"], COST_DIRECTIONS, problems)
            _one_of(f"{where}.cost.memory", cost["memory"], COST_DIRECTIONS, problems)
            _text(f"{where}.cost.note", cost["note"], True, problems)

    if "meaning" in entry:
        _text(f"{where}.meaning", entry["meaning"], False, problems)
    for key in ("more", "less"):
        if key in entry:
            _text(f"{where}.{key}", entry[key], True, problems)


def _check_sensible(where: str, sensible: Any, legal: dict | None, numeric: bool,
                    problems: list[str]) -> None:
    if not isinstance(sensible, dict):
        problems.append(f"{where}: expected null or a mapping")
        return
    if not _keys(where, sensible, SENSIBLE_KEYS, problems):
        return
    if legal is not None and not numeric:
        problems.append(f"{where}: sensible is only for int and float")
        return
    lo, hi, scale = sensible["lo"], sensible["hi"], sensible["scale"]
    bounds_ok = True
    for key, value in (("lo", lo), ("hi", hi)):
        if not _is_number(value):
            problems.append(f"{where}.{key}: expected a number, got {value!r}")
            bounds_ok = False
        elif legal is not None:
            _check_value(f"{where}.{key}", legal, value, problems)
    if bounds_ok and lo > hi:
        problems.append(f"{where}: lo {lo} is greater than hi {hi}")
    if _one_of(f"{where}.scale", scale, SCALES, problems) and scale == "log" and bounds_ok and lo <= 0:
        problems.append(f"{where}: scale log needs lo > 0, got lo {lo}")
    source = sensible["source"]
    if not isinstance(source, dict):
        problems.append(f"{where}.source: expected a mapping")
    elif _keys(f"{where}.source", source, SOURCE_REF_KEYS, problems):
        _one_of(f"{where}.source.kind", source["kind"], SOURCE_KINDS, problems)
        if not isinstance(source["ref"], str) or not source["ref"].strip():
            problems.append(f"{where}.source.ref: expected a non-empty text")


def check_declaration(doc: Any) -> list[str]:
    """Return one message per problem of a declaration, and ``[]`` for a valid one.

    ``doc`` is plain data (for example ``yaml.safe_load`` or ``json.load`` output) from any
    producer. The rules are those of docs/param-declarations.md.
    """
    if not isinstance(doc, dict):
        return [f"declaration: expected a mapping, got {type(doc).__name__}"]
    problems: list[str] = []
    _keys("", doc, DOC_KEYS, problems)
    if "schema_version" in doc and (
        doc["schema_version"] != SCHEMA_VERSION or isinstance(doc["schema_version"], bool)
    ):
        problems.append(f"schema_version: {doc['schema_version']!r} is not {SCHEMA_VERSION}")
    for key in ("method", "origin"):
        if key in doc and (not isinstance(doc[key], str) or not doc[key].strip()):
            problems.append(f"{key}: expected a non-empty text, got {doc[key]!r}")
    if "group" in doc:
        _one_of("group", doc["group"], GROUPS, problems)
    if "params" in doc:
        entries = doc["params"]
        if not isinstance(entries, dict):
            problems.append("params: expected a mapping")
        else:
            for name, entry in entries.items():
                if not isinstance(name, str) or not name:
                    problems.append(f"params: parameter name {name!r} is not a non-empty text")
                    continue
                _check_entry(f"params.{name}", entry, problems)
    return problems


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def check(doc: dict, values: dict, *, n_samples: int | None = None,
          n_features: int | None = None) -> list[str]:
    """Return one message per value in ``values`` that the declaration ``doc`` refuses.

    A value is refused when its parameter is unknown, when its role is not ``knob``, when
    ``legal`` refuses it, or when a limit refuses it and the limit's fact is given. ``sensible``
    and ``ladder`` are never read: any legal value may be submitted.
    """
    entries = doc.get("params", {})
    facts = {"n_samples": n_samples, "n_features": n_features}
    problems: list[str] = []
    for name, value in values.items():
        entry = entries.get(name)
        if entry is None:
            problems.append(f"{name}: not a parameter of {doc.get('method')!r}")
            continue
        role = entry.get("role")
        if role != "knob":
            problems.append(f"{name}: its role is {role}, so it is not offered; only a knob is")
            continue
        why = _legal_refusal(entry["legal"], value)
        if why is not None:
            problems.append(f"{name}={value!r}: refused by legal ({why})")
            continue
        for limit in entry.get("limits", []):
            why = _limit_refusal(limit, value, facts)
            if why is not None:
                problems.append(f"{name}={value!r}: {why}")
    return problems

