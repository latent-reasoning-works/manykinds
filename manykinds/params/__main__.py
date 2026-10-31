"""Command line for parameter declarations. It prints JSON and needs no screen.

``python -m manykinds.params check FILE``
    ``{"ok": bool, "problems": [{"path", "message"}]}`` on stdout, for a declaration from any
    producer. Exit 0 when it is valid, 1 when it has problems, 2 when the file is unreadable.

A file named ``*.yaml`` or ``*.yml`` is read as YAML when PyYAML is importable; without PyYAML
the command exits 2 and says so on stderr. Every other file is read as JSON with the standard
library.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from manykinds.params import check_declaration

_YAML_SUFFIXES = (".yaml", ".yml")
_NO_PYYAML = (
    "reading YAML needs PyYAML, which is not importable here. Add it to the project with "
    "`uv add pyyaml`, or run once with "
    "`uv run --with pyyaml python -m manykinds.params check FILE`; "
    "or write the declaration as JSON"
)


def _problem(message: str) -> dict:
    path, sep, rest = message.partition(": ")
    if not sep:
        return {"path": "", "message": message}
    return {"path": path, "message": rest}


def _unreadable(message: str) -> int:
    print(json.dumps({"ok": False, "problems": [{"path": "", "message": message}]}))
    return 2


def _check(path: Path) -> int:
    if path.suffix.lower() in _YAML_SUFFIXES:
        try:
            import yaml
        except ImportError:
            print(f"check: {_NO_PYYAML}", file=sys.stderr)
            return _unreadable(f"cannot read {path}: {_NO_PYYAML}")
        errors: tuple[type[BaseException], ...] = (
            OSError, UnicodeDecodeError, ValueError, yaml.YAMLError,
        )
        load = yaml.safe_load
    else:
        errors = (OSError, UnicodeDecodeError, ValueError)
        load = json.loads
    try:
        doc = load(path.read_text(encoding="utf-8"))
    except errors as exc:
        return _unreadable(f"cannot read {path}: {type(exc).__name__}: {exc}")
    problems = [_problem(m) for m in check_declaration(doc)]
    print(json.dumps({"ok": not problems, "problems": problems}))
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m manykinds.params")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="check a JSON or YAML declaration file")
    check.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    return _check(args.file)


if __name__ == "__main__":
    sys.exit(main())
