# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-10-31

### Added
- `manykinds.params`: the version-1 parameter-declaration contract, as a standard-library module.
  A declaration lists a method's parameters with their legal values, default, sensible range,
  ladder, data-dependent limits, role (`knob`, `fixed`, `infra`), compute cost and meaning.
  `check_declaration(doc)` returns problems whose messages start with the field path, and
  `check(declaration, values, n_samples=, n_features=)` checks chosen values against the legal
  values and limits. `python -m manykinds.params check FILE` prints one JSON object and exits
  0, 1 or 2. The format is defined in `docs/param-declarations.md`.

### Changed
- The core install is light: `uv add manykinds` installs no array libraries. The array kinds
  (`LabeledArray`, `SparseGraph`, `FileArtifact`, `Table`, `Sequence`) need the new extra,
  `uv add "manykinds[arrays]"`, which installs what 0.1.1 installed. A kind loads on first use and
  works whenever its own packages are installed, extra or not; otherwise the error names
  `manykinds[arrays]`.
- Relicensed from MIT to Apache-2.0.
- Persistence: the stored format is versioned, and loading no longer takes ownership of the
  caller's data.
- The `Table` parquet error now gives `uv add "manykinds[table]"`.

## [0.1.1] - 2026-07-15

### Added
- `FileArtifact`, `Table` and `Sequence` kinds, and edge weights on `SparseGraph`.
- `KindSpec`: data-free structural signatures for checking a chain of steps before it runs.
- A PyPI publish workflow: a pushed `v*` tag builds and publishes through trusted publishing.

### Fixed
- A `numba>=0.60` floor, so that `sparse` resolves to installable wheels on Python 3.11 and 3.12.

## [0.1.0] - 2026-07-07

### Added
- The `Kind` protocol and the first kinds, `LabeledArray` and `SparseGraph`.
