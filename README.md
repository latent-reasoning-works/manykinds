# manykinds

**The typed data-kind vocabulary for the latent-reasoning stack.**

A *kind* is a self-describing data object that guards its own structure — it
validates on construction and refuses an op that demands structure it lacks,
instead of being a bare array whose axes you have to trust. `manykinds` is the
small, shared package that defines that vocabulary so independent tools can
interoperate through it without depending on each other.

It's the same pattern as `anndata` in single-cell or the Array API standard for
array libraries: one light package everyone agrees on, rather than each tool
inventing (or embedding) its own data contract.

## Install

The checks (`Kind`, `KindSpec`, `manykinds.params`) need only the standard library:

```sh
uv add manykinds
```

The array kinds (`LabeledArray`, `SparseGraph`, `FileArtifact`, `Table`, `Sequence`)
need the array packages:

```sh
uv add "manykinds[arrays]"
```

A dependent that already has a kind's packages can use that kind without the extra.
`SparseGraph` and `Sequence` need only numpy. If a kind's packages are missing, the
first access to it raises `ImportError` that names the kind and the extra.

## What's in it

- **`Kind`** — a `@runtime_checkable` structural protocol (`provenance`,
  `require`, `tagged`). Orchestrators type their ops against it; a concrete kind
  is substitutable iff it offers the three members — no inheritance required.
  Lives in `manykinds.base` and imports without the array stack.
- **`LabeledArray`** — an `xarray.DataArray` with named dims + dim-aligned coords.
  The canonical kind for cell×gene matrices and embeddings. Dense and
  `sparse.COO`-backed arrays both serialize to zarr (validating on read).
- **`SparseGraph`** — a graph as numpy arrays (an E×2 integer edge list + node
  ids, plus optional `edge_weights` for weighted/signed graphs like GRNs). `.npz`.
- **`FileArtifact`** — an opaque reference to a file a tool produced (path +
  format + checksum). The escape hatch for outputs the vocabulary doesn't model
  structurally: carry the file through a pipeline unchanged.
- **`Table`** — heterogeneous columnar data (a pandas DataFrame) for mixed-dtype
  tables like AnnData `obs`/`var`. Parquet persistence (`manykinds[table]` extra, which includes `arrays`).
- **`Sequence`** — biological sequences over an alphabet (DNA/protein), with
  optional ids. `.npz`.
- **`KindSpec`** — a data-free structural signature (kind name + dims + coords).
  `spec_a.satisfies(spec_b)` is the plan-time mirror of `Kind.require`, so an
  orchestrator can type an op's inputs/outputs and check a chain is valid *before*
  running it. A concrete kind reports its own signature via `.spec()`.

## Usage

```python
import xarray as xr, numpy as np
from manykinds import Kind, LabeledArray

la = LabeledArray(
    xr.DataArray(
        np.random.randn(100, 2000),
        dims=("cell", "gene"),
        coords={"time": ("cell", np.zeros(100, dtype=int))},
    )
)

isinstance(la, Kind)                       # True — structural, no inheritance
la.require("cell", "gene", coords=("time",))   # precondition gate; raises if absent
la = la.tagged("pca")                      # append to the immutable provenance trail
la.serialize("embedding.zarr")             # dense or sparse; validates on load
```

## Parameter declarations

`manykinds.params` defines how a method or a tool declares its parameters: a plain-data JSON or
YAML document (version 1) that says which values each parameter accepts, which ones a caller
may set, and how each one is limited by the data. `check_declaration(doc)` checks a document
from any producer, and `check(declaration, values, n_samples=..., n_features=...)` checks
submitted values. The module imports only the standard library, so a tool author checks a
manifest without the array stack:

```sh
python -m manykinds.params check my_tool.json
```

The format is defined in [docs/param-declarations.md](docs/param-declarations.md).

## Persistence format

`LabeledArray` Zarr stores and `SparseGraph` / `Sequence` NPZ archives use
manykinds format version 1. A reserved `_manykinds` envelope identifies the
format, version, kind, and encoding (`dense`, `coo`, or `npz`). Readers validate
that identity before interpreting components; caller array names do not select
an encoding.

Zarr keeps the op trail separate from the array's domain attrs and each
coordinate's attrs, dims, and name. Caller attrs such as `provenance`, `kind`,
and `version` round-trip independently. The top-level `_manykinds` attr is
reserved on arrays and coordinates: serialization rejects a collision before
writing. Other attrs are nested inside the envelope, so xarray's own storage
attrs cannot consume them. Numpy scalars and arrays are recursively converted
to JSON values; bytes (including `np.bytes_`) decode as UTF-8, and tuples become
JSON lists. Internal component names are independent of caller names.

NPZ stores its envelope as a Unicode JSON scalar and its op trail as a 1-D
Unicode array, including an empty trail. Both loaders use `allow_pickle=False`,
with no pickle fallback. Graph node IDs and optional sequence IDs must be 1-D
arrays with boolean, integer, floating, complex, fixed-width byte-string, or
Unicode dtypes (`b`, `i`, `u`, `f`, `c`, `S`, `U`); dtype is preserved. Object,
structured, datetime, and other ID encodings are rejected on write and read.
Convert object string IDs explicitly to a fixed-width string dtype before
serializing. No NPZ component may require object/pickle storage.

**Compatibility decision:** unversioned Zarr and NPZ archives are rejected,
even if their contents would otherwise be safe. Unknown versions, kinds, and
encodings are also rejected. Existing archives must be regenerated from trusted
source data with this writer; there is no automatic migration or pickle
fallback. Domain attrs already overwritten by an old writer cannot be recovered.
Old package readers do not support this versioned format; upgrade readers and
writers together. Table and FileArtifact persistence are unchanged.

## Who depends on it

- **Producers** (dataset adapters like `manylatents-omics`, model wrappers, tool
  shims) *construct* kinds from their own formats.
- **Orchestrators** (e.g. an op registry / planner) type their ops against `Kind`
  and chain a producer's output into a consumer's `require`.

The vocabulary is deliberately small and stable — it changes far more slowly than
the tools that speak it.

## License

Apache-2.0; see [LICENSE](LICENSE). [NOTICE](NOTICE) keeps the MIT notice of releases up to 0.1.1.

Unless you state otherwise, any contribution you intentionally submit for inclusion in manykinds is licensed under the Apache License, Version 2.0, as section 5 of the license describes, without any additional terms or conditions. That includes the license's patent grant (section 3).
