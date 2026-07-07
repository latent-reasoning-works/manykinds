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

## What's in it

- **`Kind`** — a `@runtime_checkable` structural protocol (`provenance`,
  `require`, `tagged`). Orchestrators type their ops against it; a concrete kind
  is substitutable iff it offers the three members — no inheritance required.
  Lives in `manykinds.base` and imports without the array stack.
- **`LabeledArray`** — an `xarray.DataArray` with named dims + dim-aligned coords.
  The canonical kind for cell×gene matrices and embeddings. Dense and
  `sparse.COO`-backed arrays both serialize to zarr (validating on read).
- **`SparseGraph`** — a graph as two numpy arrays (an E×2 integer edge list +
  node ids). `.npz` persistence.

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

## Who depends on it

- **Producers** (dataset adapters like `manylatents-omics`, model wrappers, tool
  shims) *construct* kinds from their own formats.
- **Orchestrators** (e.g. an op registry / planner) type their ops against `Kind`
  and chain a producer's output into a consumer's `require`.

The vocabulary is deliberately small and stable — it changes far more slowly than
the tools that speak it.

## License

MIT
