"""Bulk read view handed to Python processes.

The hot `count` column crosses the boundary zero-copy (M2). Real v2ecoli
processes also resolve molecule names on their first update via
`bulk_name_to_idx(names, states["bulk"]["id"])`, so the view must answer
`["id"]`. `BulkView` is the count-only zero-copy structured view plus a cached
(immutable) `id` array: `view["count"]` is the zero-copy column and
`view["id"]` the cached ids. Its dtype stays `[("count", "<i8")]`.

**Subclass degradation (M5).** pb hands processes a *plain* ``np.ndarray`` for
bulk, so any value a process DERIVES from bulk (``aa_conc = counts_to_molar(
bulk["count"][idx])``, a ``.min()``/``.sum()`` reduction, an arithmetic result)
is a plain ndarray / numpy scalar there. A naive ndarray subclass instead
*propagates* itself through numpy ufuncs, reductions and field slicing, so a
bulk-derived value stays a ``BulkView`` and e.g. ``(x / f).min()`` returns a 0-d
``BulkView`` — which numba kernels reject (``array(float64, 0d)`` where a scalar
is expected), diverging from pb. So `BulkView` degrades to the base class on
EVERY derived result: a field read (``view["count"]``) returns a plain ndarray
VIEW (still zero-copy — ``np.asarray`` of a subclass shares the buffer), and
ufuncs / ``__array_wrap__`` strip the subclass. Only the top-level ``view``
object itself is a ``BulkView`` (so it can answer ``["id"]``); everything taken
out of it is a plain ndarray, exactly as under pb.
"""

import numpy as np


class BulkView(np.ndarray):
    _ids = None
    _floats = None  # name -> zero-copy float64 submass column

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "id" and self._ids is not None:
                return self._ids
            if self._floats is not None and key in self._floats:
                # A submass column (`rRNA_submass`, …): the real bulk carries
                # these; hand back the plain zero-copy float view.
                return self._floats[key]
            # Field column (e.g. "count"): hand back a PLAIN ndarray view so all
            # downstream math/reductions match pb. `np.asarray` shares the buffer
            # (no copy), so the zero-copy read path is preserved.
            return np.asarray(super().__getitem__(key))
        if isinstance(key, list) and key and all(isinstance(k, str) for k in key):
            # Multi-field structured access `bulk[[f0, f1, ...]]` (MassDeriver's
            # `bulk[ordered_submasses]`): assemble a plain structured array over
            # the requested columns, exactly as indexing pb's structured bulk does
            # (so `rfn.structured_to_unstructured` on it behaves identically).
            cols = {f: self[f] for f in key}
            n = len(next(iter(cols.values()))) if cols else len(self)
            dtype = [(f, cols[f].dtype) for f in key]
            out = np.empty(n, dtype=dtype)
            for f in key:
                out[f] = cols[f]
            return out
        return super().__getitem__(key)

    def __array_finalize__(self, obj):
        self._ids = getattr(obj, "_ids", None)
        self._floats = getattr(obj, "_floats", None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Degrade every ufunc result to the base class: cast BulkView operands to
        # plain ndarrays and dispatch the ufunc on them, so arithmetic on bulk is
        # never a BulkView (pb parity; keeps numba happy for derived scalars).
        base = tuple(
            np.asarray(x) if isinstance(x, BulkView) else x for x in inputs
        )
        out = kwargs.get("out")
        if out is not None:
            kwargs["out"] = tuple(
                np.asarray(x) if isinstance(x, BulkView) else x for x in out
            )
        return getattr(ufunc, method)(*base, **kwargs)

    def __array_wrap__(self, out_arr, context=None, return_scalar=False):
        # Reductions (`.min()`, `.sum()`, …) and other non-ufunc wraps return a
        # plain ndarray / numpy scalar, exactly as a base ndarray would.
        arr = np.asarray(out_arr)
        if arr.ndim == 0:
            return arr[()]  # 0-d -> numpy scalar, like ndarray.min() on a base array
        return arr


def make_ids(ids):
    return np.array(ids, dtype=str)


def wrap(count_view, ids_array, floats=None):
    v = count_view.view(BulkView)
    v._ids = ids_array
    v._floats = floats
    return v
