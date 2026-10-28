"""Bulk read view handed to Python processes.

The hot `count` column crosses the boundary zero-copy (M2). Real v2ecoli
processes also resolve molecule names on their first update via
`bulk_name_to_idx(names, states["bulk"]["id"])`, so the view must answer
`["id"]`. `BulkView` is the count-only zero-copy structured view plus a cached
(immutable) `id` array: `view["count"]` is the zero-copy column and
`view["id"]` the cached ids. Its dtype stays `[("count", "<i8")]`.
"""

import numpy as np


class BulkView(np.ndarray):
    _ids = None

    def __getitem__(self, key):
        if isinstance(key, str) and key == "id" and self._ids is not None:
            return self._ids
        return super().__getitem__(key)

    def __array_finalize__(self, obj):
        self._ids = getattr(obj, "_ids", None)


def make_ids(ids):
    return np.array(ids, dtype=str)


def wrap(count_view, ids_array):
    v = count_view.view(BulkView)
    v._ids = ids_array
    return v
