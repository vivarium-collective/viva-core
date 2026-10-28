"""Oracle generator for viva-core Task 3 parity tests.

Runs the AUTHORITATIVE Python apply paths the E. coli model uses:
  - bulk: v2ecoli.types.bulk_numpy apply  (state["count"][idx] += value)
  - unique: v2ecoli.library.schema.UniqueNumpyUpdater.updater (set/add/delete)

Prints the post-apply decoded state so the Rust test can hard-code the
expected values. Run inside /Users/eranagmon/code/v2ecoli/.venv.
"""
import numpy as np
from v2ecoli.library.schema import UniqueNumpyUpdater, MetadataArray

np.set_printoptions(legacy="1.21")

# ---------------------------------------------------------------------------
# 1. BULK apply : state["count"][idx] += value, for each (idx, value)
# ---------------------------------------------------------------------------
print("=== BULK ===")
bulk_dtype = np.dtype([("id", "U8"), ("count", np.int64)])
bulk = np.zeros(5, dtype=bulk_dtype)
bulk["id"] = ["A", "B", "C", "D", "E"]
bulk["count"] = [10, 20, 30, 40, 50]

# reconcile(BulkNumpyUpdate) = concatenate delta lists from a layer, then apply.
layer_updates = [
    [(0, 5), (2, -3)],
    [(0, 100), (4, 7)],
    [(2, 3), (1, 1)],
]
combined = []
for u in layer_updates:
    combined.extend(u)
# apply (bulk_numpy): count[idx] += value
for idx, value in combined:
    bulk["count"][idx] += value
print("bulk.count =", list(int(x) for x in bulk["count"]))

# ---------------------------------------------------------------------------
# 2. UNIQUE set/add/delete via the real UniqueNumpyUpdater
# ---------------------------------------------------------------------------
print("=== UNIQUE ===")
# Minimal unique type: one data column (TU_index), plus _entryState + unique_index.
uniq_dtype = np.dtype(
    [("TU_index", np.int64), ("_entryState", np.int8), ("unique_index", np.int64)]
)
# 6 rows: rows 0,1,3,5 active (entryState=1); rows 2,4 inactive (free).
arr = np.zeros(6, dtype=uniq_dtype)
arr["TU_index"] = [100, 101, 0, 103, 0, 105]
arr["_entryState"] = [1, 1, 0, 1, 0, 1]
arr["unique_index"] = [0, 1, 0, 3, 0, 5]
# next unique index counter (metadata); start at 6
current = MetadataArray(arr, 6)

updater = UniqueNumpyUpdater()
# One accumulated update then flush: set TU_index of all active rows,
# add 3 new molecules, delete active molecules 0 and 2 (0-based among the
# INITIALLY-ACTIVE rows -> rows 0 and 3), then flush with update=True.
update = {
    "set": {"TU_index": np.array([200, 201, 203, 205], dtype=np.int64)},
    "add": {"TU_index": np.array([900, 901, 902], dtype=np.int64)},
    "delete": [0, 2],
    "update": True,
}
result = updater.updater(current, update)

active = result["_entryState"].view(np.bool_)
print("result.size =", int(result.size))
print("TU_index    =", list(int(x) for x in result["TU_index"]))
print("_entryState =", list(int(x) for x in result["_entryState"]))
print("unique_index=", list(int(x) for x in result["unique_index"]))
print("metadata    =", int(result.metadata))
print("active TU_index   =", list(int(x) for x in result["TU_index"][active]))
print("active unique_idx =", list(int(x) for x in result["unique_index"][active]))
