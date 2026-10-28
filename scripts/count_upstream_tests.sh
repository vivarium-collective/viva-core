#!/usr/bin/env bash
# Re-derive the upstream-suite test counts recorded in
# python/viva_core/upstream_suite.py (_DROPIN). Versions are pinned (spec S8-B8)
# to bigraph-schema 1.6.0 / process-bigraph 1.8.3, so these counts are stable.
#
# Requires the source checkouts at ~/code/bigraph-schema and ~/code/process-bigraph
# with the v1.6.0 / v1.8.3 tags fetched (the INSTALLED wheels don't ship tests).
set -euo pipefail

bs=${BS_REPO:-$HOME/code/bigraph-schema}
pb=${PB_REPO:-$HOME/code/process-bigraph}

echo "bigraph-schema v1.6.0:"
echo "  test functions : $(git -C "$bs" show v1.6.0:tests.py | grep -cE '^def test')"

echo "process-bigraph v1.8.3:"
echo "  test functions : $(git -C "$pb" show v1.8.3:tests.py | grep -cE '^def test')"
echo "  Composite(...) : $(git -C "$pb" show v1.8.3:tests.py | grep -cE 'Composite\(')"
echo "  .run(...)      : $(git -C "$pb" show v1.8.3:tests.py | grep -cE '\.run\(')"
