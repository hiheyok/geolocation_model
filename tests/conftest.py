"""Name the release before anything imports config.

config refuses to guess a release, because guessing is what silently paired
each image with another release's cached row five separate times. The tests
are no exception -- they just have no reason to prefer one, so they declare
the smaller one here. pytest imports conftest before any test module, which
is the only point where this can be set: config reads it at import.
"""

import os

os.environ.setdefault("OSV_RELEASE", "s01")
