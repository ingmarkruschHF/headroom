"""Isolation fixtures for headroom.install tests.

`deploy_root()` (and therefore every manifest/runner-script path these tests
exercise) resolves from `$HEADROOM_WORKSPACE_DIR` when it's set in the
environment, not just `Path.home()`. Tests here only patch `Path.home`, so
running the suite in a shell that has `HEADROOM_WORKSPACE_DIR` exported (as
this personal deployment's `.zshrc` always does) makes `save_manifest` /
`delete_manifest` / runner-script writes land in the real, live workspace
instead of `tmp_path` — this actually happened once and corrupted the live
launchd deployment's manifest.json. Clear the override for every test in
this directory so `Path.home` patching is sufficient again.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_headroom_workspace_env(monkeypatch):
    monkeypatch.delenv("HEADROOM_WORKSPACE_DIR", raising=False)
    monkeypatch.delenv("HEADROOM_CONFIG_DIR", raising=False)
