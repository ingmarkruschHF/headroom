from __future__ import annotations

import contextlib
import subprocess

import click
import pytest

from headroom.install.lifecycle import (
    reject_task_lifecycle,
    remove_deployment,
    restore_deployment,
    start_deployment,
    stop_deployment,
)
from headroom.install.models import DeploymentManifest


def _manifest(
    *,
    profile: str = "default",
    preset: str = "persistent-service",
    supervisor: str = "service",
) -> DeploymentManifest:
    return DeploymentManifest(
        profile=profile,
        preset=preset,
        runtime_kind="python",
        supervisor_kind=supervisor,
        scope="user",
        provider_mode="manual",
        targets=[],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        service_name=f"headroom-{profile}",
    )


@contextlib.contextmanager
def _acquired_lock(profile):
    yield True


def test_start_deployment_noop_when_already_ready(monkeypatch):
    monkeypatch.setattr("headroom.install.lifecycle.probe_ready", lambda url, timeout=2.0: True)
    monkeypatch.setattr("headroom.install.lifecycle.acquire_runtime_start_lock", _acquired_lock)
    called = []
    monkeypatch.setattr(
        "headroom.install.lifecycle.start_supervisor", lambda manifest: called.append("start")
    )

    start_deployment(_manifest())

    assert called == []


def test_start_deployment_returns_early_when_lock_not_acquired(monkeypatch):
    @contextlib.contextmanager
    def not_acquired(profile):
        yield False

    monkeypatch.setattr("headroom.install.lifecycle.acquire_runtime_start_lock", not_acquired)
    called = []
    monkeypatch.setattr("headroom.install.lifecycle.probe_ready", lambda url, timeout=2.0: False)
    monkeypatch.setattr(
        "headroom.install.lifecycle.start_supervisor", lambda manifest: called.append("start")
    )

    start_deployment(_manifest())

    assert called == []


def test_start_deployment_dispatches_to_supervisor_for_service_kind(monkeypatch):
    monkeypatch.setattr("headroom.install.lifecycle.acquire_runtime_start_lock", _acquired_lock)
    monkeypatch.setattr("headroom.install.lifecycle.probe_ready", lambda url, timeout=2.0: False)
    monkeypatch.setattr("headroom.install.lifecycle.runtime_status", lambda manifest: "stopped")
    monkeypatch.setattr("headroom.install.lifecycle.wait_ready", lambda manifest, timeout_seconds=30: True)
    called = []
    monkeypatch.setattr(
        "headroom.install.lifecycle.start_supervisor", lambda manifest: called.append("supervisor")
    )
    monkeypatch.setattr(
        "headroom.install.lifecycle.start_detached_agent", lambda profile: called.append("detached")
    )
    monkeypatch.setattr(
        "headroom.install.lifecycle.start_persistent_docker", lambda manifest: called.append("docker")
    )

    start_deployment(_manifest(supervisor="service"))

    assert called == ["supervisor"]


def test_start_deployment_dispatches_to_detached_agent_for_task_kind(monkeypatch):
    monkeypatch.setattr("headroom.install.lifecycle.acquire_runtime_start_lock", _acquired_lock)
    monkeypatch.setattr("headroom.install.lifecycle.probe_ready", lambda url, timeout=2.0: False)
    monkeypatch.setattr("headroom.install.lifecycle.runtime_status", lambda manifest: "stopped")
    monkeypatch.setattr("headroom.install.lifecycle.wait_ready", lambda manifest, timeout_seconds=30: True)
    called = []
    monkeypatch.setattr(
        "headroom.install.lifecycle.start_supervisor", lambda manifest: called.append("supervisor")
    )
    monkeypatch.setattr(
        "headroom.install.lifecycle.start_detached_agent", lambda profile: called.append("detached")
    )

    start_deployment(_manifest(supervisor="task"))

    assert called == ["detached"]


def test_start_deployment_dispatches_to_docker_for_docker_preset(monkeypatch):
    monkeypatch.setattr("headroom.install.lifecycle.acquire_runtime_start_lock", _acquired_lock)
    monkeypatch.setattr("headroom.install.lifecycle.probe_ready", lambda url, timeout=2.0: False)
    monkeypatch.setattr("headroom.install.lifecycle.runtime_status", lambda manifest: "stopped")
    monkeypatch.setattr("headroom.install.lifecycle.wait_ready", lambda manifest, timeout_seconds=30: True)
    monkeypatch.setattr("headroom.install.lifecycle.shutil.which", lambda name: "/usr/bin/docker")
    called = []
    monkeypatch.setattr(
        "headroom.install.lifecycle.start_persistent_docker", lambda manifest: called.append("docker")
    )

    start_deployment(_manifest(preset="persistent-docker", supervisor="none"))

    assert called == ["docker"]


def test_start_deployment_raises_when_docker_missing(monkeypatch):
    monkeypatch.setattr("headroom.install.lifecycle.acquire_runtime_start_lock", _acquired_lock)
    monkeypatch.setattr("headroom.install.lifecycle.probe_ready", lambda url, timeout=2.0: False)
    monkeypatch.setattr("headroom.install.lifecycle.shutil.which", lambda name: None)

    with pytest.raises(click.ClickException, match="Docker is required"):
        start_deployment(_manifest(preset="persistent-docker", supervisor="none"))


def test_start_deployment_wraps_file_not_found_error(monkeypatch):
    monkeypatch.setattr("headroom.install.lifecycle.acquire_runtime_start_lock", _acquired_lock)
    monkeypatch.setattr("headroom.install.lifecycle.probe_ready", lambda url, timeout=2.0: False)
    monkeypatch.setattr("headroom.install.lifecycle.runtime_status", lambda manifest: "stopped")

    def raise_missing(manifest):
        raise FileNotFoundError("launchctl not found")

    monkeypatch.setattr("headroom.install.lifecycle.start_supervisor", raise_missing)

    with pytest.raises(click.ClickException, match="Cannot start deployment"):
        start_deployment(_manifest(supervisor="service"))


def test_start_deployment_wraps_called_process_error(monkeypatch):
    monkeypatch.setattr("headroom.install.lifecycle.acquire_runtime_start_lock", _acquired_lock)
    monkeypatch.setattr("headroom.install.lifecycle.probe_ready", lambda url, timeout=2.0: False)
    monkeypatch.setattr("headroom.install.lifecycle.runtime_status", lambda manifest: "stopped")

    def raise_failed(manifest):
        raise subprocess.CalledProcessError(1, ["launchctl", "bootstrap"])

    monkeypatch.setattr("headroom.install.lifecycle.start_supervisor", raise_failed)

    with pytest.raises(click.ClickException, match="Cannot start deployment"):
        start_deployment(_manifest(supervisor="service"))


def test_start_deployment_raises_when_never_becomes_ready(monkeypatch):
    monkeypatch.setattr("headroom.install.lifecycle.acquire_runtime_start_lock", _acquired_lock)
    monkeypatch.setattr("headroom.install.lifecycle.probe_ready", lambda url, timeout=2.0: False)
    monkeypatch.setattr("headroom.install.lifecycle.runtime_status", lambda manifest: "stopped")
    monkeypatch.setattr("headroom.install.lifecycle.start_supervisor", lambda manifest: None)
    monkeypatch.setattr("headroom.install.lifecycle.wait_ready", lambda manifest, timeout_seconds=30: False)

    with pytest.raises(click.ClickException, match="did not become ready"):
        start_deployment(_manifest(supervisor="service"))


def test_start_deployment_restarts_when_running_but_not_actually_ready(monkeypatch):
    monkeypatch.setattr("headroom.install.lifecycle.acquire_runtime_start_lock", _acquired_lock)
    monkeypatch.setattr("headroom.install.lifecycle.probe_ready", lambda url, timeout=2.0: False)
    monkeypatch.setattr("headroom.install.lifecycle.runtime_status", lambda manifest: "running")

    def fake_wait_ready(manifest, timeout_seconds=30):
        # First call (short startup check) fails; second call (post-start) succeeds.
        return timeout_seconds == 45

    monkeypatch.setattr("headroom.install.lifecycle.wait_ready", fake_wait_ready)
    stopped = []
    started = []
    monkeypatch.setattr(
        "headroom.install.lifecycle.stop_runtime", lambda manifest: stopped.append(manifest.profile)
    )
    monkeypatch.setattr(
        "headroom.install.lifecycle.start_supervisor", lambda manifest: started.append(manifest.profile)
    )

    start_deployment(_manifest(supervisor="service"))

    assert stopped == ["default"]
    assert started == ["default"]


def test_stop_deployment_stops_supervisor_and_runtime_for_service_kind(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "headroom.install.lifecycle.stop_supervisor", lambda manifest: calls.append("supervisor")
    )
    monkeypatch.setattr(
        "headroom.install.lifecycle.stop_runtime", lambda manifest: calls.append("runtime")
    )

    stop_deployment(_manifest(supervisor="service"))

    assert calls == ["supervisor", "runtime"]


def test_stop_deployment_skips_supervisor_for_non_service_kind(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "headroom.install.lifecycle.stop_supervisor", lambda manifest: calls.append("supervisor")
    )
    monkeypatch.setattr(
        "headroom.install.lifecycle.stop_runtime", lambda manifest: calls.append("runtime")
    )

    stop_deployment(_manifest(supervisor="task"))

    assert calls == ["runtime"]


def test_remove_deployment_always_deletes_manifest_even_if_steps_fail(monkeypatch):
    def raise_error(manifest):
        raise RuntimeError("boom")

    monkeypatch.setattr("headroom.install.lifecycle.stop_deployment", raise_error)
    monkeypatch.setattr("headroom.install.lifecycle.remove_supervisor", raise_error)
    monkeypatch.setattr("headroom.install.lifecycle.revert_mutations", raise_error)
    deleted = []
    monkeypatch.setattr(
        "headroom.install.lifecycle.delete_manifest", lambda profile: deleted.append(profile)
    )

    remove_deployment(_manifest(profile="default"))

    assert deleted == ["default"]


def test_remove_deployment_calls_every_step_when_none_fail(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "headroom.install.lifecycle.stop_deployment", lambda manifest: calls.append("stop")
    )
    monkeypatch.setattr(
        "headroom.install.lifecycle.remove_supervisor", lambda manifest: calls.append("remove_supervisor")
    )
    monkeypatch.setattr(
        "headroom.install.lifecycle.revert_mutations", lambda manifest: calls.append("revert")
    )
    monkeypatch.setattr(
        "headroom.install.lifecycle.delete_manifest", lambda profile: calls.append("delete")
    )

    remove_deployment(_manifest(profile="default"))

    assert calls == ["stop", "remove_supervisor", "revert", "delete"]


def test_restore_deployment_reapplies_mutations_and_starts(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "headroom.install.lifecycle.apply_mutations", lambda manifest: calls.append("apply") or []
    )
    monkeypatch.setattr(
        "headroom.install.lifecycle.install_supervisor", lambda manifest: calls.append("install") or []
    )
    monkeypatch.setattr(
        "headroom.install.lifecycle.save_manifest", lambda manifest: calls.append("save")
    )
    monkeypatch.setattr(
        "headroom.install.lifecycle.start_deployment", lambda manifest: calls.append("start")
    )

    original = _manifest(profile="default")
    restore_deployment(original)

    assert calls == ["apply", "install", "save", "start"]


def test_reject_task_lifecycle_raises_for_task_supervisor():
    with pytest.raises(click.ClickException, match="persistent-task scheduling"):
        reject_task_lifecycle(_manifest(supervisor="task"), "start")


@pytest.mark.parametrize("supervisor", ["service", "none"])
def test_reject_task_lifecycle_allows_non_task_supervisors(supervisor):
    reject_task_lifecycle(_manifest(supervisor=supervisor), "start")
