"""Shared start/stop/remove/restore logic for persistent deployments.

Extracted from ``headroom.cli.install`` so both the ``install`` command
group and ``headroom doctor --fix`` can start, stop, remove, or restore a
deployment through one implementation instead of duplicating the
launchd/docker/detached-agent branching in a second place.
"""

from __future__ import annotations

import shutil
import subprocess
from copy import deepcopy

import click

from .health import probe_ready
from .models import DeploymentManifest, InstallPreset, SupervisorKind
from .providers import apply_mutations, revert_mutations
from .runtime import (
    acquire_runtime_start_lock,
    runtime_status,
    start_detached_agent,
    start_persistent_docker,
    stop_runtime,
    wait_ready,
)
from .state import delete_manifest, save_manifest
from .supervisors import install_supervisor, remove_supervisor, start_supervisor, stop_supervisor

_STARTUP_READY_TIMEOUT_SECONDS = 15


def start_deployment(manifest: DeploymentManifest, *, assume_start_lock: bool = False) -> None:
    if not assume_start_lock:
        with acquire_runtime_start_lock(manifest.profile) as acquired:
            if not acquired:
                click.echo(f"Deployment '{manifest.profile}' start is already in progress.")
                return
            start_deployment(manifest, assume_start_lock=True)
            return

    if probe_ready(manifest.health_url):
        return
    if manifest.preset == InstallPreset.PERSISTENT_DOCKER.value and shutil.which("docker") is None:
        raise click.ClickException(
            "Docker is required for this deployment but 'docker' was not found on PATH."
        )
    if runtime_status(manifest) == "running":
        if wait_ready(manifest, timeout_seconds=_STARTUP_READY_TIMEOUT_SECONDS):
            return
        stop_runtime(manifest)

    try:
        if manifest.preset == InstallPreset.PERSISTENT_DOCKER.value:
            start_persistent_docker(manifest)
        elif manifest.supervisor_kind == SupervisorKind.SERVICE.value:
            start_supervisor(manifest)
        else:
            start_detached_agent(manifest.profile)
    except FileNotFoundError as e:
        # A required external binary (docker, launchctl, systemctl) is missing.
        raise click.ClickException(f"Cannot start deployment '{manifest.profile}': {e}") from None
    except subprocess.CalledProcessError as e:
        raise click.ClickException(
            f"Cannot start deployment '{manifest.profile}': command failed "
            f"({' '.join(map(str, e.cmd)) if isinstance(e.cmd, list | tuple) else e.cmd})"
        ) from None

    if not wait_ready(manifest, timeout_seconds=45):
        raise click.ClickException(
            f"Deployment '{manifest.profile}' did not become ready after start."
        )


def stop_deployment(manifest: DeploymentManifest) -> None:
    if manifest.supervisor_kind == SupervisorKind.SERVICE.value:
        stop_supervisor(manifest)
    stop_runtime(manifest)


def remove_deployment(manifest: DeploymentManifest) -> None:
    try:
        stop_deployment(manifest)
    except Exception:
        pass
    try:
        remove_supervisor(manifest)
    except Exception:
        pass
    try:
        revert_mutations(manifest)
    except Exception:
        pass
    delete_manifest(manifest.profile)


def restore_deployment(manifest: DeploymentManifest) -> None:
    restored = deepcopy(manifest)
    restored.mutations = apply_mutations(restored)
    restored.artifacts = install_supervisor(restored)
    save_manifest(restored)
    start_deployment(restored)


def reject_task_lifecycle(manifest: DeploymentManifest, action: str) -> None:
    if manifest.supervisor_kind == SupervisorKind.TASK.value:
        raise click.ClickException(
            f"Deployment '{manifest.profile}' uses persistent-task scheduling; "
            f"`headroom install {action}` is not supported for task deployments."
        )
