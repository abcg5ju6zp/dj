"""
Unit tests for deployment snapshotting (digest, drift, server-pull guards).
These are pure/mock tests and require no database container.
"""

import json
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datajunction_server.errors import DJInvalidInputException
from datajunction_server.internal.deployment import snapshot as snapshot_mod
from datajunction_server.internal.git.github_service import GitHubServiceError
from datajunction_server.models.deployment import (
    DeploymentSpec,
    GitDeploymentSource,
    TransformSpec,
)


def _spec(query: str = "SELECT 1") -> DeploymentSpec:
    return DeploymentSpec(
        namespace="ns",
        nodes=[TransformSpec(name="x", query=query)],
    )


def _git_source(**kwargs) -> GitDeploymentSource:
    defaults = {"repository": "o/r"}
    defaults.update(kwargs)
    return GitDeploymentSource(**defaults)


def test_spec_digest_is_stable_sensitive_and_hex():
    s1 = _spec()
    s2 = DeploymentSpec(**json.loads(json.dumps(s1.model_dump(mode="json"))))
    s3 = _spec("SELECT 2")
    d1, d2, d3 = (
        snapshot_mod.spec_digest(s1),
        snapshot_mod.spec_digest(s2),
        snapshot_mod.spec_digest(s3),
    )
    assert d1 == d2
    assert d1 != d3
    assert re.fullmatch(r"[0-9a-f]{64}", d1)


def test_canonical_json_ignores_dict_key_order():
    raw = _spec().model_dump(mode="json")
    reversed_raw = dict(reversed(list(raw.items())))
    assert snapshot_mod.canonical_spec_json(
        _spec(),
    ) == snapshot_mod.canonical_spec_json(DeploymentSpec(**reversed_raw))


def test_server_pull_detection():
    pull = DeploymentSpec(namespace="ns", source=_git_source(branch="feature/x"))
    pinned = DeploymentSpec(
        namespace="ns",
        source=_git_source(branch="feature/x", commit_sha="abc"),
    )
    commit_only = DeploymentSpec(namespace="ns", source=_git_source(commit_sha="abc"))
    assert snapshot_mod.is_server_pull(pull)
    assert not snapshot_mod.is_server_pull(pinned)
    assert not snapshot_mod.is_server_pull(commit_only)
    assert not snapshot_mod.is_server_pull(_spec())


def test_build_snapshot_requires_pinned_commit():
    with pytest.raises(ValueError):
        snapshot_mod.build_deployment_snapshot(
            DeploymentSpec(namespace="ns", source=_git_source(branch="main")),
        )


def test_build_snapshot_normalizes_repo_and_freezes_digest():
    spec = DeploymentSpec(
        namespace="ns",
        nodes=[TransformSpec(name="x", query="SELECT 1")],
        source=GitDeploymentSource(
            repository="https://github.com/o/r.git",
            branch="main",
            commit_sha="deadbeef",
            commit_author_name="A",
        ),
    )
    snap = snapshot_mod.build_deployment_snapshot(spec)
    assert snap.repository == "o/r"
    assert snap.branch == "main"
    assert snap.commit_sha == "deadbeef"
    assert snap.commit_author_name == "A"
    assert snap.spec_digest == snapshot_mod.spec_digest(spec)
    assert snap.pinned_at is not None


async def test_check_remote_drift_equal_head_returns_none():
    source = _git_source(branch="main", commit_sha="c1")
    mock_github = MagicMock()
    mock_github.resolve_ref_to_sha = AsyncMock(return_value="c1")
    with patch.object(snapshot_mod, "GitHubService", return_value=mock_github):
        assert await snapshot_mod.check_remote_drift(source) is None


async def test_check_remote_drift_moved_head_returns_actual():
    source = _git_source(branch="main", commit_sha="c1")
    mock_github = MagicMock()
    mock_github.resolve_ref_to_sha = AsyncMock(return_value="c2")
    with patch.object(snapshot_mod, "GitHubService", return_value=mock_github):
        assert await snapshot_mod.check_remote_drift(source) == "c2"


async def test_check_remote_drift_unreachable_propagates():
    source = _git_source(branch="main", commit_sha="c1")
    mock_github = MagicMock()
    mock_github.resolve_ref_to_sha = AsyncMock(
        side_effect=GitHubServiceError(message="boom"),
    )
    with patch.object(snapshot_mod, "GitHubService", return_value=mock_github):
        with pytest.raises(GitHubServiceError):
            await snapshot_mod.check_remote_drift(source)


async def test_check_remote_drift_commit_only_source_not_checked():
    assert await snapshot_mod.check_remote_drift(
        _git_source(commit_sha="c1"),
    ) is None


async def test_resolve_server_pull_passthrough_when_commit_pinned():
    spec = DeploymentSpec(
        namespace="ns",
        source=_git_source(branch="main", commit_sha="c1"),
    )
    assert await snapshot_mod.resolve_server_pull(None, spec) is spec


async def test_resolve_server_pull_rejects_repo_mismatch():
    spec = DeploymentSpec(
        namespace="ns",
        source=_git_source(repository="o/other", branch="main"),
    )
    with patch(
        "datajunction_server.internal.deployment.snapshot.resolve_git_config",
        new=AsyncMock(return_value=("o/configured", None, "main")),
    ):
        with pytest.raises(DJInvalidInputException, match="does not match"):
            await snapshot_mod.resolve_server_pull(None, spec)


async def test_resolve_server_pull_requires_configured_repo():
    spec = DeploymentSpec(
        namespace="ns",
        source=_git_source(branch="main"),
    )
    with patch(
        "datajunction_server.internal.deployment.snapshot.resolve_git_config",
        new=AsyncMock(return_value=(None, None, None)),
    ):
        with pytest.raises(DJInvalidInputException, match="no github_repo_path"):
            await snapshot_mod.resolve_server_pull(None, spec)


async def test_resolve_server_pull_pins_branch_head():
    spec = DeploymentSpec(
        namespace="ns",
        source=_git_source(branch="feature/x"),
    )

    mock_github = MagicMock()
    mock_github.resolve_ref_to_sha = AsyncMock(return_value="sha1")
    mock_github.get_commit_author = AsyncMock(return_value=("A", "a@e.com"))
    mock_github.download_archive = AsyncMock(return_value=b"unused")
    with (
        patch(
            "datajunction_server.internal.deployment.snapshot.resolve_git_config",
            new=AsyncMock(return_value=("o/r", None, "feature/x")),
        ),
        patch.object(snapshot_mod, "GitHubService", return_value=mock_github),
        patch(
            "datajunction_server.internal.deployment.snapshot.fetch_git_spec_dict",
            new=AsyncMock(
                return_value={
                    "namespace": "ns",
                    "nodes": [
                        {
                            "name": "x",
                            "node_type": "transform",
                            "query": "SELECT 1",
                        },
                    ],
                    "tags": [],
                    "hierarchies": [],
                },
            ),
        ) as mock_fetch,
    ):
        resolved = await snapshot_mod.resolve_server_pull(None, spec)

    assert resolved.source.commit_sha == "sha1"
    assert resolved.source.branch == "feature/x"
    assert resolved.source.repository == "o/r"
    assert resolved.source.commit_author_email == "a@e.com"
    assert resolved.nodes[0].rendered_name == "ns.x"
    mock_fetch.assert_awaited_once()
    mock_github.resolve_ref_to_sha.assert_awaited_once_with("o/r", "feature/x")


def test_normalize_repo_path_variants():
    assert snapshot_mod.normalize_repo_path("owner/repo") == "owner/repo"
    assert (
        snapshot_mod.normalize_repo_path("https://github.com/owner/repo")
        == "owner/repo"
    )
    assert (
        snapshot_mod.normalize_repo_path(
            "https://github.netflix.net/corp/repo.git",
        )
        == "corp/repo"
    )
    assert snapshot_mod.normalize_repo_path("git@host:corp/repo") == "corp/repo"
