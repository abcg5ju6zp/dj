"""
End-to-end tests for pinned git deployments: snapshot pinning, stale
detection, lock arbitration, idempotent create, cancel/resume, restart
recovery, and preserved legacy semantics.
"""

import asyncio
import io
import tarfile
import uuid as uuid_mod
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server.api.deployments import recover_pinned_deployments
from datajunction_server.database.deployment import Deployment
from datajunction_server.database.deployment_lock import DeploymentNamespaceLock
from datajunction_server.database.user import User
from datajunction_server.internal.git.github_service import GitHubServiceError
from datajunction_server.models.deployment import (
    ColumnSpec,
    DeploymentProgress,
    DeploymentSnapshot,
    DeploymentSpec,
    DeploymentStatus,
    GitDeploymentSource,
    NodeProgress,
    SourceSpec,
)
from tests.conftest import patch_session_contexts


@asynccontextmanager
async def direct_session(session_factory):
    """A session bound to the function-scoped test database."""
    session: AsyncSession = await session_factory()
    try:
        yield session
    finally:
        await session.close()


async def _wait(client: AsyncClient, deployment_id: str, timeout: float = 60) -> dict:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        resp = await client.get(f"/deployments/{deployment_id}")
        data = resp.json()
        if data["status"] in ("success", "failed", "stale", "cancelled"):
            return data
        if loop.time() > deadline:
            raise AssertionError(f"deployment {deployment_id} stuck: {data}")
        await asyncio.sleep(0.1)


async def _setup_managed(client: AsyncClient, namespace: str) -> None:
    """Flat repo owner: a managed namespace."""
    assert (await client.post(f"/namespaces/{namespace}")).status_code < 300
    resp = await client.patch(
        f"/namespaces/{namespace}/git",
        json={
            "github_repo_path": "myorg/myrepo",
            "git_branch": "main",
        },
    )
    assert resp.status_code == 200, resp.text


def _node_spec() -> SourceSpec:
    return SourceSpec(
        name="test_source",
        description="Test source",
        catalog="default",
        schema="test",
        table="test_table",
        columns=[
            ColumnSpec(name="id", type="int"),
            ColumnSpec(name="name", type="string"),
        ],
    )


def _git_spec(namespace: str, sha: str = "c1") -> DeploymentSpec:
    return DeploymentSpec(
        namespace=namespace,
        nodes=[_node_spec()],
        source=GitDeploymentSource(
            repository="myorg/myrepo",
            branch="main",
            commit_sha=sha,
        ),
    )


def _github_mock(resolve_sha="c1") -> MagicMock:
    mock = MagicMock()
    mock.verify_commit = AsyncMock(return_value=True)
    mock.resolve_ref_to_sha = AsyncMock(return_value=resolve_sha)
    return mock


def _tarball(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        root = tarfile.TarInfo(name="myorg-myrepo-sha")
        root.type = tarfile.DIRTYPE
        root.mode = 0o755
        tar.addfile(root)
        for path, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=f"myorg-myrepo-sha/{path}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


async def _insert_pinned(
    session: AsyncSession,
    *,
    namespace: str,
    status: DeploymentStatus,
    sha: str = "c1",
    with_lock: bool = False,
) -> str:
    deployment_id = str(uuid_mod.uuid4())
    snapshot = DeploymentSnapshot(
        repository="myorg/myrepo",
        branch="main",
        commit_sha=sha,
        spec_digest="0" * 64,
        pinned_at=datetime.now(UTC),
    )
    user = await User.get_by_username(session, "dj")
    row = Deployment(
        uuid=UUID(deployment_id),
        namespace=namespace,
        spec={
            "namespace": namespace,
            "nodes": [],
            "source": {
                "type": "git",
                "repository": "myorg/myrepo",
                "branch": "main",
                "commit_sha": sha,
            },
        },
        results={},
        status=status,
        created_by_id=user.id,
        deployment_snapshot=snapshot,
        deployment_progress=DeploymentProgress(),
        lock_token=1 if with_lock else None,
    )
    session.add(row)
    await session.flush()
    if with_lock:
        session.add(
            DeploymentNamespaceLock(
                namespace=namespace,
                owner_deployment_id=UUID(deployment_id),
                fencing_token=1,
                lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
            ),
        )
    await session.commit()
    return deployment_id


@pytest.mark.xdist_group(name="recovery")
class TestPinnedDeploymentRecovery:
    @pytest.mark.asyncio
    async def test_pinned_deployment_succeeds_with_snapshot_and_progress(
        self,
        client: AsyncClient,
    ):
        namespace = "recovery_pinned_success"
        await _setup_managed(client, namespace)
        with patch(
            "datajunction_server.api.deployments.GitHubService",
            return_value=_github_mock(),
        ):
            resp = await client.post(
                "/deployments",
                json=_git_spec(namespace, sha="c1").model_dump(mode="json"),
            )
            assert resp.status_code == 200, resp.text
            created = resp.json()
            assert created["status"] in ("pending", "running")
            deployment_id = created["uuid"]
            final = await _wait(client, deployment_id)

        assert final["status"] == "success", final
        snapshot = final["snapshot"]
        assert snapshot["repository"] == "myorg/myrepo"
        assert snapshot["branch"] == "main"
        assert snapshot["commit_sha"] == "c1"
        assert len(snapshot["spec_digest"]) == 64
        assert snapshot["pinned_at"]
        assert isinstance(snapshot["impact"], dict)
        progress = final["progress"]
        assert f"{namespace}.test_source" in progress["nodes"]["applied"]
        assert progress["nodes"]["phase"] == "finalized"
        node = await client.get(f"/nodes/{namespace}.test_source/")
        assert node.status_code == 200

    @pytest.mark.asyncio
    async def test_lock_released_after_success(self, client, session_factory):
        namespace = "recovery_lock_release"
        await _setup_managed(client, namespace)
        with patch(
            "datajunction_server.api.deployments.GitHubService",
            return_value=_github_mock(),
        ):
            resp = await client.post(
                "/deployments",
                json=_git_spec(namespace).model_dump(mode="json"),
            )
            await _wait(client, resp.json()["uuid"])
        async with direct_session(session_factory) as session:
            assert await session.get(DeploymentNamespaceLock, namespace) is None

    @pytest.mark.asyncio
    async def test_stale_branch_deployment_goes_stale_without_changes(
        self,
        client: AsyncClient,
        session_factory,
    ):
        namespace = "recovery_stale"
        await _setup_managed(client, namespace)
        mock = _github_mock()
        mock.resolve_ref_to_sha = AsyncMock(return_value="c2")
        with patch(
            "datajunction_server.api.deployments.GitHubService",
            return_value=mock,
        ):
            resp = await client.post(
                "/deployments",
                json=_git_spec(namespace, sha="c1").model_dump(mode="json"),
            )
            deployment_id = resp.json()["uuid"]
            final = await _wait(client, deployment_id)

        assert final["status"] == "stale", final
        message = final["results"][0]["message"]
        assert "c1" in message and "c2" in message
        node = await client.get(f"/nodes/{namespace}.test_source/")
        assert node.status_code == 404
        async with direct_session(session_factory) as session:
            assert await session.get(DeploymentNamespaceLock, namespace) is None
        resume = await client.post(f"/deployments/{deployment_id}/resume")
        assert resume.status_code == 409

    @pytest.mark.asyncio
    async def test_concurrent_deployment_to_active_lock_returns_409(
        self,
        client: AsyncClient,
        session_factory,
    ):
        namespace = "recovery_lock_409"
        await _setup_managed(client, namespace)
        async with direct_session(session_factory) as session:
            owner_id = await _insert_pinned(
                session,
                namespace=namespace,
                status=DeploymentStatus.RUNNING,
                with_lock=True,
            )
        with patch(
            "datajunction_server.api.deployments.GitHubService",
            return_value=_github_mock(),
        ):
            resp = await client.post(
                "/deployments",
                json=_git_spec(namespace, sha="c1").model_dump(mode="json"),
            )
        assert resp.status_code == 409, resp.text
        assert owner_id in resp.json()["message"]
        async with direct_session(session_factory) as session:
            count = len(
                (
                    await session.execute(
                        Deployment.__table__.select().where(
                            Deployment.namespace == namespace,
                        ),
                    )
                ).fetchall(),
            )
        assert count == 1

    @pytest.mark.asyncio
    async def test_idempotent_create_returns_same_deployment(
        self,
        client: AsyncClient,
    ):
        namespace = "recovery_idempotent"
        assert (await client.post(f"/namespaces/{namespace}")).status_code < 300
        payload = _git_spec(namespace, sha="c1")
        # Unmanaged namespace + local source -> legacy path.
        payload.source = None
        body = payload.model_dump(mode="json")
        headers = {"Idempotency-Key": "fixed-key-123"}
        first = await client.post("/deployments", json=body, headers=headers)
        second = await client.post("/deployments", json=body, headers=headers)
        assert first.status_code == 200, first.text
        assert second.json()["uuid"] == first.json()["uuid"]
        listed = await client.get(f"/deployments?namespace={namespace}")
        assert len(listed.json()) == 1

    @pytest.mark.asyncio
    async def test_cancel_pending_releases_lock_and_terminal_is_409(
        self,
        client: AsyncClient,
        session_factory,
    ):
        namespace = "recovery_cancel"
        await _setup_managed(client, namespace)
        async with direct_session(session_factory) as session:
            deployment_id = await _insert_pinned(
                session,
                namespace=namespace,
                status=DeploymentStatus.PENDING,
                with_lock=True,
            )
        resp = await client.post(f"/deployments/{deployment_id}/cancel")
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "cancelled"
        async with direct_session(session_factory) as session:
            assert await session.get(DeploymentNamespaceLock, namespace) is None
        again = await client.post(f"/deployments/{deployment_id}/cancel")
        assert again.status_code == 409

    @pytest.mark.asyncio
    async def test_failed_deployment_resumes_and_applies_node_once(
        self,
        client: AsyncClient,
    ):
        namespace = "recovery_resume"
        await _setup_managed(client, namespace)
        mock = _github_mock()
        # First drift check: remote unreachable -> failed (resumable).
        # Second (resume): branch still on the pinned commit.
        mock.resolve_ref_to_sha = AsyncMock(
            side_effect=[
                GitHubServiceError(message="boom"),
                "c1",
            ],
        )
        with patch(
            "datajunction_server.api.deployments.GitHubService",
            return_value=mock,
        ):
            create = await client.post(
                "/deployments",
                json=_git_spec(namespace, sha="c1").model_dump(mode="json"),
            )
            deployment_id = create.json()["uuid"]
            failed = await _wait(client, deployment_id)
            assert failed["status"] == "failed"
            assert failed["progress"]["nodes"]["phase"] == "failed"

            resumed = await client.post(f"/deployments/{deployment_id}/resume")
            assert resumed.status_code == 200, resumed.text
            final = await _wait(client, deployment_id)

        assert final["status"] == "success", final
        node = await client.get(f"/nodes/{namespace}.test_source/")
        assert node.status_code == 200
        assert node.json()["version"] == "v1.0"

    @pytest.mark.asyncio
    async def test_recover_only_selects_pinned_nonterminal(
        self,
        client: AsyncClient,
        session_factory,
    ):
        async with direct_session(session_factory) as session:
            pending_id = await _insert_pinned(
                session,
                namespace="recovery_restart_pending",
                status=DeploymentStatus.PENDING,
            )
            running_id = await _insert_pinned(
                session,
                namespace="recovery_restart_running",
                status=DeploymentStatus.RUNNING,
            )
            await _insert_pinned(
                session,
                namespace="recovery_restart_success",
                status=DeploymentStatus.SUCCESS,
            )
            user = await User.get_by_username(session, "dj")
            for ns, status in (
                ("recovery_restart_local", DeploymentStatus.PENDING),
                ("recovery_restart_preview", DeploymentStatus.RUNNING),
            ):
                session.add(
                    Deployment(
                        uuid=uuid_mod.uuid4(),
                        namespace=ns,
                        spec={"namespace": ns, "nodes": []},
                        results={},
                        status=status,
                        created_by_id=user.id,
                    ),
                )
            await session.commit()

        # Keep the GitHub mock and session patches active until the recovered
        # tasks finish — they keep running as background tasks after recovery.
        with (
            patch(
                "datajunction_server.api.deployments.GitHubService",
                return_value=_github_mock(),
            ),
            patch_session_contexts(session_factory),
        ):
            recovered = await recover_pinned_deployments()
            assert set(recovered) == {pending_id, running_id}
            for deployment_id in recovered:
                await _wait(client, deployment_id)

    @pytest.mark.asyncio
    async def test_resume_after_lost_terminal_write_does_not_reapply(
        self,
        client: AsyncClient,
        session_factory,
    ):
        """AC-12: a committed node is not applied twice even if the terminal
        status write was lost (row re-marked FAILED with an applied set)."""
        namespace = "recovery_lost_write"
        await _setup_managed(client, namespace)
        with patch(
            "datajunction_server.api.deployments.GitHubService",
            return_value=_github_mock(),
        ):
            create = await client.post(
                "/deployments",
                json=_git_spec(namespace, sha="c1").model_dump(mode="json"),
            )
            await _wait(client, create.json()["uuid"])

        # Simulate a lost terminal write: another attempt for the same pinned
        # content is FAILED with the node already recorded as applied.
        async with direct_session(session_factory) as session:
            user = await User.get_by_username(session, "dj")
            snapshot = DeploymentSnapshot(
                repository="myorg/myrepo",
                branch="main",
                commit_sha="c1",
                spec_digest="0" * 64,
                pinned_at=datetime.now(UTC),
            )
            retry_id = str(uuid_mod.uuid4())
            spec = _git_spec(namespace, sha="c1")
            # force=True pushes an unchanged node into to_deploy; the
            # reconciliation barrier must still skip it as already applied.
            spec.force = True
            session.add(
                Deployment(
                    uuid=UUID(retry_id),
                    namespace=namespace,
                    spec=spec.model_dump(mode="json"),
                    results={},
                    status=DeploymentStatus.FAILED,
                    created_by_id=user.id,
                    deployment_snapshot=snapshot,
                    deployment_progress=DeploymentProgress(
                        nodes=NodeProgress(
                            phase="failed",
                            applied=[f"{namespace}.test_source"],
                            attempted=[f"{namespace}.test_source"],
                        ),
                    ),
                ),
            )
            await session.commit()

        with patch(
            "datajunction_server.api.deployments.GitHubService",
            return_value=_github_mock(),
        ):
            resumed = await client.post(f"/deployments/{retry_id}/resume")
            assert resumed.status_code == 200, resumed.text
            final = await _wait(client, retry_id)

        assert final["status"] == "success", final
        skipped = [
            r
            for r in final["results"]
            if "already applied" in r["message"]
        ]
        assert skipped, final
        node = await client.get(f"/nodes/{namespace}.test_source/")
        assert node.json()["version"] == "v1.0"

    @pytest.mark.asyncio
    async def test_server_pull_pins_branch_head_and_executes(
        self,
        client: AsyncClient,
    ):
        namespace = "recovery_server_pull"
        await _setup_managed(client, namespace)
        node_yaml = """
name: ${prefix}test_source
node_type: source
description: Pulled source
catalog: default
schema_: test
table: test_table
columns:
  - name: id
    type: int
"""
        tarball = _tarball({"test_source.yaml": node_yaml})
        mock = MagicMock()
        mock.resolve_ref_to_sha = AsyncMock(return_value="pullsha1")
        mock.get_commit_author = AsyncMock(
            return_value=("Alice", "alice@example.com"),
        )
        mock.download_archive = AsyncMock(return_value=tarball)
        mock.verify_commit = AsyncMock(return_value=True)
        with (
            patch(
                "datajunction_server.internal.deployment.snapshot.GitHubService",
                return_value=mock,
            ),
            patch(
                "datajunction_server.api.deployments.GitHubService",
                return_value=mock,
            ),
        ):
            spec = DeploymentSpec(
                namespace=namespace,
                nodes=[],
                source=GitDeploymentSource(
                    repository="myorg/myrepo",
                    branch="main",
                ),
            )
            resp = await client.post(
                "/deployments",
                json=spec.model_dump(mode="json"),
            )
            assert resp.status_code == 200, resp.text
            deployment_id = resp.json()["uuid"]
            final = await _wait(client, deployment_id)

        assert final["status"] == "success", final
        assert final["snapshot"]["commit_sha"] == "pullsha1"
        assert final["source"]["commit_sha"] == "pullsha1"
        # Pulled once at creation; executor never re-downloads content.
        assert mock.download_archive.await_count == 1
        node = await client.get(f"/nodes/{namespace}.test_source/")
        assert node.status_code == 200

    @pytest.mark.asyncio
    async def test_sync_from_git_takes_and_releases_the_lock(
        self,
        client: AsyncClient,
        session_factory,
    ):
        namespace = "recovery_sync_lock"
        assert (await client.post(f"/namespaces/{namespace}")).status_code < 300
        await client.patch(
            f"/namespaces/{namespace}/git",
            json={
                "github_repo_path": "myorg/myrepo",
                "git_branch": "main",
            },
        )
        node_yaml = """
name: ${prefix}test_source
node_type: source
catalog: default
schema_: test
table: test_table
columns:
  - name: id
    type: int
"""
        tarball = _tarball({"test_source.yaml": node_yaml})
        mock = MagicMock()
        mock.resolve_ref_to_sha = AsyncMock(return_value="abc123")
        mock.get_commit_author = AsyncMock(return_value=(None, None))
        mock.download_archive = AsyncMock(return_value=tarball)
        with patch(
            "datajunction_server.api.git_sync.GitHubService",
            return_value=mock,
        ):
            resp = await client.post(f"/namespaces/{namespace}/sync-from-git")
            assert resp.status_code == 200, resp.text
        async with direct_session(session_factory) as session:
            assert await session.get(DeploymentNamespaceLock, namespace) is None

    @pytest.mark.asyncio
    async def test_sync_from_git_conflicts_with_active_lock(
        self,
        client: AsyncClient,
        session_factory,
    ):
        namespace = "recovery_sync_409"
        assert (await client.post(f"/namespaces/{namespace}")).status_code < 300
        await client.patch(
            f"/namespaces/{namespace}/git",
            json={
                "github_repo_path": "myorg/myrepo",
                "git_branch": "main",
            },
        )
        async with direct_session(session_factory) as session:
            await _insert_pinned(
                session,
                namespace=namespace,
                status=DeploymentStatus.RUNNING,
                with_lock=True,
            )
        node_yaml = """
name: ${prefix}test_source
node_type: source
catalog: default
schema_: test
table: test_table
columns:
  - name: id
    type: int
"""
        mock = MagicMock()
        mock.resolve_ref_to_sha = AsyncMock(return_value="abc123")
        mock.get_commit_author = AsyncMock(return_value=(None, None))
        mock.download_archive = AsyncMock(
            return_value=_tarball({"test_source.yaml": node_yaml}),
        )
        with patch(
            "datajunction_server.api.git_sync.GitHubService",
            return_value=mock,
        ):
            resp = await client.post(f"/namespaces/{namespace}/sync-from-git")
        assert resp.status_code == 409, resp.text

    @pytest.mark.asyncio
    async def test_local_deployment_has_no_snapshot_or_lock(
        self,
        client: AsyncClient,
        session_factory,
    ):
        namespace = "recovery_local"
        assert (await client.post(f"/namespaces/{namespace}")).status_code < 300
        spec = DeploymentSpec(namespace=namespace, nodes=[_node_spec()])
        resp = await client.post("/deployments", json=spec.model_dump(mode="json"))
        assert resp.status_code == 200
        final = await _wait(client, resp.json()["uuid"])
        assert final["status"] == "success"
        assert final["snapshot"] is None
        async with direct_session(session_factory) as session:
            assert await session.get(DeploymentNamespaceLock, namespace) is None
