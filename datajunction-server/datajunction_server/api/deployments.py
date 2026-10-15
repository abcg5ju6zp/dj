"""
Bulk deployment APIs.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from uuid import UUID

from fastapi import BackgroundTasks, Depends, Request, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from datajunction_server.database.deployment import Deployment
from datajunction_server.database.namespace import NodeNamespace
from datajunction_server.database.user import User
from datajunction_server.errors import (
    DJClientUpgradeRequiredException,
    DJDeploymentConflictException,
    DJDoesNotExistException,
    DJError,
    DJInvalidInputException,
)
from datajunction_server.instrumentation.provider import get_metrics_provider
from datajunction_server.internal.access.authentication.http import SecureAPIRouter
from datajunction_server.internal.access.authorization import (
    AccessChecker,
    AccessDenialMode,
    get_access_checker,
)
from datajunction_server.internal.caching.cachelib_cache import get_cache
from datajunction_server.internal.caching.interface import Cache
from datajunction_server.internal.deployment import locks as deployment_locks
from datajunction_server.internal.deployment.deployment import deploy
from datajunction_server.internal.deployment.orchestrator import (
    DeploymentFencedOrCanceled,
    DeploymentOrchestrator,
)
from datajunction_server.internal.deployment.snapshot import (
    build_deployment_snapshot,
    build_impact_snapshot,
    check_remote_drift,
    is_server_pull,
    managed_namespace,
    resolve_server_pull,
    stale_result,
)
from datajunction_server.internal.deployment.snapshot import (
    normalize_repo_path as _normalize_repo_path,
)
from datajunction_server.internal.deployment.utils import DeploymentContext
from datajunction_server.internal.git.github_service import (
    GitHubService,
    GitHubServiceError,
)
from datajunction_server.internal.namespaces import resolve_git_config
from datajunction_server.models import access
from datajunction_server.models.deployment import (
    DeploymentInfo,
    DeploymentProgress,
    DeploymentResult,
    DeploymentSnapshot,
    DeploymentSourceType,
    DeploymentSpec,
    DeploymentStatus,
    GitDeploymentSource,
    LocalDeploymentSource,
    LockProgress,
    SnapshotImpact,
)
from datajunction_server.service_clients import QueryServiceClient
from datajunction_server.utils import (
    get_current_user,
    get_query_service_client,
    get_session,
    get_settings,
    session_context,
)

logger = logging.getLogger(__name__)
settings = get_settings()
router = SecureAPIRouter(tags=["deployments"])


async def _verify_git_deployment(
    session: AsyncSession,
    deployment_spec: DeploymentSpec,
    namespace_obj: NodeNamespace | None,
) -> None:
    """
    Verify that a deployment to a git_only namespace meets requirements:
    1. Must have source.type = git
    2. Must include commit_sha
    3. Commit must exist in the configured repository

    Raises DJInvalidInputException if requirements not met.
    """
    # Check source type
    if not deployment_spec.source:
        raise DJInvalidInputException(
            message=f"Namespace '{deployment_spec.namespace}' is git-only. "
            "Deployments must include a git source with commit_sha.",
        )

    if deployment_spec.source.type != DeploymentSourceType.GIT:
        raise DJInvalidInputException(
            message=f"Namespace '{deployment_spec.namespace}' is git-only. "
            "Deployments must have source.type='git', not "
            f"'{deployment_spec.source.type}'.",
        )

    # Check commit_sha is provided
    if not deployment_spec.source.commit_sha:
        raise DJInvalidInputException(
            message=f"Namespace '{deployment_spec.namespace}' is git-only. "
            "Deployments must include source.commit_sha.",
        )

    # Resolve git config (may be inherited from parent for branch namespaces)
    github_repo_path, _, _ = await resolve_git_config(
        session,
        deployment_spec.namespace,
    )

    # Verify commit exists in the configured repository. A namespace can be
    # git_only without a github_repo_path (e.g. auto-locked on first git deploy,
    # or a flat namespace locked via PATCH). There's no repo to verify the
    # commit against, so skip verification rather than hard-failing the deploy —
    # the lock is about blocking UI edits, and this deploy is already git-sourced.
    if not github_repo_path:
        logger.info(
            "Namespace %s is git-only with no github_repo_path; "
            "skipping commit verification.",
            deployment_spec.namespace,
        )
        return

    try:
        github = GitHubService()
        commit_exists = await github.verify_commit(
            repo_path=github_repo_path,
            commit_sha=deployment_spec.source.commit_sha,
        )
        if not commit_exists:
            raise DJInvalidInputException(
                message=f"Commit '{deployment_spec.source.commit_sha}' not found in "
                f"repository '{github_repo_path}'. "
                "Deployments to git-only namespaces must reference valid commits.",
            )
    except GitHubServiceError as e:
        raise DJInvalidInputException(
            message=f"Failed to verify commit: {e.message}",
        ) from e

    logger.info(
        "Verified git deployment to %s: commit %s exists in %s",
        deployment_spec.namespace,
        deployment_spec.source.commit_sha[:8],
        namespace_obj.github_repo_path if namespace_obj else github_repo_path,
    )


async def _maybe_autolock_git_namespace(deployment_spec: DeploymentSpec) -> None:
    """
    Default-on-git-push: when a namespace is deployed from git, record the repo it
    tracks (``github_repo_path`` + ``git_branch``) so it becomes a *repo owner* and
    is UI-locked via ``is_repo_owner`` — the single "git-managed ⇒ read-only"
    signal. No ``git_only`` write, so unlock is a config change (detach the repo).

    Guards:
    - Only a git deploy carrying a ``commit_sha`` — a reproducible, CI-driven
      deployment. Looser branch-only git deploys are left editable.
    - Only a non-branch namespace (no ``parent_namespace``) — branch namespaces
      are meant to be git-deployed *and* editable (edit, sync back).
    - Only when ``github_repo_path`` is unset — never clobber an intentional
      config on later deploys.
    - ``git_branch`` is set to the deployed branch (non-null) so the namespace is
      a *flat* repo owner, not mistaken for a child-spawning git root.

    Called only after a successful deploy, so the namespace already exists.
    """
    src = deployment_spec.source
    if (
        not src
        or src.type != DeploymentSourceType.GIT
        or not getattr(src, "commit_sha", None)
        or not getattr(src, "repository", None)
    ):
        return
    async with session_context() as session:
        ns = await NodeNamespace.get(
            session,
            deployment_spec.namespace,
            raise_if_not_exists=False,
        )
        if (
            ns is None
            or ns.parent_namespace is not None
            or ns.github_repo_path is not None
        ):
            return
        ns.github_repo_path = _normalize_repo_path(src.repository)
        ns.git_branch = getattr(src, "branch", None) or ns.git_branch
        session.add(ns)
        await session.commit()


IDEMPOTENCY_HEADER = "Idempotency-Key"

# Result statuses that still count as an overall successful deployment.
_SUCCESS_STATUSES = {
    DeploymentResult.Status.SUCCESS,
    DeploymentResult.Status.SKIPPED,
    DeploymentResult.Status.INVALID,
    DeploymentResult.Status.WARNING,
}


class DeploymentExecutor(ABC):
    @abstractmethod
    async def submit(
        self,
        spec: DeploymentSpec,
        context: DeploymentContext,
        dry_run: bool = False,
        *,
        snapshot: DeploymentSnapshot | None = None,
        idempotency_key: str | None = None,
        lock_token: int | None = None,
    ) -> str:
        """
        Persist the deployment row and kick the job off asynchronously.

        Pinned git deployments pass ``snapshot`` (and the ``lock_token`` of the
        managed-namespace lock already acquired in the request transaction);
        their executor reconstructs everything from the persisted row.
        """
        ...  # pragma: no cover


class InProcessExecutor(DeploymentExecutor):
    """
    Runs deployments as asyncio tasks in this process.

    Two execution paths:

    * **Legacy** (local source, impact dry-run): behaves as before — the
      in-memory spec/context drive the run; not recovered after restart.
    * **Pinned git** (``snapshot`` column set): the run loads the deployment
      row as its sole input, checks remote drift, fences on a durable
      namespace lock, checkpoints authorization/node progress, and is
      resumable by ``recover_pinned_deployments`` after a restart.
    """

    def __init__(self):
        self.statuses: dict[str, DeploymentStatus] = {}
        # Deployments already scheduled in this process (best-effort guard).
        self._inflight: set[str] = set()
        # Strong references to in-flight asyncio tasks (the loop only keeps
        # weak ones); keyed by deployment id.
        self._tasks: dict[str, asyncio.Task] = {}

    async def submit(
        self,
        spec: DeploymentSpec,
        context: DeploymentContext,
        dry_run: bool = False,
        *,
        snapshot: DeploymentSnapshot | None = None,
        idempotency_key: str | None = None,
        lock_token: int | None = None,
    ) -> str:
        if idempotency_key:
            existing = await self._find_idempotent_row(idempotency_key)
            if existing is not None:
                # Response-lost retry: return the original deployment and make
                # sure it is scheduled in this process (only while active),
                # never create or re-run a second time.
                if existing.status not in DeploymentStatus.terminal():
                    self._schedule(str(existing.uuid), dry_run=False, context=context)
                return str(existing.uuid)

        deployment_uuid = str(uuid.uuid4())
        progress = None
        if snapshot is not None:
            progress = DeploymentProgress(
                lock=(
                    LockProgress(fencing_token=lock_token)
                    if lock_token is not None
                    else None
                ),
            )
        async with session_context() as session:
            deployment = Deployment(
                uuid=UUID(deployment_uuid),
                namespace=spec.namespace,
                spec=spec.model_dump(),
                status=DeploymentStatus.PENDING,
                created_by_id=context.current_user.id,
                idempotency_key=idempotency_key,
                deployment_snapshot=snapshot,
                deployment_progress=progress,
                started_at=datetime.now(UTC) if snapshot is not None else None,
            )
            session.add(deployment)
            try:
                await session.commit()
            except IntegrityError:
                # Concurrent insert with the same idempotency key.
                await session.rollback()
                existing_id = await self._find_idempotent(
                    idempotency_key or "",
                )
                if existing_id is not None:
                    self._schedule(existing_id, dry_run=False, context=context)
                    return existing_id
                raise

        if snapshot is not None:
            self._schedule(deployment_uuid)
        else:
            self._schedule(deployment_uuid, dry_run=dry_run, context=context)
        return deployment_uuid

    async def create_pinned_row(
        self,
        spec: DeploymentSpec,
        context: DeploymentContext,
        idempotency_key: str | None,
    ) -> tuple[str, bool]:
        """
        Persist a pinned deployment row and acquire the managed-namespace
        lock in the SAME transaction (an active holder raises 409 and no row
        remains). Returns ``(deployment_id, created)``; an idempotent replay
        returns the original deployment with ``created=False``.
        """
        if idempotency_key:
            existing = await self._find_idempotent_row(idempotency_key)
            if existing is not None:
                if existing.status not in DeploymentStatus.terminal():
                    self._schedule(str(existing.uuid))
                return str(existing.uuid), False

        deployment_uuid = str(uuid.uuid4())
        async with session_context() as session:
            deployment = Deployment(
                uuid=UUID(deployment_uuid),
                namespace=spec.namespace,
                spec=spec.model_dump(),
                status=DeploymentStatus.PENDING,
                created_by_id=context.current_user.id,
                idempotency_key=idempotency_key,
                started_at=datetime.now(UTC),
                deployment_progress=DeploymentProgress(),
            )
            session.add(deployment)
            await session.flush()
            token = await deployment_locks.acquire_lock(
                session,
                spec.namespace,
                UUID(deployment_uuid),
            )
            deployment.lock_token = token
            deployment.deployment_progress = DeploymentProgress(
                lock=LockProgress(fencing_token=token),
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                existing_id = await self._find_idempotent(idempotency_key or "")
                if existing_id is not None:
                    self._schedule(existing_id)
                    return existing_id, False
                raise
        return deployment_uuid, True

    async def attach_snapshot_and_schedule(
        self,
        deployment_id: str,
        spec: DeploymentSpec,
        impact: SnapshotImpact | None,
    ) -> None:
        """Freeze the create-time impact snapshot, then schedule execution."""
        snapshot = build_deployment_snapshot(spec, impact=impact)
        async with session_context() as session:
            row = await session.get(Deployment, UUID(deployment_id))
            if row is not None and not row.is_terminal:
                row.deployment_snapshot = snapshot
                flag_modified(row, "snapshot")
                await session.commit()
        self._schedule(deployment_id)

    @staticmethod
    async def _find_idempotent(idempotency_key: str) -> str | None:
        row = await InProcessExecutor._find_idempotent_row(idempotency_key)
        return str(row.uuid) if row is not None else None

    @staticmethod
    async def _find_idempotent_row(
        idempotency_key: str,
    ) -> Deployment | None:
        if not idempotency_key:
            return None
        async with session_context() as session:
            return await session.scalar(
                select(Deployment).where(
                    Deployment.idempotency_key == idempotency_key,
                ),
            )

    def _schedule(
        self,
        deployment_id: str,
        *,
        dry_run: bool = False,
        context: DeploymentContext | None = None,
    ) -> None:
        if deployment_id in self._inflight:
            return
        self._inflight.add(deployment_id)

        def _discard(task, did=deployment_id) -> None:
            self._inflight.discard(did)
            self._tasks.pop(did, None)

        task = asyncio.create_task(
            self._task(
                deployment_id,
                dry_run=dry_run,
                context=context,
            ),
        )
        self._tasks[deployment_id] = task
        task.add_done_callback(_discard)

    async def _task(
        self,
        deployment_id: str,
        *,
        dry_run: bool,
        context: DeploymentContext | None,
    ) -> None:
        try:
            if dry_run:
                async with session_context() as session:
                    row = await session.get(Deployment, UUID(deployment_id))
                    spec = DeploymentSpec(**row.spec)
                await self._run_deployment(
                    deployment_id=deployment_id,
                    deployment_spec=spec,
                    context=context,
                    dry_run=True,
                )
                return
            pinned = await self._is_pinned(deployment_id)
            if pinned:
                await self._run_pinned(deployment_id)
            else:
                async with session_context() as session:
                    row = await session.get(Deployment, UUID(deployment_id))
                    spec = DeploymentSpec(**row.spec)
                await self._run_deployment(
                    deployment_id=deployment_id,
                    deployment_spec=spec,
                    context=context,
                )
        except Exception:
            logger.exception(
                "Unhandled error running deployment task %s",
                deployment_id,
            )

    @staticmethod
    async def _is_pinned(deployment_id: str) -> bool:
        async with session_context() as session:
            row = await session.get(Deployment, UUID(deployment_id))
            return bool(row is not None and row.is_pinned)

    @staticmethod
    async def update_status(
        deployment_uuid: str,
        status: DeploymentStatus,
        results: list[DeploymentResult] | None = None,
        downstream_impacts: list | None = None,
        warnings: list[DJError] | None = None,
        check_results: list | None = None,
    ):
        async with session_context() as session:
            deployment = await session.get(Deployment, deployment_uuid)
            if deployment is None:
                # Deployment record doesn't exist (e.g., internal deployment)
                return
            deployment.status = status
            if results is not None:
                deployment.results = [r.model_dump() for r in results]
            if warnings is not None:
                deployment.deployment_warnings = warnings
            if downstream_impacts is not None:
                deployment.downstream_impacts = [
                    d.model_dump() for d in downstream_impacts
                ]
            if check_results is not None:
                deployment.deployment_check_results = check_results
            await session.commit()

    async def _run_deployment(
        self,
        deployment_id: str,
        deployment_spec: DeploymentSpec,
        context: DeploymentContext | None,
        dry_run: bool = False,
    ):
        """Legacy execution path (local deploys, impact dry-runs).

        Kept for callers that drive an in-memory spec; pinned git deployments
        go through ``_run_pinned`` and never land here.
        """
        if context is None:
            context = await self._fallback_context(deployment_id)
        await InProcessExecutor.update_status(deployment_id, DeploymentStatus.RUNNING)

        try:
            async with session_context() as session:
                execute_result = await deploy(
                    session=session,
                    deployment_id=deployment_id,
                    deployment=deployment_spec,
                    context=context,
                    dry_run=dry_run,
                )
                results = execute_result.results
                final_status = (
                    DeploymentStatus.SUCCESS
                    if all(r.status in _SUCCESS_STATUSES for r in results)
                    else DeploymentStatus.FAILED
                )
                await InProcessExecutor.update_status(
                    deployment_id,
                    final_status,
                    results,
                    downstream_impacts=execute_result.downstream_impacts,
                    warnings=execute_result.warnings,
                    check_results=execute_result.check_results,
                )
                if final_status == DeploymentStatus.SUCCESS and not dry_run:
                    await _maybe_autolock_git_namespace(deployment_spec)
        except Exception as exc:
            logger.error("Deployment %s failed: %s", deployment_id, exc, exc_info=True)
            await InProcessExecutor.update_status(
                deployment_id,
                DeploymentStatus.FAILED,
                [
                    DeploymentResult(
                        name=exc.__class__.__name__,
                        deploy_type=DeploymentResult.Type.GENERAL,
                        message=str(exc),
                        status=DeploymentResult.Status.FAILED,
                        operation=DeploymentResult.Operation.UNKNOWN,
                    ),
                ],
            )

    # ------------------------------------------------------------------
    # Pinned git deployment path
    # ------------------------------------------------------------------

    @staticmethod
    async def _fallback_context(deployment_id: str) -> DeploymentContext:
        """Reconstruct a background context from the persisted creator."""
        async with session_context() as session:
            row = await session.get(Deployment, UUID(deployment_id))
            user = await session.get(User, row.created_by_id)
        return DeploymentContext(
            current_user=user,
            query_service_client=get_query_service_client(get_settings()),
        )

    async def _run_pinned(self, deployment_id: str) -> None:
        """Execute a pinned git deployment, consuming only its persisted row.

        Drift is detected before taking the lock; the lock/fencing token is
        acquired in its own transaction; node application and the applied-set
        progress commit together; terminal state is written after.
        """
        deployment_uuid = UUID(deployment_id)

        async with session_context() as session:
            row = await session.get(Deployment, deployment_uuid)
            if row is None:
                return
            # A resume flips FAILED back to RUNNING before scheduling; any
            # other terminal state means this run is over.
            if row.is_terminal and row.status != DeploymentStatus.RUNNING:
                return
            spec = DeploymentSpec(**row.spec)
            snapshot = row.deployment_snapshot
            create_lock_token = row.lock_token
            namespace = row.namespace

        # 1. Remote drift check — stale deployments never (re)acquire the
        # lock; release any create-time lock held by this deployment.
        # Drift is checked against the namespace's CONFIGURED repo only: a
        # git_only namespace with no repo attached skips verification and
        # drift alike.
        source = spec.source
        async with session_context() as config_session:
            configured_repo, _, _ = await resolve_git_config(
                config_session,
                namespace,
            )
        if (
            isinstance(source, GitDeploymentSource)
            and source.branch
            and configured_repo
        ):
            github = GitHubService()
            try:
                actual = await check_remote_drift(source, github=github)
            except GitHubServiceError as exc:
                # Transient: fail (resumable), never declare stale blindly.
                # Release before terminalizing so a reader of FAILED never
                # sees the lock still held.
                await self._safe_release(
                    namespace,
                    deployment_uuid,
                    create_lock_token,
                )
                await self._fail_pinned(
                    deployment_id,
                    f"could not verify remote branch: {exc.message}",
                    planned=[],
                )
                return
            if actual:
                await self._safe_release(
                    namespace,
                    deployment_uuid,
                    create_lock_token,
                )
                await self._mark_stale(deployment_id, snapshot, actual)
                return

        # 2. Acquire the managed-namespace lock and go RUNNING.
        async with session_context() as session:
            row = await session.get(Deployment, deployment_uuid)
            if row is None or row.is_terminal:
                return
            namespace = row.namespace
            try:
                token = await deployment_locks.acquire_lock(
                    session,
                    namespace,
                    deployment_uuid,
                )
            except DJDeploymentConflictException as exc:
                await self._fail_pinned(
                    deployment_id,
                    exc.message,
                    planned=[],
                )
                return
            progress = row.deployment_progress or DeploymentProgress()
            progress.lock = LockProgress(fencing_token=token)
            row.deployment_progress = progress
            row.lock_token = token
            row.status = DeploymentStatus.RUNNING
            row.started_at = row.started_at or datetime.now(UTC)
            await session.commit()

        # 3. Heartbeat while the deployment runs.
        stop_event = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._heartbeat(namespace, deployment_id, token, stop_event),
        )
        try:
            await self._execute_pinned(deployment_id, token, namespace)
        finally:
            stop_event.set()
            heartbeat.cancel()
            async with session_context() as session:
                # CAS: releases only while still the owner. An older fenced
                # deployment can never release a newer owner's lock.
                await deployment_locks.release_lock(
                    session,
                    namespace,
                    deployment_uuid,
                    token,
                )
                await session.commit()

    @staticmethod
    async def _safe_release(namespace: str, deployment_id, token: int | None):
        """Release a lock if this deployment still owns the token (no-op on
        a fenced owner or when no create-time lock was held)."""
        if token is None:
            return
        owner = deployment_id if isinstance(deployment_id, UUID) else UUID(
            str(deployment_id),
        )
        async with session_context() as session:
            await deployment_locks.release_lock(
                session,
                namespace,
                owner,
                token,
            )
            await session.commit()

    async def _heartbeat(
        self,
        namespace: str,
        deployment_id: str,
        token: int,
        stop_event: asyncio.Event,
    ) -> None:
        interval = max(1, get_settings().deployment_lock_heartbeat_seconds)
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                return
            except TimeoutError:
                pass
            async with session_context() as session:
                renewed = await deployment_locks.renew_lock(
                    session,
                    namespace,
                    UUID(deployment_id),
                    token,
                )
                await session.commit()
            if not renewed:
                logger.warning(
                    "Deployment %s lost its lock fence (token=%s); stopping",
                    deployment_id,
                    token,
                )
                stop_event.set()
                return

    async def _checkpoint(
        self,
        deployment_id: str,
        token: int,
        progress: DeploymentProgress,
    ) -> str:
        """Persist progress in a short, fence-aware transaction.

        Returns ``"ok"``, ``"fenced"`` (lock lost to a newer deployment) or
        ``"cancelled"`` (the deployment was cancelled while running).
        """
        async with session_context() as session:
            row = await session.get(Deployment, UUID(deployment_id))
            if row is None:
                return "fenced"
            if row.status == DeploymentStatus.CANCELLED:
                return "cancelled"
            lock = await deployment_locks.get_lock(session, row.namespace)
            if (
                lock is None
                or str(lock.owner_deployment_id) != deployment_id
                or lock.fencing_token != token
            ):
                return "fenced"
            row.deployment_progress = progress
            flag_modified(row, "progress")
            await session.commit()
        return "ok"

    async def _execute_pinned(
        self,
        deployment_id: str,
        token: int,
        namespace: str,
    ) -> None:
        async with session_context() as session:
            row = await session.get(Deployment, UUID(deployment_id))
            if row is None or row.is_terminal:
                return
            initial_progress = row.deployment_progress or DeploymentProgress()
            spec = DeploymentSpec(**row.spec)
            user = await session.get(User, row.created_by_id)
            context = DeploymentContext(
                current_user=user,
                query_service_client=get_query_service_client(get_settings()),
            )
            orchestrator = DeploymentOrchestrator(
                deployment_id=deployment_id,
                deployment_spec=spec,
                session=session,
                context=context,
                dry_run=False,
                initial_progress=initial_progress,
                deployment_row=row,
                progress_checkpoint=lambda progress: self._checkpoint(
                    deployment_id,
                    token,
                    progress,
                ),
            )
            try:
                execute_result = await orchestrator.execute()
            except DeploymentFencedOrCanceled as signal:
                await self._handle_abort(deployment_id, signal)
                return
            except Exception as exc:
                logger.error(
                    "Pinned deployment %s failed: %s",
                    deployment_id,
                    exc,
                    exc_info=True,
                )
                await self._fail_pinned(
                    deployment_id,
                    str(exc),
                    planned=list(orchestrator.planned_names),
                )
                return
            planned = list(orchestrator.planned_names)

        await self._finalize_pinned(
            deployment_id,
            token,
            execute_result,
            planned,
        )
        if all(r.status in _SUCCESS_STATUSES for r in execute_result.results):
            await _maybe_autolock_git_namespace(spec)

    async def _finalize_pinned(
        self,
        deployment_id: str,
        token: int,
        execute_result,
        planned: list[str],
    ) -> None:
        results = execute_result.results
        overall_success = all(r.status in _SUCCESS_STATUSES for r in results)
        async with session_context() as session:
            row = await session.get(Deployment, UUID(deployment_id))
            if row is None or row.is_terminal:
                return
            progress = row.deployment_progress or DeploymentProgress()
            applied = set(progress.nodes.applied)
            progress.nodes.applied = sorted(applied)
            progress.nodes.attempted = sorted(set(planned))
            progress.nodes.pending_rollback = sorted(set(planned) - applied)
            progress.nodes.phase = (
                "finalized" if overall_success else "failed"
            )
            row.deployment_progress = progress
            flag_modified(row, "progress")
            row.status = (
                DeploymentStatus.SUCCESS
                if overall_success
                else DeploymentStatus.FAILED
            )
            row.results = [r.model_dump() for r in results]
            row.deployment_warnings = execute_result.warnings
            row.downstream_impacts = [
                impact.model_dump() for impact in execute_result.downstream_impacts
            ]
            row.deployment_check_results = execute_result.check_results
            row.finished_at = datetime.now(UTC)
            await session.commit()

    async def _fail_pinned(
        self,
        deployment_id: str,
        message: str,
        planned: list[str],
    ) -> None:
        async with session_context() as session:
            row = await session.get(Deployment, UUID(deployment_id))
            if row is None or row.is_terminal:
                return
            progress = row.deployment_progress or DeploymentProgress()
            applied = set(progress.nodes.applied)
            planned_set = set(planned)
            progress.nodes.applied = sorted(applied)
            progress.nodes.attempted = sorted(planned_set)
            progress.nodes.pending_rollback = sorted(planned_set - applied)
            progress.nodes.phase = "failed"
            row.deployment_progress = progress
            flag_modified(row, "progress")
            row.status = DeploymentStatus.FAILED
            row.finished_at = datetime.now(UTC)
            existing = list(row.deployment_results)
            existing.append(
                DeploymentResult(
                    name="DeploymentError",
                    deploy_type=DeploymentResult.Type.GENERAL,
                    message=message,
                    status=DeploymentResult.Status.FAILED,
                    operation=DeploymentResult.Operation.UNKNOWN,
                ),
            )
            row.results = [r.model_dump() for r in existing]
            await session.commit()

    async def _mark_stale(
        self,
        deployment_id: str,
        snapshot: DeploymentSnapshot | None,
        actual_sha: str,
    ) -> None:
        async with session_context() as session:
            row = await session.get(Deployment, UUID(deployment_id))
            if row is None or row.is_terminal:
                return
            pinned_sha = snapshot.commit_sha if snapshot else "unknown"
            result = stale_result(deployment_id, snapshot, actual_sha) \
                if snapshot else DeploymentResult(
                    name=deployment_id,
                    deploy_type=DeploymentResult.Type.GENERAL,
                    status=DeploymentResult.Status.FAILED,
                    operation=DeploymentResult.Operation.UNKNOWN,
                    message=(
                        f"Deployment pinned to {pinned_sha} but remote moved "
                        f"to {actual_sha}."
                    ),
                )
            row.status = DeploymentStatus.STALE
            row.results = [result.model_dump()]
            row.finished_at = datetime.now(UTC)
            await session.commit()

    async def _handle_abort(
        self,
        deployment_id: str,
        signal: DeploymentFencedOrCanceled,
    ) -> None:
        async with session_context() as session:
            row = await session.get(Deployment, UUID(deployment_id))
            if row is None or row.is_terminal:
                return
            if signal.cancelled:
                row.status = DeploymentStatus.CANCELLED
            else:
                row.status = DeploymentStatus.FAILED
                existing = list(row.deployment_results)
                existing.append(
                    DeploymentResult(
                        name="FencedDeployment",
                        deploy_type=DeploymentResult.Type.GENERAL,
                        message=(
                            "A newer deployment took over the namespace lock; "
                            "this run was fenced off and stopped."
                        ),
                        status=DeploymentResult.Status.FAILED,
                        operation=DeploymentResult.Operation.UNKNOWN,
                    ),
                )
                row.results = [r.model_dump() for r in existing]
            row.finished_at = datetime.now(UTC)
            await session.commit()


async def cancel_deployment(deployment_id: str) -> Deployment:
    """Cancel a pending/running deployment.

    PENDING runs stop on their next load; RUNNING runs are interrupted
    cooperatively at their next progress checkpoint. The namespace lock is
    released only when this deployment is still its owner, so a newer owner
    is never unlocked by an old cancellation.
    """
    async with session_context() as session:
        row = await session.get(Deployment, UUID(deployment_id))
        if row is None:
            raise DJDoesNotExistException(
                message=f"Deployment {deployment_id} not found",
            )
        if row.is_terminal:
            raise DJDeploymentConflictException(
                message=(
                    f"Deployment {deployment_id} is already {row.status.value}; "
                    "it cannot be cancelled."
                ),
            )
        token = row.lock_token
        row.status = DeploymentStatus.CANCELLED
        row.finished_at = datetime.now(UTC)
        if token is not None:
            # Release in the same transaction: once CANCELLED is visible,
            # the lock is already gone.
            await deployment_locks.release_lock(
                session,
                row.namespace,
                row.uuid,
                token,
            )
        await session.commit()
        return row


async def resume_deployment(deployment_id: str) -> Deployment:
    """Resume a partially-failed pinned deployment from its persisted progress."""
    async with session_context() as session:
        row = await session.get(Deployment, UUID(deployment_id))
        if row is None:
            raise DJDoesNotExistException(
                message=f"Deployment {deployment_id} not found",
            )
        if not row.is_pinned:
            raise DJDeploymentConflictException(
                message="Only pinned git deployments can be resumed.",
            )
        if row.status != DeploymentStatus.FAILED:
            raise DJDeploymentConflictException(
                message=(
                    f"Deployment {deployment_id} is {row.status.value}; only "
                    "partially-failed deployments can be resumed. Create a new "
                    "deployment from the current commit to pick up newer "
                    "content."
                ),
            )
        # Flip back to RUNNING so the executor re-enters; it re-validates
        # drift and acquires the lock with a fresh fencing token.
        row.status = DeploymentStatus.RUNNING
        row.finished_at = None
        row.lock_token = None
        await session.commit()
    executor._schedule(deployment_id)
    return row


async def recover_pinned_deployments(limit: int = 100) -> list[str]:
    """
    Re-schedule pinned deployments left non-terminal by a process restart.

    Only rows with a snapshot qualify: local deploys and impact previews keep
    their previous (fire-and-forget) semantics. Each resumed run takes over
    its expired lock with a new fencing token and applies only nodes not
    already confirmed.
    """
    async with session_context() as session:
        rows = (
            await session.scalars(
                select(Deployment)
                .where(Deployment.snapshot.is_not(None))
                .where(
                    Deployment.status.in_(
                        [DeploymentStatus.PENDING, DeploymentStatus.RUNNING],
                    ),
                )
                .order_by(Deployment.created_at)
                .limit(limit),
            )
        ).all()
        ids = [str(row.uuid) for row in rows]
    for deployment_id in ids:
        logger.info("Recovering pinned deployment %s after restart", deployment_id)
        executor._schedule(deployment_id)
    return ids


executor = InProcessExecutor()


@router.post(
    "/deployments",
    name="Creates a bulk deployment",
    response_model=DeploymentInfo,
)
async def create_deployment(
    deployment_spec: DeploymentSpec,
    background_tasks: BackgroundTasks,
    request: Request,
    *,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
    query_service_client: QueryServiceClient = Depends(get_query_service_client),
    cache: Cache = Depends(get_cache),
    access_checker: AccessChecker = Depends(get_access_checker),
) -> DeploymentInfo:
    """
    This endpoint takes a deployment specification (namespace, nodes, tags), topologically
    sorts and validates the deployable objects, and deploys the nodes in parallel where
    possible. It returns a summary of the deployment.
    """
    access_checker.add_request(
        access.ResourceRequest(
            verb=access.ResourceAction.WRITE,
            access_object=access.Resource(
                resource_type=access.ResourceType.NAMESPACE,
                name=deployment_spec.namespace,
            ),
        ),
    )
    await access_checker.check(on_denied=AccessDenialMode.RAISE)

    # Git-managed namespaces (explicitly git_only, or a repo owner) require a
    # verified, pinned git deployment. A repo owner is any namespace that
    # directly owns a repo (github_repo_path set, no parent) — covers both git
    # roots and flat git-backed namespaces.
    namespace_obj = await NodeNamespace.get(
        session,
        deployment_spec.namespace,
        raise_if_not_exists=False,
    )
    managed_git = managed_namespace(namespace_obj)

    _t0 = time.monotonic()
    source_type = deployment_spec.source.type if deployment_spec.source else "local"
    _metrics_tags = {
        "namespace": deployment_spec.namespace,
        "source_type": str(source_type),
    }

    request_context = DeploymentContext(
        current_user=current_user,
        request=request,
        query_service_client=query_service_client,
        background_tasks=background_tasks,
        cache=cache,
    )

    if managed_git:
        deployment_id = await _create_pinned_deployment(
            session=session,
            spec=deployment_spec,
            context=request_context,
            idempotency_key=request.headers.get(IDEMPOTENCY_HEADER),
            namespace_obj=namespace_obj,
        )
    else:
        deployment_id = await executor.submit(
            spec=deployment_spec,
            context=request_context,
            idempotency_key=request.headers.get(IDEMPOTENCY_HEADER),
        )

    async with session_context() as read_session:
        deployment = await read_session.get(Deployment, UUID(deployment_id))
    status = deployment.status.value if deployment else "unknown"
    get_metrics_provider().timer(
        "dj.deployments.create_latency_ms",
        (time.monotonic() - _t0) * 1000,
        _metrics_tags,
    )
    get_metrics_provider().counter(
        "dj.deployments.create",
        tags={**_metrics_tags, "status": status},
    )
    return _deployment_info(deployment)


async def _create_pinned_deployment(
    *,
    session: AsyncSession,
    spec: DeploymentSpec,
    context: DeploymentContext,
    idempotency_key: str | None,
    namespace_obj: NodeNamespace | None,
) -> str:
    """
    Create a pinned deployment to a managed namespace:

    * branch-only requests are resolved server-side (branch HEAD fetched and
      frozen); CI requests carrying a commit keep their spec after the commit
      existence check;
    * the row and namespace lock are committed atomically (409 on contention);
    * the create-time impact snapshot is a dry run against live state;
    * the executor afterwards consumes only the persisted snapshot.
    """
    # Branch-only requests are resolved server-side only for a flat repo
    # owner; git_only branch namespaces keep the strict commit-sha gate.
    flat_owner = (
        namespace_obj is not None
        and namespace_obj.github_repo_path is not None
        and namespace_obj.parent_namespace is None
    )
    if is_server_pull(spec) and flat_owner:
        spec = await resolve_server_pull(session, spec)
    else:
        await _verify_git_deployment(session, spec, namespace_obj)

    deployment_id, created = await executor.create_pinned_row(
        spec,
        context,
        idempotency_key,
    )
    if not created:
        # Idempotent replay: return the original deployment as-is.
        return deployment_id

    try:
        async with session_context() as impact_session:
            impact = await build_impact_snapshot(impact_session, spec, context)
    except Exception as exc:
        # NFR-2: a create whose impact validation fails must not leave a
        # running deployment behind. Terminal failure releases nothing the
        # next deployment cannot take over (terminal holders are stealable).
        logger.warning(
            "Pinned deployment %s impact snapshot failed: %s",
            deployment_id,
            exc,
        )
        await executor._fail_pinned(
            deployment_id,
            f"create-time validation failed: {exc}",
            planned=[],
        )
        if isinstance(exc, DJInvalidInputException):
            raise
        raise DJInvalidInputException(message=str(exc)) from exc

    await executor.attach_snapshot_and_schedule(deployment_id, spec, impact)
    return deployment_id


def _source_for_row(deployment: Deployment):
    source_data = deployment.spec.get("source") if deployment.spec else None
    if source_data and source_data.get("type") == DeploymentSourceType.GIT:
        return GitDeploymentSource(**source_data)
    if source_data and source_data.get("type") == DeploymentSourceType.LOCAL:
        return LocalDeploymentSource(**source_data)
    return None


def _status_value(status) -> str:
    return status.value if hasattr(status, "value") else str(status)


def _deployment_info(deployment: Deployment) -> DeploymentInfo:
    return DeploymentInfo(
        uuid=str(deployment.uuid),
        namespace=deployment.namespace,
        status=_status_value(deployment.status),
        results=deployment.deployment_results,
        check_results=deployment.deployment_check_results,
        warnings=deployment.deployment_warnings,
        downstream_impacts=deployment.deployment_downstream_impacts,
        created_at=deployment.created_at.isoformat() if deployment.created_at else None,
        created_by=deployment.created_by.username if deployment.created_by else None,
        source=_source_for_row(deployment),
        snapshot=deployment.deployment_snapshot,
        progress=deployment.deployment_progress,
    )


@router.post("/deployments/{deployment_id}/cancel", response_model=DeploymentInfo)
async def cancel_deployment_endpoint(
    deployment_id: str,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
    access_checker: AccessChecker = Depends(get_access_checker),
) -> DeploymentInfo:
    """Cancel a pending/running deployment (idempotent ownership checks)."""
    try:
        deployment = await session.get(
            Deployment,
            UUID(deployment_id),
        )
    except ValueError:
        deployment = None
    # Authorize before revealing whether the deployment exists.
    namespace = deployment.namespace if deployment else deployment_id
    access_checker.add_namespace(
        namespace,
        access.ResourceAction.WRITE,
    )
    await access_checker.check(on_denied=AccessDenialMode.RAISE)
    if not deployment:
        raise DJDoesNotExistException(
            message=f"Deployment {deployment_id} not found",
        )
    row = await cancel_deployment(deployment_id)
    return _deployment_info(row)


@router.post("/deployments/{deployment_id}/resume", response_model=DeploymentInfo)
async def resume_deployment_endpoint(
    deployment_id: str,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
    access_checker: AccessChecker = Depends(get_access_checker),
) -> DeploymentInfo:
    """Resume a partially-failed pinned deployment, applying only pending nodes."""
    try:
        deployment = await session.get(
            Deployment,
            UUID(deployment_id),
        )
    except ValueError:
        deployment = None
    # Authorize before revealing whether the deployment exists.
    namespace = deployment.namespace if deployment else deployment_id
    access_checker.add_namespace(
        namespace,
        access.ResourceAction.WRITE,
    )
    await access_checker.check(on_denied=AccessDenialMode.RAISE)
    if not deployment:
        raise DJDoesNotExistException(
            message=f"Deployment {deployment_id} not found",
        )
    row = await resume_deployment(deployment_id)
    return _deployment_info(row)


@router.get("/deployments/{deployment_id}", response_model=DeploymentInfo)
async def get_deployment_status(
    deployment_id: str,
    session: AsyncSession = Depends(get_session),
) -> DeploymentInfo:
    deployment = await session.get(Deployment, deployment_id)
    if not deployment:
        raise DJDoesNotExistException(
            message=f"Deployment {deployment_id} not found",
        )  # pragma: no cover
    return _deployment_info(deployment)


@router.get("/deployments", response_model=list[DeploymentInfo])
async def list_deployments(  # pragma: no cover
    namespace: str | None = None,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
) -> list[DeploymentInfo]:
    statement = select(Deployment).order_by(Deployment.created_at.desc())
    if namespace:
        statement = statement.where(Deployment.namespace == namespace)
    statement = statement.limit(limit)
    deployments = (await session.execute(statement)).scalars().all()
    return [_deployment_info(deployment) for deployment in deployments]


PREFER_HEADER = "Prefer"
PREFER_RESPOND_ASYNC = "respond-async"


@router.post(
    "/deployments/impact",
    name="Preview deployment impact",
    response_model=DeploymentInfo,
)
async def preview_deployment_impact(
    deployment_spec: DeploymentSpec,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    *,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
    query_service_client: QueryServiceClient = Depends(get_query_service_client),
    cache: Cache = Depends(get_cache),
    access_checker: AccessChecker = Depends(get_access_checker),
) -> DeploymentInfo:
    """
    Submit a deployment impact preview for asynchronous processing.

    This endpoint is asynchronous: it submits a dry-run through the deployment
    orchestrator and returns immediately with a deployment ``uuid`` in
    PENDING/RUNNING status. Callers must poll ``GET /deployments/{uuid}`` until
    the status is terminal (SUCCESS/FAILED) to get the full impact analysis:
    nodes are validated and deployed into a database SAVEPOINT, downstream
    impact is computed via BFS, then the SAVEPOINT is rolled back so no changes
    are persisted.

    Requires a ``Prefer: respond-async`` header (RFC 7240), since this endpoint
    used to respond synchronously with the full result.
    """
    prefer = request.headers.get(PREFER_HEADER, "")
    preferences = [p.strip().split("=")[0] for p in prefer.split(",")]
    if PREFER_RESPOND_ASYNC not in preferences:
        raise DJClientUpgradeRequiredException(
            message=(
                "This endpoint is now asynchronous — upgrade datajunction-clients "
                "to a version that sends `Prefer: respond-async` and polls for "
                "/deployments/impact."
            ),
        )
    response.headers["Preference-Applied"] = PREFER_RESPOND_ASYNC

    access_checker.add_request(
        access.ResourceRequest(
            verb=access.ResourceAction.READ,
            access_object=access.Resource(
                resource_type=access.ResourceType.NAMESPACE,
                name=deployment_spec.namespace,
            ),
        ),
    )
    await access_checker.check(on_denied=AccessDenialMode.RAISE)

    deployment_id = await executor.submit(
        spec=deployment_spec,
        context=DeploymentContext(
            current_user=current_user,
            request=request,
            query_service_client=query_service_client,
            background_tasks=background_tasks,
            cache=cache,
        ),
        dry_run=True,
    )

    deployment = await session.get(Deployment, deployment_id)
    return DeploymentInfo(
        uuid=deployment_id,
        namespace=deployment.namespace,
        status=deployment.status.value,
        results=deployment.deployment_results,
        check_results=deployment.deployment_check_results,
        warnings=deployment.deployment_warnings,
        downstream_impacts=deployment.deployment_downstream_impacts,
    )
