"""
Deployment snapshotting: canonical spec digests, create-time impact snapshots,
server-side git pull resolution, and remote drift detection.

A pinned deployment freezes, at creation time:
  * the repository, branch and exact commit its content came from,
  * a SHA-256 digest of the canonicalized spec,
  * the dry-run impact analysis (downstream impacts, check results),

and the executor consumes only that frozen snapshot. If the tracked branch has
moved by the time execution starts, the snapshot is reported as stale rather
than silently applying a mix of two commits.
"""

import hashlib
import io
import json
import logging
import tarfile
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server.database.namespace import NodeNamespace
from datajunction_server.errors import DJInvalidInputException
from datajunction_server.internal.deployment.orchestrator import (
    DeploymentOrchestrator,
)
from datajunction_server.internal.deployment.utils import DeploymentContext
from datajunction_server.internal.git.github_service import (
    GitHubService,
    GitHubServiceError,
)
from datajunction_server.internal.namespaces import resolve_git_config
from datajunction_server.models.deployment import (
    DeploymentResult,
    DeploymentSnapshot,
    DeploymentSpec,
    GitDeploymentSource,
    SnapshotImpact,
    SnapshotPlannedChange,
)

logger = logging.getLogger(__name__)

_DIGEST_ALGORITHM = "sha256"


def normalize_repo_path(repository: str) -> str:
    """
    Normalize a deployment source repository to the ``owner/repo`` form stored
    in ``NodeNamespace.github_repo_path``. Accepts ``github.com/org/repo``,
    ``https://github.netflix.net/corp/repo.git``, ``git@host:corp/repo``, or an
    already-normalized ``corp/repo`` — and returns the trailing ``owner/repo``.
    """
    repo = repository.strip()
    for prefix in ("https://", "http://", "ssh://", "git@"):
        repo = repo.removeprefix(prefix)
    repo = repo.replace(":", "/")
    repo = repo.removesuffix(".git")
    parts = [segment for segment in repo.split("/") if segment]
    return "/".join(parts[-2:]) if len(parts) >= 2 else repo


def canonical_spec_json(spec: DeploymentSpec) -> str:
    """
    Deterministic JSON serialization of a spec, with keys sorted and no
    insignificant whitespace. Two specs that compare equal serialize equal.
    """
    return json.dumps(
        spec.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def spec_digest(spec: DeploymentSpec) -> str:
    """SHA-256 hex digest of the canonical spec (64 lowercase hex chars)."""
    return hashlib.sha256(
        canonical_spec_json(spec).encode("utf-8"),
    ).hexdigest()


def is_git_source(spec: DeploymentSpec) -> bool:
    return spec.source is not None and spec.source.type == "git"


def is_server_pull(spec: DeploymentSpec) -> bool:
    """
    A git deployment whose ref must be resolved server-side: a branch is named
    but no commit was pinned by the caller. CI callers always send the checked
    out commit_sha, so they never take this path.
    """
    return (
        is_git_source(spec)
        and bool(getattr(spec.source, "branch", None))
        and not getattr(spec.source, "commit_sha", None)
    )


def _planned_changes(results: list[DeploymentResult]) -> list[SnapshotPlannedChange]:
    planned = []
    for result in results:
        if result.operation == DeploymentResult.Operation.NOOP:
            continue
        planned.append(
            SnapshotPlannedChange(
                name=result.name,
                operation=result.operation.value
                if hasattr(result.operation, "value")
                else str(result.operation),
                deploy_type=result.deploy_type.value
                if hasattr(result.deploy_type, "value")
                else str(result.deploy_type),
                change_tier=result.change_tier,
            ),
        )
    return planned


async def build_impact_snapshot(
    session: AsyncSession,
    spec: DeploymentSpec,
    context: DeploymentContext,
) -> SnapshotImpact:
    """
    Run the deployment as a dry run and freeze its impact analysis.

    The orchestrator rolls back its SAVEPOINT, so no nodes, namespaces, tags
    or other resources persist — this is validation against live state without
    mutation.
    """
    snapshot_deployment_id = f"snapshot:{uuid.uuid4()}"
    orchestrator = DeploymentOrchestrator(
        deployment_id=snapshot_deployment_id,
        deployment_spec=spec,
        session=session,
        context=context,
        dry_run=True,
    )
    execute_result = await orchestrator.execute()
    return SnapshotImpact(
        results=list(execute_result.results),
        downstream_impacts=list(execute_result.downstream_impacts),
        check_results=list(execute_result.check_results),
        warnings=list(execute_result.warnings),
        planned=_planned_changes(list(execute_result.results)),
    )


async def fetch_git_spec_dict(
    github: GitHubService,
    repo_path: str,
    ref: str,
    git_path: str | None,
    namespace: str,
) -> dict:
    """
    Download the repo archive at ``ref`` and parse its YAML files into a
    DeploymentSpec-compatible dict (nodes, tags, hierarchies).

    Implements the DJ-pull model: the server fetches content directly, so
    user-submitted content is never trusted as the source of truth.
    """
    logger.info(
        "Fetching deployment spec from git: %s @ %s (path: %s)",
        repo_path,
        ref[:12] if len(ref) > 12 else ref,
        git_path or "/",
    )
    archive_bytes = await github.download_archive(
        repo_path=repo_path,
        branch=ref,
        format="tarball",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
            tar.extractall(tmpdir)
        extracted_dirs = list(Path(tmpdir).iterdir())
        if not extracted_dirs:
            raise DJInvalidInputException(
                message="Failed to extract archive - no directories found",
            )
        repo_dir = extracted_dirs[0]
        files_dir = (
            repo_dir / git_path.strip("/") if git_path else repo_dir
        )
        if not files_dir.exists():
            raise DJInvalidInputException(
                message=(
                    f"Path '{git_path}' not found in repository at ref '{ref}'"
                ),
            )

        nodes: list[dict] = []
        hierarchies: list[dict] = []
        tags: dict[str, dict] = {}
        for yaml_file in files_dir.rglob("*.yaml"):
            if yaml_file.name == "dj.yaml":
                try:
                    with open(yaml_file, encoding="utf-8") as f:
                        project_config = yaml.safe_load(f)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "Skipping invalid project config %s: %s",
                        yaml_file,
                        exc,
                    )
                    continue
                if isinstance(project_config, dict):
                    for tag in project_config.get("tags") or []:
                        if isinstance(tag, dict) and tag.get("name"):
                            tags[tag["name"]] = tag
                continue
            try:
                with open(yaml_file, encoding="utf-8") as f:
                    spec = yaml.safe_load(f)
                if not isinstance(spec, dict) or "name" not in spec:
                    continue
                if spec.get("type") == "hierarchy" and "node_type" not in spec:
                    hierarchies.append(spec)
                else:
                    nodes.append(spec)
            except Exception as exc:
                logger.warning("Skipping invalid YAML file %s: %s", yaml_file, exc)
                continue

    logger.info(
        "Fetched %d node specs, %d hierarchy specs and %d tag specs from git: %s @ %s",
        len(nodes),
        len(hierarchies),
        len(tags),
        repo_path,
        ref[:12] if len(ref) > 12 else ref,
    )
    return {
        "namespace": namespace,
        "nodes": nodes,
        "tags": list(tags.values()),
        "hierarchies": hierarchies,
    }


async def resolve_server_pull(
    session: AsyncSession,
    spec: DeploymentSpec,
) -> DeploymentSpec:
    """
    Resolve a branch-only git deployment server-side: pin the branch HEAD to an
    exact commit, fetch the archive at that commit, and return a fully-hydrated
    DeploymentSpec with the pinned source.

    The namespace's configured repository is the only repository honored; an
    explicit source repository must normalize to the same value.
    """
    if not is_server_pull(spec):
        return spec
    source: GitDeploymentSource = spec.source  # type: ignore[assignment]

    configured_repo, git_path, configured_branch = await resolve_git_config(
        session,
        spec.namespace,
    )
    if not configured_repo:
        raise DJInvalidInputException(
            message=(
                f"Namespace '{spec.namespace}' has no github_repo_path configured; "
                "a branch-only git deployment cannot be resolved server-side."
            ),
        )
    branch = source.branch or configured_branch
    if not branch:
        raise DJInvalidInputException(
            message=(
                f"Namespace '{spec.namespace}' has no git branch to resolve "
                "(neither source.branch nor namespace git_branch is set)."
            ),
        )
    if source.repository:
        sent = normalize_repo_path(source.repository)
        if sent != configured_repo:
            raise DJInvalidInputException(
                message=(
                    f"source.repository '{sent}' does not match the repository "
                    f"configured for namespace '{spec.namespace}': "
                    f"'{configured_repo}'."
                ),
            )

    github = GitHubService()
    try:
        commit_sha = await github.resolve_ref_to_sha(configured_repo, branch)
        author_name = None
        author_email = None
        try:
            author_name, author_email = await github.get_commit_author(
                configured_repo,
                commit_sha,
            )
        except GitHubServiceError:
            logger.warning(
                "Could not fetch commit author for %s @ %s; history will use the "
                "API caller",
                configured_repo,
                commit_sha[:12],
            )
        spec_dict = await fetch_git_spec_dict(
            github,
            configured_repo,
            commit_sha,
            git_path,
            spec.namespace,
        )
    except GitHubServiceError as exc:
        raise DJInvalidInputException(message=f"Failed to resolve git ref: {exc}") from exc

    if not spec_dict.get("nodes") and not spec_dict.get("hierarchies"):
        raise DJInvalidInputException(
            message=(
                f"No node or hierarchy definitions found in repository "
                f"'{configured_repo}' at branch '{branch}'"
                + (f" in path '{git_path}'" if git_path else "")
            ),
        )

    spec_dict["source"] = {
        "type": "git",
        "repository": configured_repo,
        "branch": branch,
        "commit_sha": commit_sha,
        "commit_author_name": author_name,
        "commit_author_email": author_email,
    }
    return DeploymentSpec(**spec_dict)


def build_deployment_snapshot(
    spec: DeploymentSpec,
    impact: SnapshotImpact | None = None,
    pinned_at: datetime | None = None,
) -> DeploymentSnapshot:
    """
    Assemble the immutable snapshot for a git deployment. Callers must ensure
    the git source carries a pinned commit_sha (after verify/resolve).
    """
    source = spec.source
    if source is None or source.type != "git" or not source.commit_sha:
        raise ValueError("Cannot snapshot a deployment without a pinned git commit")
    return DeploymentSnapshot(
        repository=normalize_repo_path(source.repository),
        branch=source.branch,
        commit_sha=source.commit_sha,
        commit_author_name=source.commit_author_name,
        commit_author_email=source.commit_author_email,
        spec_digest=spec_digest(spec),
        impact=impact,
        pinned_at=pinned_at or datetime.now(UTC),
    )


async def check_remote_drift(
    source: GitDeploymentSource,
    github: GitHubService | None = None,
) -> str | None:
    """
    Resolve the tracked branch on the remote.

    Returns the remote HEAD sha when it differs from the pinned commit
    (deployment is stale), or ``None`` when it still matches. A commit-only
    source (no branch) cannot be checked and always returns ``None``.

    ``github`` may be injected (the API executor builds the service so callers
    patch one symbol). Raises ``GitHubServiceError`` when the remote is
    unreachable; that is a transient failure, not staleness.
    """
    if not source.branch or not source.repository:
        return None
    github = github or GitHubService()
    actual = await github.resolve_ref_to_sha(
        normalize_repo_path(source.repository),
        source.branch,
    )
    if actual == source.commit_sha:
        return None
    return actual


def managed_namespace(namespace_obj: NodeNamespace | None) -> bool:
    """
    Whether a namespace enforces the managed-git deployment contract:
    explicitly git_only, or a flat repo owner (github_repo_path set, no parent).
    """
    if namespace_obj is None:
        return False
    return bool(
        namespace_obj.git_only
        or (
            namespace_obj.github_repo_path is not None
            and namespace_obj.parent_namespace is None
        ),
    )


def stale_result(
    deployment_id: str,
    snapshot: DeploymentSnapshot,
    actual_sha: str,
) -> DeploymentResult:
    """The single result recorded on a deployment that went stale."""
    return DeploymentResult(
        name=deployment_id,
        deploy_type=DeploymentResult.Type.GENERAL,
        status=DeploymentResult.Status.FAILED,
        operation=DeploymentResult.Operation.UNKNOWN,
        message=(
            f"Deployment snapshot is stale: branch '{snapshot.branch}' of "
            f"'{snapshot.repository}' now points at {actual_sha}, but this "
            f"deployment was pinned to {snapshot.commit_sha}. Create a new "
            "deployment from the current commit."
        ),
    )


__all__ = [
    "build_deployment_snapshot",
    "build_impact_snapshot",
    "canonical_spec_json",
    "check_remote_drift",
    "fetch_git_spec_dict",
    "is_git_source",
    "is_server_pull",
    "managed_namespace",
    "normalize_repo_path",
    "resolve_server_pull",
    "spec_digest",
    "stale_result",
]
