"""
Unit tests for the orchestrator's idempotent reconciliation and recoverable
progress hooks (pinned deployments). These use hand-built plans and need no
database container.
"""

from types import SimpleNamespace

import pytest

from datajunction_server.database.deployment import Deployment
from datajunction_server.errors import DJException
from datajunction_server.internal.deployment.orchestrator import (
    DeploymentFencedOrCanceled,
    DeploymentOrchestrator,
    DeploymentPlan,
)
from datajunction_server.models.deployment import (
    AuthzProgress,
    DeploymentProgress,
    DeploymentResult,
    DeploymentSpec,
    LockProgress,
    NodeProgress,
    TransformSpec,
)


def _orchestrator(
    *,
    dry_run: bool = False,
    deployment_row=None,
    initial_progress=None,
    checkpoint=None,
):
    spec = DeploymentSpec(namespace="ns")
    context = SimpleNamespace(current_user=object())
    orch = DeploymentOrchestrator(
        deployment_spec=spec,
        deployment_id="dep-1",
        session=SimpleNamespace(),  # never touched by the units under test
        context=context,
        dry_run=dry_run,
        initial_progress=initial_progress,
        deployment_row=deployment_row,
        progress_checkpoint=checkpoint,
    )
    return orch


def _plan(specs, existing=None) -> DeploymentPlan:
    return DeploymentPlan(
        to_deploy=list(specs),
        to_skip=[],
        to_delete=[],
        existing_specs=dict(existing or {}),
        node_graph={},
        external_deps=set(),
    )


def _transform(name="a", query="SELECT 1") -> TransformSpec:
    spec = TransformSpec(name=name, query=query)
    spec.namespace = "ns"
    return spec


async def test_reconciliation_is_noop_without_deployment_row():
    spec = _transform()
    plan = _plan([spec], existing={"ns.a": spec})
    orch = _orchestrator()
    await orch._reconcile_applied_nodes(plan)
    assert plan.to_deploy == [spec]
    assert orch.planned_names == []


async def test_reconciliation_is_noop_in_dry_run():
    spec = _transform()
    plan = _plan([spec], existing={"ns.a": spec})
    orch = _orchestrator(dry_run=True, deployment_row=SimpleNamespace(progress=None))
    await orch._reconcile_applied_nodes(plan)
    assert plan.to_deploy == [spec]


async def test_reconciliation_skips_nodes_already_at_target():
    spec = _transform()
    plan = _plan([spec], existing={"ns.a": _transform(query="SELECT 1")})
    row = SimpleNamespace(progress=None)
    orch = _orchestrator(deployment_row=row)
    await orch._reconcile_applied_nodes(plan)
    assert plan.to_deploy == []
    assert orch.planned_names == []
    assert orch.reconciled_names == {"ns.a"}
    result = orch.deployed_results[-1]
    assert result.status == DeploymentResult.Status.SKIPPED
    assert "already applied" in result.message


async def test_reconciliation_keeps_nodes_with_pending_changes():
    spec = _transform(query="SELECT 2")
    plan = _plan([spec], existing={"ns.a": _transform(query="SELECT 1")})
    orch = _orchestrator(deployment_row=SimpleNamespace(progress=None))
    await orch._reconcile_applied_nodes(plan)
    assert [s.rendered_name for s in plan.to_deploy] == ["ns.a"]
    assert orch.planned_names == ["ns.a"]
    assert not orch.reconciled_names


async def test_reconciliation_rejects_applied_node_that_drifted():
    # Progress claims applied but live content differs: refuse a third state.
    prior = DeploymentProgress(
        nodes=NodeProgress(phase="failed", applied=["ns.a"]),
    )
    spec = _transform(query="SELECT 999")
    plan = _plan([spec], existing={"ns.a": _transform(query="SELECT 1")})
    orch = _orchestrator(
        deployment_row=SimpleNamespace(progress=None),
        initial_progress=prior,
    )
    with pytest.raises(DJException, match="cannot resume"):
        await orch._reconcile_applied_nodes(plan)


async def test_checkpoint_honors_fence_and_cancel_verdicts():
    async def fenced_checkpoint(_progress):
        return "fenced"

    async def cancelled_checkpoint(_progress):
        return "cancelled"

    orch_fenced = _orchestrator(
        deployment_row=SimpleNamespace(progress=None),
        initial_progress=DeploymentProgress(),
        checkpoint=fenced_checkpoint,
    )
    orch_fenced.planned_names = ["ns.a"]
    with pytest.raises(DeploymentFencedOrCanceled) as exc:
        await orch_fenced._checkpoint_phase("planned")
    assert not exc.value.cancelled

    orch_cancel = _orchestrator(
        deployment_row=SimpleNamespace(progress=None),
        initial_progress=DeploymentProgress(),
        checkpoint=cancelled_checkpoint,
    )
    with pytest.raises(DeploymentFencedOrCanceled) as exc:
        await orch_cancel._checkpoint_phase("planned")
    assert exc.value.cancelled


async def test_checkpoint_persists_planned_pending_names():
    seen = {}

    async def capture(progress):
        seen["progress"] = progress
        return "ok"

    orch = _orchestrator(
        deployment_row=SimpleNamespace(progress=None),
        initial_progress=DeploymentProgress(
            lock=LockProgress(fencing_token=7),
            nodes=NodeProgress(phase="applying", applied=["ns.old"]),
        ),
        checkpoint=capture,
    )
    orch.planned_names = ["ns.a", "ns.b"]
    await orch._checkpoint_phase("planned")
    progress = seen["progress"]
    # Lock progress preserved; prior applied names not claimed pre-commit; new
    # planned names are pending rollback until confirmed.
    assert progress.lock.fencing_token == 7
    assert progress.nodes.applied == ["ns.old"]
    assert progress.nodes.phase == "planned"
    assert set(progress.nodes.pending_rollback) == {"ns.a", "ns.b"}


def test_precommit_progress_is_atomic_with_revisions():
    row = Deployment(spec={}, results={})  # transient ORM instance, never flushed
    orch = _orchestrator(
        deployment_row=row,
        initial_progress=DeploymentProgress(
            lock=LockProgress(fencing_token=3),
            nodes=NodeProgress(phase="applying", applied=["ns.old"]),
        ),
    )
    orch.planned_names = ["ns.a", "ns.b"]
    orch.reconciled_names.add("ns.b")
    orch.deployed_results.append(
        DeploymentResult(
            name="ns.a",
            deploy_type=DeploymentResult.Type.NODE,
            status=DeploymentResult.Status.SUCCESS,
            operation=DeploymentResult.Operation.CREATE,
        ),
    )
    orch._write_progress_precommit()
    # Folded into the same-txn progress payload.
    assert row.progress["nodes"]["phase"] == "applying"
    assert set(row.progress["nodes"]["applied"]) == {"ns.old", "ns.a", "ns.b"}
    assert row.progress["nodes"]["pending_rollback"] == []
    assert row.progress["lock"]["fencing_token"] == 3


def test_authorization_digest_stable_and_sensitive():
    orch = _orchestrator()
    plan_a = _plan([_transform("a")])
    plan_b = _plan([_transform("a")])
    plan_c = _plan([_transform("a"), _transform("c", query="SELECT 3")])
    assert orch._plan_authorization_digest(plan_a) == orch._plan_authorization_digest(
        plan_b,
    )
    assert orch._plan_authorization_digest(plan_a) != orch._plan_authorization_digest(
        plan_c,
    )


async def test_authorization_skipped_for_resumed_identical_plan():
    digest_seed_plan = _plan([_transform("a")])
    digest = _orchestrator()._plan_authorization_digest(digest_seed_plan)
    prior = DeploymentProgress(
        authz=AuthzProgress(state="done", plan_digest=digest),
    )
    orch = _orchestrator(
        deployment_row=SimpleNamespace(progress=None),
        initial_progress=prior,
    )
    # Would raise if re-authorized: the stubbed session cannot run checks.
    await orch._authorize_with_progress(digest_seed_plan)
